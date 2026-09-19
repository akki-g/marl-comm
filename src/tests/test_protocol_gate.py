"""Protocol authorizations in isolated fixture trees; never approve repository runs."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from commstudy.experiments.config import load_experiment_spec, scientific_config_sha256
from commstudy.experiments.protocols import (
    APPROVAL_FIELDS,
    ProtocolGateError,
    actual_model_contract,
    content_sha256,
    load_protocol,
    protocol_spec,
    validate_protocol_launch,
)
from commstudy.experiments.provenance import source_fingerprint
from commstudy.experiments.sweeps import (
    expand_suite_config,
    read_manifest,
    validate_plan,
    write_manifest,
)


@pytest.fixture
def protocol_case(tmp_path, config_root, monkeypatch):
    root = tmp_path / 'isolated_protocol_repository'
    shutil.copytree(config_root, root / 'configs')
    source = root / 'src' / 'commstudy' / 'fixture.py'
    source.parent.mkdir(parents=True)
    source.write_text('PROTOCOL_FIXTURE = 1\n')
    path = root / 'configs' / 'protocols' / 'pcp_corrected_v1.yaml'
    protocol = load_protocol(path)
    protocol['status'] = 'frozen'
    protocol['runtime'].update(device='cpu', threads=1, python=platform.python_version())
    protocol['runtime']['packages'] = {
        name: importlib.metadata.version(name) for name in protocol['runtime']['packages']
    }
    for key in ('sampling_device', 'train_device', 'buffer_device'):
        protocol['base_spec']['experiment'][key] = 'cpu'
    for stage in ('confirmation', 'comparison', 'ablation'):
        protocol['selection'][f'{stage}_seeds'] = [0]
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(key, '1')

    def save():
        OmegaConf.save(OmegaConf.create(protocol), path)

    save()
    case = SimpleNamespace(root=root, path=path, protocol=protocol, source=source, save=save)

    def approve(stage='comparison', *, validated_factors=()):
        evidence_dir = root / 'evidence'
        evidence_dir.mkdir(exist_ok=True)
        kind = {'confirmation': 'candidate_review',
                'comparison': 'corrected_protocol_confirmation',
                'ablation': 'main_comparison_review'}[stage]
        evidence = evidence_dir / f'{stage}_test_evidence.json'
        if stage == 'confirmation':
            evidence.write_text(json.dumps({'fixture_only': True, 'stage': stage}))
        else:
            prerequisite = 'confirmation' if stage == 'comparison' else 'comparison'
            confirmed_path = evidence_dir / f'{stage}_confirmed_protocol.yaml'
            shutil.copyfile(path, confirmed_path)
            confirmed = load_protocol(confirmed_path)
            common = {'protocol_id': confirmed['protocol_id'],
                      'protocol_sha256': content_sha256(confirmed),
                      'source_sha256': source_fingerprint(root)}
            seeds = confirmed['selection'][f'{prerequisite}_seeds']
            models = (
                ['pcp_comm_identity'] if prerequisite == 'confirmation' else confirmed['methods']
            )
            conditions = confirmed.get('launch_scope', {}).get(prerequisite, [
                {'model': model, 'ablation': 'main', 'ablation_value': None} for model in models
            ])
            report_path = evidence_dir / f'{stage}_report.json'
            report_path.write_text(json.dumps({
                **common, 'checks_passed': True, 'issues': [], 'expected_seeds': seeds,
                'runs': [{**condition, 'seed': seed, 'validation_passed': True}
                         for condition in conditions for seed in seeds],
            }))
            review_path = evidence_dir / f'{stage}_review.json'
            report_ref = _reference(report_path)
            review_path.write_text(json.dumps({
                **common, 'overall_verdict': 'CONFIRMED', 'completion_gate_passed': True,
                'stage': f'{prerequisite}_review_only', 'reviewer': 'isolated test fixture',
                f'{prerequisite}_report_sha256': report_ref['sha256'],
                'evidence': {'report': report_ref},
            }))
            evidence.write_text(json.dumps({
                'schema_version': 1, 'decision': 'confirmed',
                'authorized_protocol_sha256': content_sha256(load_protocol(path)),
                'authorized_stage': stage, 'confirmed_protocol': _reference(confirmed_path),
                'review': _reference(review_path), 'report': report_ref, 'source_bridge': None,
            }))
        payload = {
            'schema_version': 1, 'protocol_id': protocol['protocol_id'],
            'protocol_sha256': content_sha256(load_protocol(path)),
            'source_sha256': source_fingerprint(root), 'stage': stage,
            'decision': 'approved', 'reviewer': 'isolated test fixture',
            'rationale': 'Test the gate mechanism; this authorizes no repository experiment.',
            'evidence': [{'path': str(evidence),
                          'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest(),
                          'kind': kind}],
            'validated_factors': list(validated_factors),
        }
        assert set(payload) == APPROVAL_FIELDS
        approval = root / 'configs' / 'protocols' / 'approvals' / (
            f"{protocol['protocol_id']}_{stage}.json"
        )
        approval.parent.mkdir(exist_ok=True)
        approval.write_text(json.dumps(payload))
        return approval, evidence, payload

    def bind(stage='comparison', *, model='pcp_comm_identity', seed=0,
             ablation='main', ablation_value=None, approve_now=True):
        approval = root / 'configs' / 'protocols' / 'approvals' / (
            f"{protocol['protocol_id']}_{stage}.json"
        )
        if approve_now:
            approve(stage)
        return {
            'path': str(path), 'sha256': content_sha256(load_protocol(path)),
            'source_sha256': source_fingerprint(root), 'approval': str(approval),
            'stage': stage, 'model': model, 'seed': seed,
            'ablation': ablation, 'ablation_value': ablation_value,
        }

    case.approve = approve
    case.bind = bind
    case.spec = lambda **kwargs: protocol_spec(load_protocol(path), **{
        'model': 'pcp_comm_identity', 'seed': 0, **kwargs,
    })
    yield case
    torch.set_num_threads(original_threads)


def _reference(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _update_certificate(binding, change):
    """Rehash deliberately invalid content, so tests exercise semantics, not stale bytes."""
    approval_path = Path(binding['approval'])
    approval = json.loads(approval_path.read_text())
    certificate_path = Path(approval['evidence'][0]['path'])
    certificate = json.loads(certificate_path.read_text())
    review_path = Path(certificate['review']['path'])
    report_path = Path(certificate['report']['path'])
    review, report = json.loads(review_path.read_text()), json.loads(report_path.read_text())
    change(certificate, review, report)
    report_path.write_text(json.dumps(report))
    certificate['report'] = _reference(report_path)
    review['confirmation_report_sha256'] = certificate['report']['sha256']
    review['evidence']['report'] = certificate['report']
    review_path.write_text(json.dumps(review))
    certificate['review'] = _reference(review_path)
    certificate_path.write_text(json.dumps(certificate))
    approval['evidence'][0].update(_reference(certificate_path))
    approval_path.write_text(json.dumps(approval))


@pytest.mark.parametrize('bad_evidence', [
    'unconfirmed_review', 'incomplete_packet', 'invalid_report', 'report_issues',
    'missing_seed', 'duplicate_seed', 'invalid_run', 'wrong_review_protocol',
    'wrong_stage', 'wrong_target', 'changed_source', 'lost_packet_artifact',
])
def test_kind_and_valid_hash_cannot_promote_failed_unrelated_or_incomplete_packet(
    protocol_case, bad_evidence,
):
    case = protocol_case
    binding = case.bind()

    def corrupt(certificate, review, report):
        if bad_evidence == 'unconfirmed_review':
            review['overall_verdict'] = 'NOT CONFIRMED'
        elif bad_evidence == 'incomplete_packet':
            review['completion_gate_passed'] = False
        elif bad_evidence == 'invalid_report':
            report['checks_passed'] = 1  # truthy is not a validated boolean
        elif bad_evidence == 'report_issues':
            report['issues'] = ['mandatory 480k checkpoint missing']
        elif bad_evidence == 'missing_seed':
            report['runs'] = []
        elif bad_evidence == 'duplicate_seed':
            report['runs'].append(copy.deepcopy(report['runs'][0]))
        elif bad_evidence == 'invalid_run':
            report['runs'][0]['validation_passed'] = False
        elif bad_evidence == 'wrong_review_protocol':
            review['protocol_sha256'] = '0' * 64
        elif bad_evidence == 'wrong_stage':
            certificate['authorized_stage'] = 'ablation'
        elif bad_evidence == 'wrong_target':
            certificate['authorized_protocol_sha256'] = '0' * 64
        elif bad_evidence == 'changed_source':
            review['source_sha256'] = report['source_sha256'] = '0' * 64
        else:
            review['evidence']['required_checkpoint'] = {
                'path': str(case.root / 'missing_checkpoint.pt'), 'sha256': '0' * 64,
            }

    _update_certificate(binding, corrupt)
    with pytest.raises((ValueError, TypeError)):
        validate_protocol_launch(case.spec(), binding, repo_root=case.root)


def test_bare_kind_label_is_not_a_promotion_certificate(protocol_case):
    case = protocol_case
    binding = case.bind()
    path = Path(binding['approval'])
    approval = json.loads(path.read_text())
    evidence = Path(approval['evidence'][0]['path'])
    evidence.write_text(json.dumps({'kind': 'corrected_protocol_confirmation', 'passed': True}))
    approval['evidence'][0].update(_reference(evidence))
    path.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match='Promotion certificate'):
        validate_protocol_launch(case.spec(), binding, repo_root=case.root)


def _controlled_comparison(case):
    models = ['pcp_comm_identity', 'pcp_comm_attention']
    case.protocol['factors']['visibility'] = {
        'models': models,
        'values': {
            value: {
                model: {
                    'task_config': {**copy.deepcopy(case.protocol['base_spec']['task_config']),
                                    'params': {
                                        **case.protocol['base_spec']['task_config']['params'],
                                        'predator_sensing_radius': radius,
                                    }},
                    'model_config': copy.deepcopy(case.protocol['methods'][model]),
                }
                for model in models
            }
            for value, radius in [('radius1', 1.0), ('global', None)]
        },
    }
    case.protocol['launch_scope'] = {'comparison': [
        {'model': model, 'ablation': 'visibility', 'ablation_value': value}
        for model in models for value in ('radius1', 'global')
    ]}
    case.save()


def test_declared_comparison_factor_freezes_complete_task_and_model_without_relabeling(
    protocol_case,
):
    case = protocol_case
    _controlled_comparison(case)
    for value, radius in [('radius1', 1.0), ('global', None)]:
        factors = {'model': 'pcp_comm_identity', 'ablation': 'visibility', 'ablation_value': value}
        spec = case.spec(**factors)
        binding = case.bind('comparison', **factors)
        result = validate_protocol_launch(spec, binding, repo_root=case.root)
        assert result['stage'] == 'comparison'
        assert spec.task_config['params']['predator_sensing_radius'] == radius
        assert spec.model_config == case.protocol['methods']['pcp_comm_identity']
        assert spec.experiment['max_n_frames'] == 600000


@pytest.mark.parametrize('stage,factors', [
    ('comparison', {'model': 'pcp_comm_identity'}),
    ('comparison', {'model': 'pcp_comm_graph'}),
    ('comparison', {'model': 'pcp_comm_attention', 'ablation': 'message_dim',
                    'ablation_value': '8'}),
    ('ablation', {'model': 'pcp_comm_identity', 'ablation': 'visibility',
                  'ablation_value': 'global'}),
    ('confirmation', {'model': 'pcp_comm_identity'}),
])
def test_controlled_scope_rejects_undeclared_methods_conditions_and_stages(
    protocol_case, stage, factors,
):
    case = protocol_case
    _controlled_comparison(case)
    binding = case.bind(stage, **factors)
    with pytest.raises(ProtocolGateError, match='launch_scope'):
        validate_protocol_launch(case.spec(**factors), binding, repo_root=case.root)


@pytest.mark.parametrize('malformation', ['unknown_stage', 'duplicate', 'empty', 'bad_model',
                                        'bad_factor', 'extra_key', 'wrong_type'])
def test_controlled_scope_rejects_malformed_or_unresolvable_conditions(protocol_case, malformation):
    case = protocol_case
    _controlled_comparison(case)
    scope = case.protocol['launch_scope']
    rows = scope['comparison']
    if malformation == 'unknown_stage':
        scope['free_training'] = rows
    elif malformation == 'duplicate':
        rows.append(copy.deepcopy(rows[0]))
    elif malformation == 'empty':
        scope['comparison'] = []
    elif malformation == 'bad_model':
        rows[0]['model'] = 'undeclared_model'
    elif malformation == 'bad_factor':
        rows[0]['ablation_value'] = 'unreviewed_radius'
    elif malformation == 'extra_key':
        rows[0]['skip_verification'] = True
    else:
        rows[0]['ablation_value'] = 1
    case.save()
    with pytest.raises((ValueError, TypeError)):
        load_protocol(case.path)


def test_controlled_comparison_manifest_retains_conditions_and_exact_seed_matrix(protocol_case):
    case = protocol_case
    _controlled_comparison(case)
    case.protocol['selection']['comparison_seeds'] = [20, 21, 22, 23, 24]
    case.save()
    suite = {
        'suite_id': 'controlled_fixture', 'protocol': str(case.path.relative_to(case.root)),
        'stage': 'comparison', 'seeds': [20, 21, 22, 23, 24],
        'models': ['pcp_comm_identity', 'pcp_comm_attention'],
        'runs': [{'ablation': 'visibility', 'ablation_value': value}
                 for value in ('radius1', 'global')],
    }
    plans = expand_suite_config(suite, repo_root=case.root)
    assert len(plans) == 20
    assert {(plan.model, plan.ablation_value, plan.seed) for plan in plans} == {
        (model, visibility, seed) for model in suite['models']
        for visibility in ('radius1', 'global') for seed in suite['seeds']
    }
    assert all(plan.protocol['stage'] == 'comparison' for plan in plans)
    assert all(plan.max_n_frames == 600000 for plan in plans)
    # Planning a candidate is permitted; execution still requires its promotion certificate.
    with pytest.raises(ProtocolGateError, match='missing'):
        validate_plan(plans[0], config_root=case.root / 'configs', repo_root=case.root)


@pytest.mark.parametrize('corruption', [None, 'unapproved', 'wrong_source_pair', 'wrong_model',
                                      'different_numeric_metric', 'nonfinite', 'short_prefix',
                                      'wrong_count', 'different_seed', 'different_spec',
                                      'different_runtime', 'changed_metric_file',
                                      'wrong_task', 'wrong_algorithm'])
def test_source_change_requires_approved_exact_bounded_identity_metric_evidence(
    protocol_case, corruption,
):
    case = protocol_case
    binding = case.bind()
    old_source, current_source = '0' * 64, source_fingerprint(case.root)
    bridge_dir = case.root / 'evidence' / 'bridge'
    bridge_dir.mkdir()
    equivalence = {
        'schema_version': 1, 'checks_passed': True,
        'previous_source_sha256': old_source, 'source_sha256': current_source,
        'model': 'pcp_comm_identity', 'frames': 6000, 'seed': 0, 'compared_scalar_count': 2,
    }
    for label, source in [('previous', old_source), ('current', current_source)]:
        directory = bridge_dir / label
        directory.mkdir()
        metadata_path, metrics_path = directory / 'metadata.json', directory / 'metrics.csv'
        metadata = {
            'source_sha256': source, 'seed': 0, 'model': 'pcp_comm_identity',
            'task': 'vmas_predator_capture_prey', 'algorithm': 'mappo',
            'scientific_config_sha256': 'a' * 64,
            'versions': {'torch': 'fixture'}, 'runtime': {'torch_num_threads': 1, 'device': 'cpu'},
        }
        if label == 'current':
            if corruption == 'different_seed':
                metadata['seed'] = 1
            elif corruption == 'different_spec':
                metadata['scientific_config_sha256'] = 'b' * 64
            elif corruption == 'different_runtime':
                metadata['runtime']['torch_num_threads'] = 2
            elif corruption == 'wrong_task':
                metadata['task'] = 'vmas_simple_spread'
            elif corruption == 'wrong_algorithm':
                metadata['algorithm'] = 'qmix'
        metadata_path.write_text(json.dumps(metadata))
        # Timestamp and wall time differ; replay metrics must match exactly too.
        value = '2.0'
        if label == 'current' and corruption == 'different_numeric_metric':
            value = '2.0000000000000004'
        elif label == 'current' and corruption == 'nonfinite':
            value = 'nan'
        frame = 5999 if corruption == 'short_prefix' else 6000
        metrics_path.write_text(
            'timestamp,frames,iteration,phase,group,metric,sample,value\n'
            f'{label},{frame},0,train,adversary,loss_critic,,{value}\n'
            f'{label},{frame},0,collection,adversary,replay_log_prob_max_abs_error,,0\n'
            f'{label},{frame},0,train,,wall_time_seconds,,{1 if label == "previous" else 2}\n'
        )
        equivalence[f'{label}_metadata'] = _reference(metadata_path)
        equivalence[f'{label}_metrics'] = _reference(metrics_path)
    if corruption == 'wrong_count':
        equivalence['compared_scalar_count'] = 3
    elif corruption == 'wrong_model':
        equivalence['model'] = 'pcp_actor'
    equivalence_path = bridge_dir / 'equivalence.json'
    equivalence_path.write_text(json.dumps(equivalence))
    bridge = {
        'schema_version': 1, 'approved': corruption != 'unapproved',
        'reviewer': 'isolated test fixture',
        'rationale': 'Fixture source bridge; no real launches.',
        'previous_source_sha256': old_source, 'source_sha256': current_source,
        'identity_equivalence': _reference(equivalence_path),
    }
    if corruption == 'wrong_source_pair':
        bridge['previous_source_sha256'] = 'f' * 64
    bridge_path = bridge_dir / 'review.json'
    bridge_path.write_text(json.dumps(bridge))

    def use_bridge(certificate, review, report):
        review['source_sha256'] = report['source_sha256'] = old_source
        certificate['source_bridge'] = _reference(bridge_path)

    _update_certificate(binding, use_bridge)
    if corruption == 'changed_metric_file':
        Path(equivalence['current_metrics']['path']).write_text('changed after review')
    if corruption is None:
        result = validate_protocol_launch(case.spec(), binding, repo_root=case.root)
        assert result['stage'] == 'comparison'
    else:
        with pytest.raises((ValueError, TypeError)):
            validate_protocol_launch(case.spec(), binding, repo_root=case.root)


@pytest.mark.parametrize('path,value', [('seed', 999), ('experiment.max_n_frames', 1200000),
                                     ('algorithm_config.params.entropy_coef', 0.2)])
def test_task_model_factor_cannot_silently_override_seed_or_training_protocol(
    protocol_case, path, value,
):
    case = protocol_case
    _controlled_comparison(case)
    case.protocol['factors']['visibility']['values']['global']['pcp_comm_identity'][path] = value
    case.save()
    with pytest.raises(ValueError, match='task_config/model_config only'):
        load_protocol(case.path)


@pytest.mark.parametrize('change', ['gamma', 'horizon', 'algorithm', 'critic', 'task',
                                  'actor', 'runtime', 'contract'])
def test_promotion_cannot_reuse_confirmation_for_a_changed_baseline(protocol_case, change):
    case = protocol_case
    binding = case.bind()
    base = case.protocol['base_spec']
    if change == 'gamma':
        base['experiment']['gamma'] = 0.9
    elif change == 'horizon':
        base['experiment']['max_n_frames'] = 1200000
    elif change == 'algorithm':
        base['algorithm_config']['params']['lmbda'] = 0.8
    elif change == 'critic':
        base['critic_model']['groups']['adversary']['params']['num_cells'] = [256, 128]
    elif change == 'task':
        base['task_config']['params']['num_landmarks'] = 3
    elif change == 'actor':
        base['model_config']['groups']['adversary']['params']['hidden_dim'] = 256
    elif change == 'runtime':
        case.protocol['runtime']['threads'] = 2
    else:
        case.protocol['contracts']['training_health_protocol'] = 'different_semantics'
    case.save()
    binding['sha256'] = content_sha256(load_protocol(case.path))
    approval_path = Path(binding['approval'])
    approval = json.loads(approval_path.read_text())
    approval['protocol_sha256'] = binding['sha256']
    approval_path.write_text(json.dumps(approval))

    def retarget(certificate, review, report):
        certificate['authorized_protocol_sha256'] = binding['sha256']

    _update_certificate(binding, retarget)
    with pytest.raises(ProtocolGateError, match='confirmed baseline'):
        validate_protocol_launch(case.spec(), binding, repo_root=case.root, check_runtime=False)


@pytest.mark.parametrize('stage', ['confirmation', 'comparison'])
def test_exact_approved_binding_passes_and_records_decision_digest(protocol_case, stage):
    case = protocol_case
    binding = case.bind(stage)
    result = validate_protocol_launch(case.spec(), binding, repo_root=case.root)
    assert result['protocol_id'] == case.protocol['protocol_id']
    assert result['stage'] == stage
    assert len(result['approval_sha256']) == 64
    assert result['source_sha256'] == source_fingerprint(case.root)


def test_missing_approval_cannot_be_replaced_by_candidate_status(protocol_case):
    case = protocol_case
    case.protocol['status'] = 'candidate'
    case.save()
    with pytest.raises(ProtocolGateError, match='missing'):
        validate_protocol_launch(case.spec(), case.bind('confirmation', approve_now=False),
                                 repo_root=case.root)


@pytest.mark.parametrize('change', ['protocol', 'source', 'evidence', 'approval_source',
                                  'approval_protocol', 'decision', 'evidence_kind'])
def test_changed_authorization_inputs_fail_closed(protocol_case, change):
    case = protocol_case
    binding = case.bind()
    spec = case.spec()
    approval_path = Path(binding['approval'])
    approval = json.loads(approval_path.read_text())
    if change == 'protocol':
        case.protocol['description'] += ' changed'
        case.save()
    elif change == 'source':
        case.source.write_text('PROTOCOL_FIXTURE = 2\n')
    elif change == 'evidence':
        Path(approval['evidence'][0]['path']).write_text('changed review evidence')
    else:
        if change == 'approval_source':
            approval['source_sha256'] = '0' * 64
        elif change == 'approval_protocol':
            approval['protocol_sha256'] = '0' * 64
        elif change == 'decision':
            approval['decision'] = 'rejected'
        else:
            approval['evidence'][0]['kind'] = 'historical_candidate_selection_only'
        approval_path.write_text(json.dumps(approval))
    with pytest.raises(ProtocolGateError):
        validate_protocol_launch(spec, binding, repo_root=case.root)


@pytest.mark.parametrize('changed', [
    {'gamma': 0.99},
    {'evaluation_episodes': 1},
    {'max_n_frames': 120000},
    {'loggers': [], 'create_json': False},
    {'restore_file': '/unreviewed/checkpoint.pt'},
    {'checkpoint_at_end': False},
])
def test_protocol_freezes_training_evaluation_and_evidence_settings(protocol_case, changed):
    case = protocol_case
    spec = case.spec()
    changed_spec = dataclasses.replace(spec, experiment={**spec.experiment, **changed})
    with pytest.raises(ProtocolGateError):
        validate_protocol_launch(changed_spec, case.bind(), repo_root=case.root)


def test_managed_output_directory_is_an_allowed_non_scientific_relocation(protocol_case):
    case = protocol_case
    spec = case.spec()
    moved = dataclasses.replace(spec, experiment={**spec.experiment,
                               'save_folder': str(case.root / 'different_run_output')})
    result = validate_protocol_launch(moved, case.bind(), repo_root=case.root)
    assert result['protocol_id'] == case.protocol['protocol_id']


@pytest.mark.parametrize('stage,model,ablation,value', [
    ('confirmation', 'pcp_comm_attention', 'main', None),
    ('comparison', 'pcp_comm_attention', 'message_dim', '8'),
    ('ablation', 'pcp_comm_identity', 'main', None),
])
def test_manual_binding_cannot_bypass_stage_or_confirmation_scope(
    protocol_case, stage, model, ablation, value,
):
    case = protocol_case
    factors = {'model': model, 'ablation': ablation, 'ablation_value': value}
    with pytest.raises(ProtocolGateError):
        validate_protocol_launch(case.spec(**factors), case.bind(stage, **factors),
                                 repo_root=case.root)


@pytest.mark.parametrize('stage', ['confirmation', 'comparison', 'ablation'])
def test_stage_seed_allocation_is_enforced_even_for_handcrafted_bindings(protocol_case, stage):
    case = protocol_case
    factors = ({'model': 'pcp_comm_attention', 'ablation': 'message_dim', 'ablation_value': '8'}
               if stage == 'ablation' else {})
    with pytest.raises(ProtocolGateError, match='seed|Seed'):
        validate_protocol_launch(case.spec(seed=999, **factors),
                                 case.bind(stage, seed=999, **factors), repo_root=case.root)


def test_candidate_cannot_launch_comparison_even_with_matching_review(protocol_case):
    case = protocol_case
    case.protocol['status'] = 'candidate'
    case.save()
    with pytest.raises(ProtocolGateError, match='frozen'):
        validate_protocol_launch(case.spec(), case.bind(), repo_root=case.root)


def test_guarded_communication_factor_requires_explicit_validation(protocol_case):
    case = protocol_case
    factors = {'model': 'pcp_comm_attention', 'ablation': 'message_dim', 'ablation_value': '64'}
    with pytest.raises(ProtocolGateError, match='validation evidence'):
        validate_protocol_launch(case.spec(**factors), case.bind('ablation', **factors),
                                 repo_root=case.root)


def test_guarded_factor_needs_both_reviewed_declaration_and_hashed_factor_evidence(protocol_case):
    case = protocol_case
    factors = {'model': 'pcp_comm_attention', 'ablation': 'message_dim', 'ablation_value': '64'}
    binding = case.bind('ablation', **factors)
    approval_path, evidence, approval = case.approve(
        'ablation', validated_factors=['message_dim/64']
    )
    with pytest.raises(ProtocolGateError, match='factor-specific evidence'):
        validate_protocol_launch(case.spec(**factors), binding, repo_root=case.root)
    approval['evidence'].append({
        'path': str(evidence), 'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest(),
        'kind': 'factor:message_dim/64',
    })
    approval_path.write_text(json.dumps(approval))
    result = validate_protocol_launch(case.spec(**factors), binding, repo_root=case.root)
    assert result['factors']['message_dim'] == '64'


def test_unguarded_declared_factor_passes_and_changes_only_declared_capacity(protocol_case):
    case = protocol_case
    factors = {'model': 'pcp_comm_attention', 'ablation': 'message_dim', 'ablation_value': '8'}
    spec = case.spec(**factors)
    result = validate_protocol_launch(spec, case.bind('ablation', **factors), repo_root=case.root)
    kwargs = spec.model_config['groups']['adversary']['params']['comm_kwargs']
    assert kwargs['message_dim'] == 8
    assert kwargs['key_dim'] == 32
    assert result['factors']['message_dim'] == '8'


@pytest.mark.parametrize('location', ['protocol', 'runtime', 'contracts', 'approval', 'binding'])
def test_unknown_gate_fields_are_errors_not_ignored_annotations(protocol_case, location):
    case = protocol_case
    binding = case.bind()
    if location in {'protocol', 'runtime', 'contracts'}:
        target = case.protocol if location == 'protocol' else case.protocol[location]
        target['unrecognized_gate_option'] = True
        case.save()
        with pytest.raises((TypeError, ValueError)):
            load_protocol(case.path)
    else:
        if location == 'binding':
            binding['skip_verification'] = True
        else:
            path = Path(binding['approval'])
            approval = json.loads(path.read_text())
            approval['skip_verification'] = True
            path.write_text(json.dumps(approval))
        with pytest.raises((TypeError, ValueError)):
            validate_protocol_launch(case.spec(), binding, repo_root=case.root)


@pytest.mark.parametrize('malformation', ['guarded_string', 'unknown_guarded_value',
                                        'boolean_schema', 'invalid_group_type'])
def test_malformed_protocol_controls_fail_at_load(protocol_case, malformation):
    case = protocol_case
    if malformation == 'guarded_string':
        case.protocol['factors']['message_dim']['requires_evidence'] = '64'
    elif malformation == 'unknown_guarded_value':
        case.protocol['factors']['message_dim']['requires_evidence'] = ['undeclared_width']
    elif malformation == 'boolean_schema':
        case.protocol['schema_version'] = True
    else:
        case.protocol['contracts']['training_groups'] = [42]
    case.save()
    with pytest.raises((TypeError, ValueError)):
        load_protocol(case.path)


@pytest.mark.parametrize('field,value', [
    ('measured_groups', ['agent']),
    ('reward_protocol', 'unsupported_reward_semantics'),
    ('prey_policy', 'fixed_not_optimized'),
])
def test_protocol_contract_cannot_mislabel_returns_or_scripted_prey_training(
    protocol_case, field, value,
):
    case = protocol_case
    case.protocol['contracts'][field] = value
    case.save()
    with pytest.raises((TypeError, ValueError)):
        load_protocol(case.path)


@pytest.mark.parametrize('runtime_change', ['threads', 'package', 'python', 'device'])
def test_launch_rejects_unapproved_runtime(protocol_case, monkeypatch, runtime_change):
    case = protocol_case
    if runtime_change == 'threads':
        monkeypatch.setenv('OMP_NUM_THREADS', '8')
    elif runtime_change == 'package':
        case.protocol['runtime']['packages']['torch'] = '0.0.0'
        case.save()
    elif runtime_change == 'python':
        case.protocol['runtime']['python'] = '0.0.0'
        case.save()
    else:
        spec = case.spec()
        spec = dataclasses.replace(spec, experiment={**spec.experiment, 'train_device': 'cuda'})
        with pytest.raises(ProtocolGateError, match='device|runtime'):
            validate_protocol_launch(spec, case.bind(), repo_root=case.root)
        return
    with pytest.raises(ProtocolGateError, match='Runtime'):
        validate_protocol_launch(case.spec(), case.bind(), repo_root=case.root)


def _suite(case):
    return {'suite_id': 'isolated_protocol_fixture',
            'protocol': str(case.path.relative_to(case.root)), 'stage': 'confirmation',
            'models': ['pcp_comm_identity'], 'seeds': [0], 'ablation': 'main'}


def test_manifest_roundtrip_retains_complete_protocol_and_built_model_contract(protocol_case):
    case = protocol_case
    case.approve('confirmation')
    plans = expand_suite_config(_suite(case), repo_root=case.root)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.model_contract['groups']['adversary']['parameters']
    assert any('CommPolicyModel' in name
               for name in plan.model_contract['groups']['adversary']['classes'])
    assert plan.model_contract['critic']['adversary']['parameters']
    assert plan.model_contract['parameter_counts']['group/adversary/communication_total'] == 0
    assert plan.model_contract['task_contract']['randomness_protocol'] == 'pcp_rng_v2'
    manifest = case.root / 'manifest.csv'
    write_manifest(manifest, plans)
    restored = read_manifest(manifest)[0]
    assert restored == plan
    spec = validate_plan(restored, config_root=case.root / 'configs', repo_root=case.root)
    assert scientific_config_sha256(spec) == plan.scientific_hash


@pytest.mark.parametrize('model', [
    'pcp_actor', 'pcp_comm_identity', 'pcp_comm_attention', 'pcp_comm_graph',
])
def test_real_pcp_contract_matches_functional_critics_and_each_actor_group(
    config_root, tmp_path, model,
):
    """Do not infer functionalized critic capacity from its class name or actor shape."""
    from commstudy.experiments.runner import build_experiment

    spec = load_experiment_spec(config_root, [
        'task=vmas_predator_capture_prey', f'model={model}', 'critic_model=pcp_critic',
        'experiment.evaluation=false', 'experiment.on_policy_n_envs_per_worker=1',
    ])
    spec = dataclasses.replace(spec, experiment={
        **spec.experiment, 'save_folder': str(tmp_path), 'loggers': [], 'create_json': False,
    })
    experiment = build_experiment(spec)
    try:
        contract = actual_model_contract(experiment)
        assert contract['training_groups'] == ['adversary', 'agent']
        assert set(contract['groups']) == set(contract['critic']) == {'adversary', 'agent'}
        for group in ('adversary', 'agent'):
            critic = contract['critic'][group]
            functional_shapes = {
                '.'.join(key): list(value.shape)
                for key, value in experiment.losses[group].critic_network_params.items(True, True)
                if isinstance(value, torch.nn.Parameter)
            }
            assert critic['parameters'] == functional_shapes
            assert critic['parameters']
            assert any(name.endswith('.Mlp') for name in critic['classes'])
            expected_width = 128 if group == 'adversary' else 8
            assert [expected_width, 22] in critic['parameters'].values()
            assert sum(math.prod(shape) for shape in critic['parameters'].values()) == (
                contract['parameter_counts'][f'group/{group}/critic_total']
            )
            actor = contract['groups'][group]
            observation_width = 17 if group == 'adversary' else 14
            assert [expected_width, observation_width] in actor['parameters'].values()
            assert sum(math.prod(shape) for shape in actor['parameters'].values()) == (
                contract['parameter_counts'][f'group/{group}/actor_total']
            )
        assert contract['parameter_counts']['group/adversary/critic_total'] == 19585
        assert contract['parameter_counts']['group/agent/critic_total'] == 193
        assert contract['parameter_counts']['group/agent/actor_total'] == 156
    finally:
        experiment.close()


def test_managed_scientific_pcp_requires_contract_before_creating_run_directory(protocol_case):
    from commstudy.experiments.bookkeeping import RunContext
    from commstudy.experiments.runner import run_managed_experiment

    case = protocol_case
    context = RunContext(suite_id='fixture', run_id='missing_contract',
                         output_root=case.root / 'runs')
    with pytest.raises(ProtocolGateError, match='requires a built model contract'):
        run_managed_experiment(case.spec(), context, repo_root=case.root,
                               protocol_binding=case.bind(), expected_contract=None)
    assert not context.run_dir.exists()


def test_protocol_export_preserves_critic_override_independent_of_named_yaml(protocol_case):
    case = protocol_case
    declared_critic = case.protocol['base_spec']['critic_model']
    declared_critic['groups']['adversary']['params']['num_cells'] = [64, 64]
    case.save()
    case.approve('confirmation')
    plan = expand_suite_config(_suite(case), repo_root=case.root)[0]
    spec = validate_plan(plan, config_root=case.root / 'configs', repo_root=case.root)
    assert spec.critic_model == declared_critic


@pytest.mark.parametrize('missing', [
    'resolved_spec', 'scientific_hash', 'protocol', 'model_contract',
])
def test_protocol_manifest_cannot_lose_its_required_bindings(protocol_case, missing):
    case = protocol_case
    case.approve('confirmation')
    plan = expand_suite_config(_suite(case), repo_root=case.root)[0]
    altered = dataclasses.replace(plan, **{missing: None})
    with pytest.raises(ProtocolGateError):
        validate_plan(altered, config_root=case.root / 'configs', repo_root=case.root)


def test_manifest_cannot_diverge_its_snapshot_labels_or_explanatory_overrides(protocol_case):
    case = protocol_case
    case.approve('confirmation')
    plan = expand_suite_config(_suite(case), repo_root=case.root)[0]
    snapshot = copy.deepcopy(plan.resolved_spec)
    snapshot['experiment']['gamma'] = 0.99
    variants = [
        dataclasses.replace(plan, resolved_spec=snapshot),
        dataclasses.replace(plan, seed=99),
        dataclasses.replace(plan, overrides=(*plan.overrides, 'experiment.gamma=0.99')),
    ]
    for altered in variants:
        with pytest.raises(ProtocolGateError):
            validate_plan(altered, config_root=case.root / 'configs', repo_root=case.root)


def test_built_contract_cannot_claim_different_semantics_or_training_groups(protocol_case):
    from commstudy.experiments.protocols import validate_built_contract

    case = protocol_case
    plan = expand_suite_config(_suite(case), repo_root=case.root)[0]
    validate_built_contract(case.protocol, plan.model_contract)
    for field, value in [('randomness_protocol', 'stale_rng'), ('training_groups', ['adversary'])]:
        changed_protocol = copy.deepcopy(case.protocol)
        changed_protocol['contracts'][field] = value
        with pytest.raises(ProtocolGateError):
            validate_built_contract(changed_protocol, plan.model_contract)


def test_repository_candidate_has_no_approval_and_cannot_launch(config_root):
    root = config_root.parent
    path = config_root / 'protocols' / 'pcp_corrected_v1.yaml'
    protocol = load_protocol(path)
    assert protocol['status'] == 'candidate'
    stage = 'confirmation'
    approval = config_root / 'protocols' / 'approvals' / f"{protocol['protocol_id']}_{stage}.json"
    assert not approval.exists(), 'Implementation tests must never generate scientific approval.'
    seed = protocol['selection']['confirmation_seeds'][0]
    binding = {
        'path': str(path), 'sha256': content_sha256(protocol),
        'source_sha256': source_fingerprint(root), 'approval': str(approval),
        'stage': stage, 'model': 'pcp_comm_identity', 'seed': seed,
        'ablation': 'main', 'ablation_value': None,
    }
    spec = protocol_spec(protocol, model='pcp_comm_identity', seed=seed)
    with pytest.raises(ProtocolGateError, match='missing'):
        validate_protocol_launch(spec, binding, repo_root=root, check_runtime=False)
