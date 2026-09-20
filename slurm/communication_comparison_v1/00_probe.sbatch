#!/usr/bin/env bash
set -euo pipefail
# sbatch --chdir sets the immutable source root, even from Slurm spool copies.
source "$PWD/slurm/communication_comparison_v1/common_env.sh"
suite_profile "$1" "$2"
export COMM_SUITE_PROFILE="$1"
[[ -n "${SLURM_JOB_ID:-}" ]]
command -v srun
command -v sacct
scontrol --version
scontrol show config | sed -n '/MaxArraySize/p'
sinfo -h -o '%P %a %l %c %m %G'
scontrol show job "$SLURM_JOB_ID"
srun --kill-on-bad-exit=1 "$BOOTSTRAP_PYTHON" - <<'PY'
import hashlib, json, platform, shutil, os, subprocess
from pathlib import Path
if platform.system() != 'Linux':
    raise SystemExit('Target profile requires Linux; local results cannot qualify Newton')
root = Path(os.environ['RUN_ROOT'])
if shutil.which('lfs'):
    command = ['lfs', 'quota', '-u', str(os.getuid()), str(root)]
elif shutil.which('quota'):
    command = ['quota', '-s']
else:
    raise SystemExit('No quota query tool available; stage a site-supported quota check before qualification')
quota = subprocess.run(command, text=True, capture_output=True, check=True)
record = {'status':'passed', 'system':platform.system(), 'machine':platform.machine(),
 'python':platform.python_version(), 'free_bytes':shutil.disk_usage(root).free,
 'device':'cpu', 'runtime_qualified':False, 'quota_command':command,
 'quota_report':quota.stdout, 'slurm_job_id':os.environ['SLURM_JOB_ID'],
 'profile_sha256':hashlib.sha256(Path(os.environ['COMM_SUITE_PROFILE']).read_bytes()).hexdigest()}
if record['free_bytes'] < int(os.environ['MIN_FREE_GIB']) * 1024**3:
    raise SystemExit('Insufficient physical free space for the requested profile')
(root / 'gates').mkdir(exist_ok=True)
target = root / 'gates/cluster_probe.json'
temporary = target.with_name('.cluster_probe.' + str(os.getpid()) + '.tmp')
with temporary.open('x') as stream:
    json.dump(record, stream, indent=2, allow_nan=False)
    stream.flush()
    os.fsync(stream.fileno())
os.link(temporary, target)  # atomic create, refuses an existing probe receipt
temporary.unlink()
print(json.dumps(record, indent=2))
PY
