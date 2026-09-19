"""Synthetic network fixture.

MAPDN's real case33/141/322 data is a multi-GB download that is not in the
repository, so the tests build a small schema-faithful stand-in instead: the
same pandapower layout (zoned buses, sgen["name"] carrying the zone label) and
the same three CSVs. Zones are deliberately different sizes so the observation
zero-padding path is exercised.
"""
import numpy as np
import pandas as pd
import pandapower as pp
import pytest

ZONES = {"zone1": 3, "zone2": 5, "zone3": 4}
# (zone, index of the bus within that zone) for each PV inverter / agent
SGEN_PLAN = [("zone1", 0), ("zone1", 2), ("zone2", 1), ("zone3", 0), ("zone3", 3)]
N_AGENTS = len(SGEN_PLAN)


def _build_network(path):
    rng = np.random.default_rng(0)
    net = pp.create_empty_network(name="test_net")

    slack = pp.create_bus(net, vn_kv=12.66, name="bus0", zone="main")
    pp.create_ext_grid(net, bus=slack, vm_pu=1.0)

    zone_buses = {}
    for zone, n_buses in ZONES.items():
        prev, buses = slack, []
        for k in range(n_buses):
            bus = pp.create_bus(net, vn_kv=12.66, name=f"{zone}_b{k}", zone=zone)
            pp.create_line_from_parameters(
                net, from_bus=prev, to_bus=bus, length_km=1.0,
                r_ohm_per_km=0.35, x_ohm_per_km=0.18,
                c_nf_per_km=0.0, max_i_ka=1.0,
            )
            buses.append(bus)
            prev = bus
        zone_buses[zone] = buses

    for zone, buses in zone_buses.items():
        for bus in buses:
            pp.create_load(net, bus=bus, p_mw=0.1, q_mvar=0.05)

    for zone, idx in SGEN_PLAN:
        pp.create_sgen(net, bus=zone_buses[zone][idx], p_mw=0.05, q_mvar=0.0, name=zone)

    pp.runpp(net)
    pp.to_pickle(net, str(path / "model.p"))

    n_steps = 4 * 24 * 20  # 4 days at 3-minute resolution
    index = pd.date_range("2020-01-01", periods=n_steps, freq="3min")
    phase = 2 * np.pi * (np.arange(n_steps) % 480) / 480.0
    solar = np.clip(np.sin(phase - np.pi / 2), 0, None)

    frames = {
        "pv_active.csv": {
            f"pv{i}": 0.12 * solar * (0.8 + 0.4 * rng.random()) + 0.005 * rng.random(n_steps)
            for i in range(len(net.sgen))
        },
        "load_active.csv": {
            f"la{i}": 0.10 * (0.7 + 0.3 * np.sin(phase)) + 0.01 * rng.random(n_steps)
            for i in range(len(net.load))
        },
        "load_reactive.csv": {
            f"lr{i}": 0.05 * (0.7 + 0.3 * np.sin(phase)) + 0.005 * rng.random(n_steps)
            for i in range(len(net.load))
        },
    }
    for name, columns in frames.items():
        frame = pd.DataFrame(columns, index=index)
        frame.insert(0, "time", index.strftime("%Y-%m-%d %H:%M:%S"))
        frame.to_csv(path / name, index=False)


@pytest.fixture(scope="session")
def data_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("mapdn_data")
    _build_network(path)
    return str(path)


@pytest.fixture
def config(data_path):
    return {
        "data_path": data_path,
        "mode": "distributed",
        "voltage_barrier_type": "l1",
        "voltage_weight": 1.0,
        "q_weight": 0.1,
        "line_weight": None,
        "dq_dv_weight": None,
        "history": 1,
        "pv_scale": 1.0,
        "demand_scale": 1.0,
        "state_space": ["pv", "demand", "reactive", "vm_pu", "va_degree"],
        "v_upper": 1.05,
        "v_lower": 0.95,
        "episode_limit": 10,
        "action_scale": 0.8,
        "action_bias": 0.0,
        "reset_action": True,
        "seed": 0,
    }
