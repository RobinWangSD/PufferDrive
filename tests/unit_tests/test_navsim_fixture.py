"""Tests for the navsim fixture scene and the navsim replay conventions (docs/navsim_data.md).

A navsim bin is one navsim sample: 20 s at 10 Hz starting at the sample's first history frame, so the navsim
"current frame" (the planning instant) is step 15 and the navsim task horizon is the next 4 s = 40 steps. The
`navsim_single` / `navsim_multi` benchmarks therefore run `init_step=15, scenario_length=55`.

Fixture map
-----------
`pufferlib/resources/drive/binaries/navsim/navsim__b214f8e744075e96.bin`, a navsim-mini scene converted by
closed-loop-e2e (115 agents, 11 traffic lights, per-point lane widths), whose ego has a route.
"""

import os

import numpy as np
import pytest

from pufferlib.ocean.drive.drive import Drive


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "navsim")

INIT_STEP = 15  # the navsim current frame: 4 history frames at 2 Hz = steps 0, 5, 10, 15
HORIZON_STEPS = 40  # the navsim horizon: 4 s at 10 Hz
SCENARIO_LENGTH = INIT_STEP + HORIZON_STEPS


def _make_sdc_replay_env():
    """The `navsim_single` settings with a log-replay ego, so the run is deterministic."""
    return Drive(
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=1,
        map_dir=MAP_DIR,
        simulation_mode="replay",
        control_mode="control_sdc_only",
        sdc_controller="replay",
        non_sdc_controller="replay",
        dt=0.1,
        init_step=INIT_STEP,
        scenario_length=SCENARIO_LENGTH,
        resample_frequency=1_000_000,  # don't resample mid-episode
        termination_mode=False,
        stagger_first_episode=False,
        goal_source="gt_map",
        num_goals=5,
        use_neighbor_cache=False,
        report_interval=1,
    )


def _scenario(env):
    state = env.get_state()
    return state[0] if isinstance(state, list) else state


def test_navsim_replay_follows_log_over_the_horizon():
    """From step 15 the replayed ego sits on its logged pose at every step, observations stay finite, and the
    episode truncates after exactly 40 steps (scenario_length 55)."""
    env = _make_sdc_replay_env()
    try:
        env.reset(seed=0)
        ego = _scenario(env)["agents"][0]
        log_x = np.asarray(ego["log_trajectory_x"], dtype=np.float64)
        log_y = np.asarray(ego["log_trajectory_y"], dtype=np.float64)
        assert len(log_x) > SCENARIO_LENGTH, "the fixture must cover the whole navsim horizon"
        assert abs(ego["sim_x"] - log_x[INIT_STEP]) < 1e-3 and abs(ego["sim_y"] - log_y[INIT_STEP]) < 1e-3

        zero_action = np.zeros_like(env.actions)
        truncated_at = None
        for step in range(1, HORIZON_STEPS + 1):
            obs, _, _, truncations, _ = env.step(zero_action)
            assert np.isfinite(obs).all()
            ego = _scenario(env)["agents"][0]
            t = INIT_STEP + step
            assert abs(ego["sim_x"] - log_x[t]) < 1e-3 and abs(ego["sim_y"] - log_y[t]) < 1e-3, f"step {t}"
            if truncations[0]:
                truncated_at = step
                break
        assert truncated_at == HORIZON_STEPS, (
            f"expected the episode to end after {HORIZON_STEPS} steps, got {truncated_at}"
        )
    finally:
        env.close()


def test_navsim_multi_agent_config_resets():
    """The `navsim_multi` settings (vehicles with routes are controlled, the rest replayed) build and reset on the fixture."""
    env = Drive(
        num_agents=64,
        min_agents_per_env=1,
        max_agents_per_env=64,
        num_maps=1,
        map_dir=MAP_DIR,
        simulation_mode="replay",
        control_mode="control_vehicles",
        init_mode="create_controllable_types",
        replay_expert_agents=0,
        dt=0.1,
        init_step=INIT_STEP,
        scenario_length=SCENARIO_LENGTH,
        resample_frequency=1_000_000,
        termination_mode=False,
        stagger_first_episode=False,
        goal_source="gt_map",
        num_goals=5,
        use_neighbor_cache=False,
        report_interval=1,
    )
    try:
        env.reset(seed=0)
        scenario = _scenario(env)
        assert scenario["active_agent_count"] >= 1
        obs, _, _, _, _ = env.step(np.zeros_like(env.actions))
        assert np.isfinite(obs).all()
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
