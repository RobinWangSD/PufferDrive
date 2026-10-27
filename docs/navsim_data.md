# navsim data — conversion, layout, usage

How to turn [navsim](https://github.com/autonomousvision/navsim) scenes into the `.bin` scenario files
PufferDrive's replay mode consumes, and how the `navsim_single` / `navsim_multi` benchmarks use them.

navsim (NAVSIM v1/v2, "non-reactive autonomous vehicle simulation") is a planning benchmark built on the
nuPlan logs resampled to 2 Hz (the OpenScene release): `navtrain` (103,288 samples) and `navtest` (12,146
samples), each sample = 4 history frames + 10 future frames of one log, with the nuPlan HD maps. Licence:
CC BY-NC-SA 4.0 (nuPlan, Motional) — non-commercial use only, attribution required, share-alike; the same terms
as the nuPlan bins in `data_utils/datasets.yaml`.

## No pre-converted set

There is no S3 copy of navsim bins (nothing in `data_utils/datasets.yaml`: the registry is for fetchable
datasets, and an entry without a bucket would only advertise a fetch that fails). Convert them yourself.

## Converting: the closed-loop-e2e converter

The converter lives in [closed-loop-e2e](https://github.com/RobinWangSD/closed-loop-e2e)
(`scripts/convert_navsim_puffer.py`, `surrogate/sim/puffer_scenario.py`), which reads the OpenScene per-log
pickles and the nuPlan maps through nuplan-devkit — the data a navsim installation already has. It follows
123Drive's nuPlan pipeline (lanes with topology, speed limits and per-point widths, road lines, road edges,
crosswalks, per-step traffic-light states, stop / yield signs, DP lane routes, control states, the
log-only-agent overlap rule, a Dijkstra lane graph) so a navsim scene lands here like the official nuPlan bins.

```bash
# in the closed-loop-e2e checkout; NAVDATA_LOGS = the split's log pickles, NAVDATA_MAPS = the nuPlan maps
python scripts/convert_navsim_puffer.py --index data_cache/index_navtest_eval.json --workers 32 --out data/navsim_navtest
python scripts/convert_navsim_puffer.py --token b214f8e744075e96                        # one sample
python scripts/convert_navsim_puffer.py --index ... --shard 0/4 --workers 32            # one of 4 contiguous row ranges
```

What a bin is:

- **one navsim sample**, 20 s at 10 Hz (T = 201) starting at the sample's first history frame, so the navsim
  **current frame is step 15** (4 history frames at 2 Hz = steps 0, 5, 10, 15) and the navsim task horizon is
  the next 40 steps; a sample near the end of its log is shorter (always ≥ 66 steps);
- the 2 Hz log frames are **interpolated** to 0.1 s (positions and headings linear, velocities from the log),
  so replayed partners move piecewise-linearly between the logged frames — the one difference from the 10 Hz
  nuPlan bins;
- the ego is agent 0 (the nuPlan Pacifica box, 5.176 x 2.297 m), every annotated vehicle / pedestrian / bicycle
  is an agent, cones / barriers / signs are objects; the ego's route extension prefers the navsim route
  roadblocks;
- the optional `LANEWID1` tail section carries per-point lane widths from the nuPlan lane boundaries (the
  nuPlan bins fall back to 3.7 m); no `TLPHASE1` junction phases, no `SPDZONE1` zones;
- next to each bin the converter writes `navsim__<token>.json`, a sidecar with the scene anchor (city
  coordinates minus the window's mean ego position), the track tokens per agent index, the nuPlan lane ids per
  lane index, the traffic-light connector ids and the logged ego status per frame — what an external planner
  needs to map simulator state back to navsim / nuPlan identities. PufferDrive ignores it.

Conversion cost: 1.5-6.5 s per scene on one core (the route search of dense scenes dominates), 1.3-3 MB per bin.

## Expected output layout

A flat directory of one `.bin` per sample, named by the navsim sample token:

```
navsim_navtest/
├── navsim__b214f8e744075e96.bin
├── navsim__b214f8e744075e96.json      # sidecar, ignored by PufferDrive
├── navsim__...bin
└── summary.json                       # the converter's per-scene records, ignored by PufferDrive
```

A checked-in sample lives at `pufferlib/resources/drive/binaries/navsim/navsim__b214f8e744075e96.bin`
(a Las Vegas scene: 115 agents, 11 traffic lights).

## Using the bins

Replay training on navsim bins, controlling only the SDC, with episodes that start at the navsim current frame
and last the navsim horizon (`env.scenario_length=200` runs the whole bin instead):

```bash
puffer train puffer_drive \
    env.map_dir=data/navsim_navtrain \
    env.num_maps=103288 \
    env.simulation_mode=replay \
    env.control_mode=control_sdc_only \
    env.dt=0.1 \
    env.init_step=15 \
    env.scenario_length=55 \
    env.goal_source=gt_map \
    env.num_goals=5
# a subset of the bins: env.num_maps=1000 env.allow_map_subset=true
```

`env.dt` must be 0.1 (the bins' frame rate; the default is 0.3 and replay never checks it). `gt_map` goals are
spread over the whole 20 s log, so `num_goals=5` is what makes the first goal (3.7 s ahead) reachable within the
horizon.

Evaluation with the `navsim_single` (SDC only) and `navsim_multi` (every routed vehicle controlled) benchmarks
of `pufferlib/config/evaluation/benchmark.yaml`, which carry these settings:

```bash
puffer eval puffer_drive navsim_single eval.map_dir=data/navsim_navtest load_model_path=<run>/models/model_puffer_drive_XXXX.pt
# a subset, or the checked-in sample: env.num_maps=N eval.num_scenarios=N
```

An external planner (one that outputs waypoints) can drive the ego of a navsim scene through the co-sim
setters (`Drive.set_agent_states` on agent 0 with `sdc_controller=static`, `pufferlib/ocean/cosim`); the
closed-loop-e2e repository does that for its object-level planner (its `docs/PUFFERDRIVE.md`).
