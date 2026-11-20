# Kyle Field counter-UAS: behavior recognition (simulated data)

Everything here runs on simulated data. Sensor performance numbers are assumptions (see the top of `simulate.py`).

## Quick look
Open `cuas_demo.html` in a browser (double-click; works offline). Pick a scenario, press Play.

## Sensor layout (v9)
Six sites, one base kit (camera + RF sensor); two of them also carry a radar:
- 4 stadium corner roofs: camera + RF sensor (they watch the bowl, the stands and the walls)
- Central Campus Garage and West Campus Garage roofs (7 levels each): 360-degree radar + camera + RF sensor

Line of sight for every sensor uses a height map (`H_sim.npy`, `canopy_sim.npy`) built from our FreeCAD stadium model
plus OpenStreetMap building footprints around Kyle Field (heights from floor counts where available, otherwise estimated;
flat ground; trees not included). Building data (c) OpenStreetMap contributors, ODbL.

## Sensor assumptions (v9)
Each value is set at the top of `simulate.py`. Where the source is our own assumption, it says so.
| What | Value | Source |
|---|---|---|
| Radar range by drone size | under 250 g: 1.1 km; ~1 kg: 1.8 km; ~4 kg: 3.0 km; ~10 kg: 4.2 km (full rate to 75% of that, zero at it) | IRIS: 1.1 km on a 0.249 kg drone, 7.6 km on 52 kg; the middle classes are interpolated on a log-log line between those two points (ours) |
| Drone size mix | hobbyists 45% under 250 g / 45% ~1 kg / 10% ~4 kg; attackers 20/55/25%; payload drones ~4-10 kg; light-show drones under 250 g to ~1 kg; delivery drones ~4-10 kg | assumption |
| Radar on birds | ~1.8 km (similar radar size to a ~1 kg drone); small birds and bats ~1.1 km | assumption |
| RF direction finding | +-5 degrees per sensor | Dedrone RF-300 datasheet |
| Corner cameras (standard PTZ) | ~1 km day, ~400 m night (IR), on a ~1 kg drone | Dedrone standard PTZ |
| Garage cameras (long-range PTZ) | ~2 km day; 800 m night | day: Dedrone long-range PTZ; night: our assumption (same ratio as the standard camera) |
| Camera range vs size | x0.7 under 250 g, x1.4 ~4 kg, x1.8 ~10 kg | more pixels on a bigger target; capped below the size ratio (ours) |
| Sensor heights | corners: camera 39 m, RF 40 m; garages: camera 3.0 m, RF 3.4 m, radar 4.2 m above the top deck (23.9 m) | CAD models |
| Remote ID | heard reliably to ~400 m, nothing past 1 km | field tests with phone apps: ~100-500 m |
| Traffic per game | ~10 careless hobbyists (NFL: 2,845 incursions in the 2023 season); birds, phones, approved drones and the rest are assumptions | see `gameday_fp.py` |

The model is never told a drone's size; it only sees what the sensors report.

## Headline results (v9, simulated; denominators stated)
From `v9_train.txt`, `v9_gd.txt`, `v9_stress.txt`, `v9_sw.txt`, `threat_cost_v9.csv`, `case_stats.csv`, `latency_study.txt`.
- Snapshot-level accuracy (one row = one track at one moment): 92.1% on track snapshots from test cases of the 64
  trained scenario types (11,402 rows, 1,236 cases); 76.4% on snapshots from the 7 held-out scenario types (4,540 rows,
  527 cases). The held-out set (`HELD` in `simulate.py`) has been the same since v3 and was looked at during development,
  so it is a generalisation check, not an untouched final validation.
- Track-level alarm timing (`threat_cost.py`, 400 sampled threat cases from unseen cases or held-out types; 840 threat
  tracks): of the 304 outside-origin tracks that reached the 120 m midfield radius, 301 were alarmed by arrival (median
  warning 81 s, 10th percentile 40 s) and 304 were notified by arrival. Objects never detected produce no track and are
  not in this evaluation.
- Case-level detection (`case_stats.csv`): 42 of 3,125 threat cases were never detected within the 150 s simulation.
  Replaying those 42 trajectories, 39 were still 1-3.7 km out when the window ended and 3 entered the 120 m midfield
  radius undetected (all radio-silent and low). Warning from first detection by smallest drone in the case (outside
  starts): under 250 g median 69 s (10th pct 29 s); ~1 kg 79 s (19 s); ~4 kg 92 s (45 s); ~10 kg 97 s (53 s).
- Game day, assumed typical traffic mix: about 0.5 threat alarms per game on authorized or non-drone traffic plus about
  3.5 careless-hobbyist escalations to THREAT; about 49 notifications per game on authorized/non-drone traffic plus about
  10 on careless hobbyists (every hobbyist encounter notified), before any real threats. Stress test: 6.3 false threat
  alarms per game.
- Swarms: individual drone alerts 97-100%; whole-group recognition is weaker (37-60% for flyover, tailgate and post-game
  attacks; 30% for an extra drone hidden in an approved light show). No false swarm alarms were observed on bird flocks
  or approved shows in this test set.

## What the simulator does and does not model
- Position fusion (`simulate.py`, "fused position"): each second, the measured source with the smallest modeled
  position error (radar, RF direction-finding fix, or camera) is used; a Remote ID self-reported position is used only
  when nothing measured the object. This is best-available-source selection, not weighted averaging or a Kalman filter.
- Simplifications: altitude is truth plus 3 m noise; approved-list (manifest) membership comes from the scenario
  definition, so identity association is assumed; camera "drone / bird / other" calls are simulated observations whose
  accuracy depends on range and conditions. No image detector was trained. The trained model classifies behaviour from
  these inputs. Results do not validate a complete real-world sensing pipeline.
- Display persistence (`UP_S, DOWN_S = 2, 8`): a call must persist 2 s to raise concern and 8 s to lower it. Swarm links
  need 7 of the last 10 s; group status steps down after 10 s.
- Flight-plan check: deviation counts only beyond 2.5x the track's own modeled position error (per-axis sigma). For a
  single independent sample from a 2-D isotropic Gaussian that is exceeded by noise about 4.4% of the time
  (exp(-2.5^2/2)); smoothing and nearest-point route distance make the real rate differ.
- Demo: the page replays precomputed simulation and model outputs. Scenario-list runs are curated (best of several
  candidates; swarm examples must show the expected group outcome). "Challenge the system" picks at random from a pool
  of 94 precomputed runs generated with fresh random seeds (not hand-picked); nothing is simulated or trained in the browser.

## Timing measurement (`latency_study.py` -> `latency_study.txt`)
Development container: Intel Xeon @ 2.30 GHz, 2 vCPU; Python 3.13.18, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.5;
OMP_NUM_THREADS=2, random forest n_jobs=1. Feature extraction averaged over 200 calls; model over 30 calls (1 track,
after one warm-up) and 10 calls (batch of 50 tracks). Result: 0.6 ms features, 24 ms model (1 track), 32 ms (50 tracks).
Not measured on deployment hardware.

## Setup
`pip install -r requirements.txt` (versions the saved model was built and loaded with). `model.pkl.gz` is a pickle:
load it with the same scikit-learn version (1.9.1) and keep `cuas_model.py` next to it. Keep every file in this folder.

## Pipeline (run in this order to rebuild from scratch)
| Step | Command | Makes |
|---|---|---|
| 1. Generate scenarios | `python gen_cases.py` | `cuas_test_cases.csv`, `cuas_scenarios.csv` (7,485 cases, 71 scenario types) |
| 2. Simulate sensors and extract measurements | `python simulate.py` (or `python run_sim_par.py`, the 2-process version used for v9) | `features.csv` (52 measurements per track snapshot), `case_stats.csv` |
| 3. Train the model | `python train.py` | `model.pkl.gz` (random forest + gradient boosting, averaged; needs `cuas_model.py` to load) |
| 4. Build the demo page | `python build_demo.py` | `cuas_demo.html` (needs `demo_template.html`, `three.min.js`, `KyleField_demo_mesh.json`, `swarm_cases.csv`, `buildings_demo.json`) |

Steps 2 and 4 take several minutes each; step 3 a few minutes.

## What each code file is
| File | Role |
|---|---|
| `gen_cases.py` | Scenario definitions and random case generation |
| `simulate.py` | Flight paths, sensor models (radar, RF, Remote ID, shared cameras), fusion, the 52 measurements |
| `geo.py` | Line of sight on the stadium + campus height map (`H_sim.npy`, `canopy_sim.npy`, `garages.json`) |
| `tracker.py` | Track IDs, coasting when a track is lost, re-association, radar handoff |
| `train.py` | Trains and evaluates the model (held-out scenario types, test split) |
| `cuas_model.py` | The averaged model class (needed to load `model.pkl.gz`) |
| `alert_rules.py` | Confirm-before-alarm rule and status levels (alarm / alert / watching), incl. the radio ghost check |
| `swarm.py` | Swarm module: group IDs, coordination indicators, group status |
| `build_demo.py` | Picks demo runs, replays the model second by second, writes the page |
| `demo_template.html` | The demo page layout (data gets pasted in) |
| `site_survey/` | How the radar sites were chosen: `world.py` (height map from OSM + CAD), `sites.py` and `rank.py` (coverage of every candidate pair), `kf.geojson` (OSM export), `radar_sites_kyle_field.png` |

## Studies and reports (optional)
| Command | What it measures |
|---|---|
| `python scenario_report.py` | Accuracy per scenario type |
| `python compare_models.py` | Random forest vs gradient boosting vs the average |
| `python eval_steady.py` | Effect of the hold-steady rule; hijack alert times; camera wait |
| `python make_gameday_cases.py` then `python gameday_fp.py` then `python gameday_report.py` and `python notify_burden.py` | False alarms per game day |
| `python threat_cost.py` | What the alert rule costs on real threats |
| `python make_swarm_cases.py`, `python swarm_eval.py simulate 0 1`, then `python swarm_eval.py` | Swarm detection vs benign groups |

Saved results: `gameday_tracks.csv`, `gameday_tracks_v9_stress.csv`, `threat_cost_v9.csv`, `swarm_eval_0.csv`, and the
text reports `v9_train.txt`, `v9_gd.txt`, `v9_stress.txt`, `v9_sw.txt`.

## Site survey (`site_survey/`)
Inputs: `kf.geojson` (OpenStreetMap export), `origin.json` (map origin aligned to the CAD stadium), `../KyleField_demo_mesh.json`.
1. `python make_heightmap.py` -> `H.npy` (2 m height map) and `blds.pkl` (buildings with estimated heights)
2. `python sites.py` -> `vis.pkl` (line of sight from every candidate site to every target point at 15/30/60 m)
3. `python rank.py` -> printed coverage of site pairs (geometric line of sight, area-weighted, targets 160-1,300 m from
   midfield over open ground; not probability of detection)
The saved `H.npy`, `blds.pkl` and `vis.pkl` are included. Paths are relative to this folder. Only step 3 was re-run after
the path change; steps 1-2 were not re-run. Note: `vis.pkl` was computed with West Campus Garage at an older, lower
height (radar 19 m above ground vs 26.9 m used in the simulator), so its West Campus Garage coverage is conservative.

## Rebuild status
Not re-verified end to end after packaging. Verified after packaging: the model loads and predicts with the listed
versions; `site_survey/rank.py` runs from its folder.

## Rules-only baseline vs. model (rules_baseline.py -> rules_baseline.txt)
Following the workshop advice "start with clear rules, then upgrade one block with ML and compare": on the same test and
held-out snapshots, a rules-only alarm (confirmed drone + urgent + not consistent with an approved flight) is compared with
our model alarm (model THREAT + confirmed + urgent). Snapshot level, no persistence/latching.
