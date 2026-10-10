# Port of Paper_ws filter code into ws_mobile

Source: github.com/pratiyushparashar/paper_ws commit efe31ef0ccbbd50f4be8455ced6e0dfc9bda6adf (read-only clone /home/kushagra/paper_ws_ref).
Copied: zupt/, config/, tests/, tools/check_imu_integration.py, tools/gen_terrain_world.py,
docs/ (as docs_paperws/). Added: zupt/paths.py.

Edits (default paths only; no computation changed):
- tools/check_imu_integration.py: 1 default path(s)
- zupt/replay.py: 2 default path(s)
- zupt/evaluation/eval_5c.py: 4 default path(s)
- zupt/evaluation/eval_stuck.py: 3 default path(s)
- zupt/evaluation/eval_5b.py: 3 default path(s)
- zupt/evaluation/dev_eval.py: 4 default path(s)
- zupt/evaluation/eval_change1.py: 4 default path(s)
- zupt/ekf/run.py: 3 default path(s)
- tools/gen_terrain_world.py: OUT -> <ws_mobile>/src/mobile_robot/model/terrain_world.sdf
- terrain_world.sdf: copied byte-identical into src/mobile_robot/model/ (colcon build needed)

Check after the port: unit tests + eval_change1 on Paper_ws data must reproduce
docs_paperws/change1.md exactly.
