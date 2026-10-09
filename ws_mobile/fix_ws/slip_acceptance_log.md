# Phase 3B session acceptance log (PREREG_slip.md §3)

| session | role | route | slip_accept | reason / remark | replacements used |
|---|---|---|---|---|---|
| 20261009_140649 | gate | B2 slip gate (corridor patch) | gate only | recorded before the protocol; REJECT on P2, P5 expected; not a dev session | — |
| 20261009_145330 | dev | 1 corridor+north (west exit) | REJECT | P1 duration 117.8 s > 100 s (P2–P6 ok) | dev 1/3 |
| 20261009_150044 | dev | 1 corridor+north (west exit), replacement | ACCEPT | 82.2 s | — |
| 20261009_150546 | dev | 2 corridor+south (west exit) | ACCEPT | 86.4 s | — |
| 20261009_151240 | dev | 3 north+south (west strip), 0.6 m/s | ACCEPT | 98.9 s | — |
| 20261009_151812 | dev | 4 all three patches | ACCEPT | 83.7 s; remark: max speed 0.886 m/s (above ~0.8 guidance; 0 clipped) | — |
| 20261009_152418 | dev | 5 corridor+north, 0.6 m/s, turn on north | ACCEPT | 78.3 s | — |
| 20261009_152957 | dev | 6 south (2 stop/starts + turn) then corridor, 0.6 m/s | ACCEPT | 91.6 s | — |
| 20261009_153500 | dev | 7 north (turn) then corridor from west (turn), 0.5 m/s | ACCEPT | 73.1 s | — |
| 20261009_154001 | dev | 8 corridor with reverse, then south (right turn), 0.5 m/s | ACCEPT | 80.1 s | — |
| 20261009_154452 | calib | 9 corridor (right turn) + north (east-part stop), 0.5 m/s | ACCEPT | 79.3 s | — |
| 20261009_155140 | calib | 10 corridor (east-part stop) + south (turn, right first), 0.5 m/s | ACCEPT | 73.5 s | — |
| 20261009_155922 | calib | 11 south (2 stops, 180 turn on patch) then north via west strip, 0.5 m/s | ACCEPT | 83.9 s | north visit 1.5 s (near patch edge) |

Note: 20261009_145330 (REJECT) stays in slip_sessions.json as role=dev because slip_sessions.py is hashed in Part A. Every Part B script uses a session only if its role matches AND slip_accept.py exits 0; the loader asserts 8 dev + 3 calib.
| 20261009_185245 | test | T1 replacement, same route | REJECT | check-sessions: health check failed (logger post-processing did not run); 59.3 s (< 60 s, P1) | test 2/3 |
Note: 184810 and 185245 — logger post-processing failed because the logger was started without use_sim_time:=true (wrong command in the instructions; cmd_vel stamped with wall-clock time → clock mismatch in preprocess_data.py). Both were also < 60 s (P1). Fixed for the next recording.
| 20261009_185847 | test | T1 replacement 3, same route | REJECT | P3 1 stop on patch, P4 1 start, P5 no turn on patch (65.1 s) | test 3/3 |
DEVIATION (2026-10-09 19:05, before any test evaluation): test replacements exceed the max of 3 (Part A §3). 2 of the 3 rejections were caused by an instruction error (logger without use_sim_time). Further replacements are allowed until 6 test sessions are accepted; every attempt is logged here and reported in the paper. Acceptance rules unchanged.
| 20261009_190456 | test | T1 replacement 4, same route (timed) | REJECT | P1 55.9 s < 60 s (P2–P6 ok) | test 4 (deviation) |
| 20261009_190827 | test | T1 corridor (2 stops, 180 turn on patch) -> west strip -> north stop, 0.5 m/s, timed | ACCEPT | 66.5 s | — |
| 20261009_191524 | test | T2 west strip -> south (2 stops, turn) -> strip -> corridor, timed | REJECT | P2 corridor visit 1.0 s (< 1.0 s before rounding); 78.5 s, P3–P6 ok | test 5 (deviation) |
| 20261009_191915 | test | T2 west strip -> south (2 stops, 180 turn on patch) -> strip -> corridor stop, timed | ACCEPT | 75.5 s | — |
| 20261009_192336 | test | T3 west strip -> north (2 stops, 180 turn on patch) -> strip -> corridor stop, timed | ACCEPT | 75.9 s | corridor visit 1.1 s |
| 20261009_193020 | test | T4 corridor (2 stops, 180 turn on patch) -> west strip -> south stop, timed | ACCEPT | 60.7 s | near the 60 s limit |
| 20261009_193520 | test | T5 west strip -> north (stop, 180 turn on patch) -> strip -> corridor (2 stops), timed | ACCEPT | 74.3 s | — |
| 20261009_193917 | test | T6 west strip -> south (stop, 180 turn on patch) -> strip -> corridor (2 stops), timed | ACCEPT | 77.1 s | — |
