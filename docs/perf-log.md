# Speed rounds on pequod

Bench: `make bench` on the first 10s of the test clip (301 frames,
1620x1080 in, 1280x720 out), rendered on pequod (M2 Max). ms are per-frame
means from the rig's `metrics.json`. Target 16.7ms (60 fps).

| round | change | total ms | segment | halo | compose | fps | kept |
|------:|--------|---------:|--------:|-----:|--------:|----:|------|
| 0 | baseline, main at 97179e1 | 163.2 | 138.5 | 14.2 | 9.7 | 6.1 | - |
| 1 | MediaPipe models on the Metal GPU delegate (RGBA input; SRGB aborts the process) | 48.6 | 23.7 | 14.5 | 9.6 | 20.6 | yes |
| 2 | shrink frame to 320 wide before inference, upscale masks | 57.8 | 32.8 | 14.7 | 9.6 | 17.3 | no: the two float resizes cost more than the GPU's own |
| 3 | halfplane by polygon fill, in-place mask maths, box by cv2.moments instead of np.nonzero | 43.0 | 18.6 | 14.5 | 9.2 | 23.3 | yes |
| 4 | halo: in-place cv2 ops, cached grids, moments for the centre, no final clip; output bit-identical | 38.2 | 18.3 | 9.7 | 9.6 | 26.1 | yes |
| 5 | compose in uint8 via cv2.blendLinear; spill added in uint8 | 29.5 | 15.9 | 9.4 | 3.4 | 33.9 | yes |
| 5b | shrink frame to 320 wide before inference, linear resize (round 2 again, without INTER_AREA) | (same bench) | 15.9 | | | | yes: segment 18.3 -> 15.9 |
| 6 | segmenter on a worker thread, one frame behind the loop (`--seg-thread`, default on) | 14.9 | 0.1 (wait) | 10.1 | 3.9 | 67.0 | yes |
| 7 | segmenter post-processing at inference size (worker 16 -> 10.6 ms/frame; bench loop unchanged) | 14.4 | 0.1 | 9.9 | 3.6 | 69.3 | yes |
| 8 | halo: drop the three modulo passes; remap's BORDER_WRAP already tiles the texture | 12.8 | 0.1 | 7.9 | 4.0 | 78.0 | yes |
| 9 | halo: HSV via one cv2.merge, inside zeroed on V only; output identical | 11.8 | 0.1 | 7.2 | 3.7 | 84.8 | yes |
