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
