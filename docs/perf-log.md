# Speed rounds on pequod

Bench: `make bench` on the first 10s of the test clip (301 frames,
1620x1080 in, 1280x720 out), rendered on pequod (M2 Max). ms are per-frame
means from the rig's `metrics.json`. Target 16.7ms (60 fps).

| round | change | total ms | segment | halo | compose | fps | kept |
|------:|--------|---------:|--------:|-----:|--------:|----:|------|
| 0 | baseline, main at 97179e1 | 163.2 | 138.5 | 14.2 | 9.7 | 6.1 | - |
| 1 | MediaPipe models on the Metal GPU delegate (RGBA input; SRGB aborts the process) | 48.6 | 23.7 | 14.5 | 9.6 | 20.6 | yes |
