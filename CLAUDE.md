# obs-lsd

A macOS command-line tool that reads a webcam, isolates the user's head at
the neck, centers and stabilizes it on a black frame, wraps it in an
animated psychedelic halo, and publishes the result as a virtual camera for
Zoom and OBS.

## Modules (src/obs_lsd/)

- `capture.py` — camera / video-file frame source
- `segment.py` — head mask + chin line via MediaPipe
- `track.py` — smoothing + camera-to-output transform
- `halo.py` — ray renderer (numpy backend)
- `compose.py` — black + halo + head compositing
- `output.py` — pyvirtualcam sink, cv2 preview, key handling
- `pipeline.py` — wires the above modules together
- `cli.py` — click entry point (`obs-lsd`)

## Make targets

`venv install-dev test lint format models smoke run`

## Working here

TDD: write a failing test, then the code, then run it. Run `make smoke`
before claiming a change works end to end — passing unit tests alone is not
enough. mediapipe must stay `<1.0` on macOS; 1.x crashes.
