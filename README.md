# obs-lsd

obs-lsd reads your webcam, cuts out your head at the neck, centers it on a
black background at a steady size, surrounds it with an animated rainbow
halo, and publishes the result as a virtual camera. Point Zoom (or OBS, as a
Video Capture Device source) at it instead of your real camera.

Requires Python 3.12 and macOS. Run `make install-dev models && make run` to
set up the environment, fetch the MediaPipe models, and start a live preview
with the virtual camera. OBS's virtual camera system extension provides the
video device this tool publishes to — install OBS and, once, launch it and
click Start Virtual Camera so macOS registers the device. macOS will also
prompt for camera access the first time obs-lsd runs; grant it in System
Settings > Privacy & Security > Camera.
