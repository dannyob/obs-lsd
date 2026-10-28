"""Command-line interface for obs-lsd."""

from __future__ import annotations

import contextlib
import functools

import click

from obs_lsd.capture import CameraError, FrameSource, list_cameras
from obs_lsd.halo import HaloParams
from obs_lsd.pipeline import Pipeline, PipelineConfig


@click.command()
@click.version_option()
@click.option("--camera", default=None, help="Camera index or name substring (default 0)")
@click.option("--input", "input_", default=None, help="Video/image file instead of a camera")
@click.option("--list-cameras", "list_cams", is_flag=True, help="List cameras and exit")
@click.option("--width", default=1280, show_default=True)
@click.option("--height", default=720, show_default=True)
@click.option("--fps", default=30.0, show_default=True)
@click.option(
    "--head-frac", default=0.55, show_default=True, help="Head height as fraction of frame"
)
@click.option(
    "--seg-thread/--no-seg-thread",
    default=True,
    show_default=True,
    help="Segment on a worker thread, one frame behind the loop (faster); "
    "--no-seg-thread segments inline with no lag",
)
@click.option(
    "--halo-thread/--no-halo-thread",
    default=True,
    show_default=True,
    help="Render the halo on a worker thread, one frame behind the loop (faster); "
    "--no-halo-thread renders inline",
)
@click.option(
    "--float",
    "float_amp",
    default=0.03,
    show_default=True,
    help="Gentle drift of the head about the centre, as a fraction of frame height; 0 disables",
)
@click.option("--no-vcam", is_flag=True, help="Do not open the virtual camera")
@click.option("--no-preview", is_flag=True, help="No preview window")
@click.option("--seconds", default=None, type=float, help="Stop after N seconds")
@click.option("--dump-frame", default=None, type=click.Path(), help="Write the last frame as PNG")
@click.option(
    "--output", "output_path", default=None, type=click.Path(), help="Write every frame to an mp4"
)
@click.option("--stats", is_flag=True, help="Print per-stage timings once a second")
@click.option("--halo-reach", default=HaloParams.reach, show_default=True)
@click.option("--halo-density", default=HaloParams.density, show_default=True)
@click.option("--halo-speed", default=HaloParams.speed, show_default=True)
@click.option(
    "--halo-rotate", default=HaloParams.rotate, show_default=True, help="Ray spin, rev/second"
)
@click.option("--halo-flicker", default=HaloParams.flicker, show_default=True)
@click.option("--halo-hue-speed", default=HaloParams.hue_speed, show_default=True)
@click.option(
    "--halo-spill",
    default=HaloParams.spill_gain,
    show_default=True,
    help="Halo colour wrap onto the head edge",
)
@click.option("--halo-scale", default=HaloParams.scale, show_default=True)
def cli(
    camera,
    input_,
    list_cams,
    width,
    height,
    fps,
    head_frac,
    seg_thread,
    halo_thread,
    float_amp,
    no_vcam,
    no_preview,
    seconds,
    dump_frame,
    output_path,
    stats,
    halo_reach,
    halo_density,
    halo_speed,
    halo_rotate,
    halo_flicker,
    halo_hue_speed,
    halo_spill,
    halo_scale,
):
    """Glowing floating-head virtual camera."""
    if list_cams:
        for i, n in enumerate(list_cameras()):
            click.echo(f"{i}: {n}")
        return
    halo = HaloParams(
        reach=halo_reach,
        density=halo_density,
        speed=halo_speed,
        rotate=halo_rotate,
        flicker=halo_flicker,
        hue_speed=halo_hue_speed,
        spill_gain=halo_spill,
        scale=halo_scale,
    )
    cfg = PipelineConfig(
        width=width,
        height=height,
        head_frac=head_frac,
        halo=halo,
        float_amp=float_amp,
        seg_lag=1 if seg_thread else 0,
        halo_lag=1 if halo_thread else 0,
    )
    code = run(
        cfg,
        input_ if input_ else camera,
        fps,
        no_vcam,
        no_preview,
        seconds,
        dump_frame,
        stats,
        output_path=output_path,
    )
    if code:
        raise SystemExit(code)


def run(
    cfg: PipelineConfig,
    source_spec,
    fps,
    no_vcam,
    no_preview,
    seconds,
    dump_frame,
    stats,
    output_path: str | None = None,
) -> int:
    # a headless replay (no preview, no vcam) runs flat out; anything live is paced
    realtime = not (no_vcam and no_preview)
    try:
        src = FrameSource(source_spec, cfg.width, cfg.height, fps, realtime=realtime)
    except CameraError as e:
        click.echo(f"error: {e}", err=True)
        return 1
    try:
        pipe = Pipeline(cfg)
    except Exception as e:  # noqa: BLE001 -- ModelsMissing and friends
        click.echo(f"error: {e}", err=True)
        src.close()
        return 1
    vcam = preview = sink = None
    try:
        if output_path:
            from obs_lsd.output import VideoFileSink

            sink = VideoFileSink(output_path, cfg.width, cfg.height, src.file_fps or fps)
        if not no_vcam:
            from obs_lsd.output import VirtualCam, VirtualCamMissing

            try:
                vcam = VirtualCam(cfg.width, cfg.height, fps)
                click.echo(f"virtual camera: {vcam.device}", err=True)
            except VirtualCamMissing as e:
                click.echo(f"error: {e}", err=True)
                return 1
        if not no_preview:
            from obs_lsd.output import Preview

            preview = Preview()
        last = None
        last_stat = 0.0
        try:
            for frame, t in src:
                out = pipe.step(frame, t)
                last = out
                if vcam:
                    vcam.send(out)
                if sink:
                    sink.write(out)
                if preview:
                    action = preview.show(out)
                    if action == "quit":
                        break
                    _apply_action(action, pipe)
                if stats and t - last_stat >= 1.0:
                    last_stat = t
                    click.echo(
                        " ".join(f"{k}={v:.1f}ms" for k, v in pipe.timings.items()), err=True
                    )
                if seconds is not None and t >= seconds:
                    break
        except KeyboardInterrupt:
            pass  # ctrl-C is how you stop a live run; not an error
        if dump_frame and last is not None:
            import cv2

            cv2.imwrite(dump_frame, last)
        return 0
    finally:
        # a failure in any one of these must not leave the camera or the
        # virtual camera device held open
        cleanups = []
        if preview:
            cleanups.append(preview.close)
        if sink:
            cleanups.append(sink.close)
        if vcam:
            cleanups.append(functools.partial(vcam.__exit__, None, None, None))
        cleanups += [pipe.close, src.close]
        for cleanup in cleanups:
            with contextlib.suppress(Exception):
                cleanup()


def _apply_action(action: str | None, pipe: Pipeline) -> None:
    if action is None:
        return
    p = pipe.halo.p
    if action == "size+":
        pipe.tracker.target_frac = min(0.95, pipe.tracker.target_frac + 0.05)
    elif action == "size-":
        pipe.tracker.target_frac = max(0.1, pipe.tracker.target_frac - 0.05)
    elif action == "hue+":
        p.hue_speed *= 1.25
    elif action == "hue-":
        p.hue_speed /= 1.25
    elif action == "reach+":
        p.reach *= 1.15
    elif action == "reach-":
        p.reach /= 1.15
    elif action == "freeze":
        pipe.frozen = not pipe.frozen
    elif action == "print":
        click.echo(f"head_frac={pipe.tracker.target_frac:.2f} {p}", err=True)


if __name__ == "__main__":
    cli()
