import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from obs_lsd.cli import cli


def test_cli_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_headless_run_dumps_frame(tmp_path, monkeypatch):
    # fake pipeline so no models are needed
    import obs_lsd.cli as climod

    class FakePipeline:
        timings = {"total": 1.0}  # noqa: RUF012 -- test fixture, single instance
        frozen = False

        def __init__(self, cfg, segmenter=None):
            self.cfg = cfg

        def step(self, frame, t):
            out = np.zeros((self.cfg.height, self.cfg.width, 3), np.uint8)
            out[0, 0] = 255
            return out

        def close(self):
            pass

    monkeypatch.setattr(climod, "Pipeline", FakePipeline)
    src = tmp_path / "in.png"
    cv2.imwrite(str(src), np.zeros((20, 30, 3), np.uint8))
    dump = tmp_path / "out.png"
    r = CliRunner().invoke(
        cli,
        [
            "--input",
            str(src),
            "--seconds",
            "0.2",
            "--fps",
            "50",
            "--no-vcam",
            "--no-preview",
            "--dump-frame",
            str(dump),
            "--width",
            "64",
            "--height",
            "36",
        ],
    )
    assert r.exit_code == 0, r.output
    img = cv2.imread(str(dump))
    assert img.shape == (36, 64, 3) and img[0, 0].max() == 255


@pytest.fixture
def still_png(tmp_path):
    p = tmp_path / "in.png"
    cv2.imwrite(str(p), np.zeros((20, 30, 3), np.uint8))
    return p


def _headless_args(src, dump, extra=()):
    return [
        "--input",
        str(src),
        "--seconds",
        "5",
        "--fps",
        "100",
        "--no-vcam",
        "--no-preview",
        "--dump-frame",
        str(dump),
        "--width",
        "64",
        "--height",
        "36",
        *extra,
    ]


class _FakePipeline:
    """Pipeline stand-in: no models, and step() can be made to raise."""

    timings = {"total": 1.0}  # noqa: RUF012 -- test fixture, single instance
    frozen = False
    raise_after = None
    close_raises = False
    closed = False

    def __init__(self, cfg, segmenter=None):
        self.cfg = cfg
        self.steps = 0

    def step(self, frame, t):
        self.steps += 1
        if self.raise_after is not None and self.steps > self.raise_after:
            raise KeyboardInterrupt
        out = np.zeros((self.cfg.height, self.cfg.width, 3), np.uint8)
        out[0, 0] = 255
        return out

    def close(self):
        type(self).closed = True
        if self.close_raises:
            raise RuntimeError("close blew up")


def test_keyboard_interrupt_exits_zero_and_still_dumps(still_png, tmp_path, monkeypatch):
    import obs_lsd.cli as climod

    monkeypatch.setattr(_FakePipeline, "raise_after", 2)
    monkeypatch.setattr(climod, "Pipeline", _FakePipeline)
    dump = tmp_path / "out.png"
    r = CliRunner().invoke(cli, _headless_args(still_png, dump))
    assert r.exit_code == 0, r.output
    assert "Traceback" not in r.output
    assert cv2.imread(str(dump)).shape == (36, 64, 3)


def test_failing_cleanup_still_releases_the_camera(still_png, tmp_path, monkeypatch):
    import obs_lsd.cli as climod

    closed = []

    class FakeSource:
        def __init__(self, *a, **k):
            pass

        def __iter__(self):
            yield np.zeros((20, 30, 3), np.uint8), 0.0

        def close(self):
            closed.append(True)

    monkeypatch.setattr(_FakePipeline, "close_raises", True)
    monkeypatch.setattr(climod, "Pipeline", _FakePipeline)
    monkeypatch.setattr(climod, "FrameSource", FakeSource)
    r = CliRunner().invoke(cli, _headless_args(still_png, tmp_path / "out.png"))
    assert r.exit_code == 0, r.output
    assert closed == [True], "src.close() skipped when an earlier cleanup raised"


def test_video_input_to_video_output(tmp_path, monkeypatch):
    import obs_lsd.cli as climod

    class FakePipeline:
        timings = {"total": 1.0}  # noqa: RUF012
        frozen = False

        def __init__(self, cfg, segmenter=None):
            self.cfg = cfg
            self.n = 0

        def step(self, frame, t):
            self.n += 1
            out = np.zeros((self.cfg.height, self.cfg.width, 3), np.uint8)
            out[:, :, 1] = min(255, self.n * 10)
            return out

        def close(self):
            pass

    monkeypatch.setattr(climod, "Pipeline", FakePipeline)
    clip = tmp_path / "in.mp4"
    w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for _ in range(12):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    out = tmp_path / "out.mp4"
    r = CliRunner().invoke(
        cli,
        [
            "--input",
            str(clip),
            "--output",
            str(out),
            "--no-vcam",
            "--no-preview",
            "--width",
            "64",
            "--height",
            "48",
        ],
    )
    assert r.exit_code == 0, r.output
    cap = cv2.VideoCapture(str(out))
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        assert f.shape == (48, 64, 3)
    assert n == 12
    assert abs(cap.get(cv2.CAP_PROP_FPS) - 10.0) < 0.1  # output fps follows the input file
