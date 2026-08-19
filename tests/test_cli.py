import cv2
import numpy as np
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
