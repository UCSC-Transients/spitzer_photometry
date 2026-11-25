import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from astropy.io import fits


def _run(root: Path, output: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "check_stage1_real_data.py"),
            "--output",
            str(output),
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_missing_private_manifest_is_explicit_and_nonfatal(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "status.json"
    completed = _run(root, output, "--manifest", "")
    assert completed.returncode == 0
    report = json.loads(output.read_text())
    assert report["status"] == "missing"
    assert report["files_checked"] == 0


def test_empty_or_invalid_manifest_cannot_report_available(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name, content in (("empty.json", '{"files": []}'), ("invalid.json", "not-json")):
        manifest = tmp_path / name
        manifest.write_text(content)
        output = tmp_path / f"{name}.status.json"
        completed = _run(root, output, "--manifest", str(manifest))
        assert completed.returncode != 0
        report = json.loads(output.read_text())
        assert report["status"] == "invalid"
        assert report["files_checked"] == 0


def test_checksum_verified_real_fits_smoke_omits_private_path(tmp_path):
    root = Path(__file__).resolve().parents[1]
    private_dir = tmp_path / "private-inputs"
    private_dir.mkdir()
    fits_path = private_dir / "private_bcd.fits"
    fits.PrimaryHDU(np.ones((5, 7), dtype=np.float32)).writeto(fits_path)
    checksum = hashlib.sha256(fits_path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {"id": "heldout-bcd-001", "path": str(fits_path), "sha256": checksum}
                ]
            }
        )
    )
    output = tmp_path / "status.json"
    completed = _run(root, output, "--manifest", str(manifest))
    assert completed.returncode == 0, completed.stderr
    report_text = output.read_text()
    report = json.loads(report_text)
    assert report["status"] == "available"
    assert report["files_checked"] == 1
    assert str(private_dir) not in report_text
