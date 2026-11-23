#!/usr/bin/env python3
"""Record explicit status for optional checksum-verified real BCD inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from astropy.io import fits


def _sha256_file(path: Path) -> str:
    """Hash a potentially large private input without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_manifest(manifest_path: str | None) -> dict[str, object]:
    """Verify files in a private local manifest without copying or publishing them."""

    if not manifest_path:
        return {
            "status": "missing",
            "reason": "SPITZER_REAL_DATA_MANIFEST is unset; real-BCD acceptance gate remains unmet",
            "files_checked": 0,
        }
    path = Path(manifest_path)
    if not path.is_file():
        return {"status": "missing", "reason": "configured manifest path is unavailable", "files_checked": 0}
    payload = json.loads(path.read_text())
    failures = []
    checked = 0
    for item in payload.get("files", []):
        file_path = Path(item["path"])
        if not file_path.is_file():
            failures.append({"id": item.get("id", file_path.name), "reason": "missing"})
            continue
        digest = _sha256_file(file_path)
        checked += 1
        if digest != item["sha256"]:
            failures.append({"id": item.get("id", file_path.name), "reason": "checksum_mismatch"})
            continue
        try:
            with fits.open(file_path, memmap=True) as hdul:
                data = np.asarray(hdul[0].data)
                finite_fraction = float(np.mean(np.isfinite(data)))
                if data.ndim != 2 or data.size == 0 or finite_fraction == 0.0:
                    failures.append({"id": item.get("id", file_path.name), "reason": "invalid_primary_image"})
        except (OSError, ValueError, TypeError):
            failures.append({"id": item.get("id", file_path.name), "reason": "unreadable_fits"})
    return {
        "status": "failed" if failures else "available",
        "files_checked": checked,
        "failures": failures,
        "privacy": "private paths and FITS content are omitted from this report; only manifest IDs and failure classes are recorded",
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; missing optional data is explicit but not a workflow failure."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=os.environ.get("SPITZER_REAL_DATA_MANIFEST"))
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    report = check_manifest(arguments.manifest)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
