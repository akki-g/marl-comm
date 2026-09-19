"""Fetch the four real case33 inputs from a pinned upstream ZIP using byte ranges.

The full multi-feeder ZIP is not downloaded. ZIP CRCs validate every extracted
member; a receipt records its SHA256 and the immutable upstream dataset revision.
The upstream whole-ZIP digest is recorded, not claimed to have been verified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


REVISION = "a41d8c269e402a827d36d9a153a4bce73a990e53"
URL = (
    "https://huggingface.co/datasets/hsvgbkhgbv/"
    f"Multi-Agent-Power-Distribution-Networks/resolve/{REVISION}/voltage_control_data.zip"
)
ARCHIVE_BYTES = 3763536396
ARCHIVE_SHA256 = "002de887013e8a2dbd2d79e5b6e2ed2c00584f3cf1ae81947f986061cfd81de2"
PREFIX = "voltage_control_data/case33_3min_final/"
MEMBERS = ("load_active.csv", "load_reactive.csv", "pv_active.csv", "model.p")


class RangeReader(io.RawIOBase):
    """A seekable read-only view that rejects ignored, short or inconsistent ranges."""

    def __init__(self, url=URL, size=ARCHIVE_BYTES):
        super().__init__()
        self.url, self.size, self.position = url, size, 0
        self.transferred_bytes = 0
        self.etag = None

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek origin.")
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if position < 0:
            raise ValueError("Cannot seek before the archive.")
        self.position = position
        return position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        requested = f"bytes {self.position}-{end - 1}/{self.size}"
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-{end - 1}"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206 or response.headers.get("Content-Range") != requested:
                raise ValueError("Server did not honor the exact pinned archive byte range.")
            etag = response.headers.get("ETag")
            if not etag or (self.etag is not None and self.etag != etag):
                raise ValueError("Archive identity changed between range requests.")
            self.etag = etag
            data = response.read(end - self.position + 1)
        if len(data) != end - self.position:
            raise ValueError("Truncated or oversized archive range.")
        self.transferred_bytes += len(data)
        self.position = end
        return data


def fetch(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": 1,
        "started_utc": datetime.now(UTC).isoformat(),
        "source_url": URL,
        "dataset_revision": REVISION,
        "upstream_archive_bytes": ARCHIVE_BYTES,
        "upstream_archive_sha256": ARCHIVE_SHA256,
        "whole_archive_sha256_verified": False,
        "member_crc_verified": False,
        "files": {},
    }
    try:
        with RangeReader() as remote, zipfile.ZipFile(remote) as archive:
            names = archive.namelist()
            for name in MEMBERS:
                member = PREFIX + name
                if names.count(member) != 1:
                    raise ValueError(f"Require one upstream member: {member}")
                info = archive.getinfo(member)
                partial = out_dir / (name + ".part")
                digest = hashlib.sha256()
                count = 0
                with archive.open(info) as source, partial.open("xb") as target:
                    while block := source.read(8 * 1024**2):
                        target.write(block)
                        digest.update(block)
                        count += len(block)
                if count != info.file_size:
                    raise ValueError(f"Extracted size differs: {member}")
                partial.rename(out_dir / name)
                receipt["files"][name] = {
                    "member": member,
                    "bytes": count,
                    "compressed_bytes": info.compress_size,
                    "zip_crc32": f"{info.CRC:08x}",
                    "sha256": digest.hexdigest(),
                }
                print(f"Verified {name}: {count} bytes", flush=True)
            receipt.update({"member_crc_verified": True, "checks_passed": True,
                            "transferred_bytes": remote.transferred_bytes,
                            "range_etag": remote.etag})
    except Exception as error:
        receipt.update({"checks_passed": False, "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        receipt["finished_utc"] = datetime.now(UTC).isoformat()
        with (out_dir / "download_receipt.json").open("x") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    fetch(args.out_dir)


if __name__ == "__main__":
    main()
