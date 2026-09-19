"""The real-data fetcher must reject corrupt or inconsistent upstream bytes."""

import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest


@pytest.fixture
def fetcher():
    path = Path(__file__).resolve().parents[2] / "scripts/fetch_mapdn_data.py"
    spec = importlib.util.spec_from_file_location("mapdn_data_fetch_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Response(io.BytesIO):
    def __init__(self, data, *, status=206, content_range="bytes 0-3/4", etag="fixed"):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range, "ETag": etag}


@pytest.mark.parametrize("values", [
    {"status": 200}, {"content_range": "bytes 1-4/4"}, {"etag": None},
    {"data": b"abc"}, {"data": b"abcde"},
])
def test_reject_ignored_wrong_or_short_ranges(fetcher, monkeypatch, values):
    response = Response(**{"data": b"abcd", **values})
    monkeypatch.setattr(fetcher.urllib.request, "urlopen", lambda *a, **k: response)
    reader = fetcher.RangeReader("https://example.org/archive", 4)
    with pytest.raises(ValueError):
        reader.read(4)
    assert reader.tell() == 0


def test_reject_identity_change_between_ranges(fetcher, monkeypatch):
    replies = iter([Response(b"ab", content_range="bytes 0-1/4"),
                    Response(b"cd", content_range="bytes 2-3/4", etag="changed")])
    monkeypatch.setattr(fetcher.urllib.request, "urlopen", lambda *a, **k: next(replies))
    reader = fetcher.RangeReader("https://example.org/archive", 4)
    assert reader.read(2) == b"ab"
    with pytest.raises(ValueError, match="identity"):
        reader.read(2)


def archive_reader(fetcher, *, corrupt=False):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in fetcher.MEMBERS:
            archive.writestr(fetcher.PREFIX + name, b"fixture-member-content")
    data = buffer.getvalue()
    if corrupt:
        data = data.replace(b"fixture-member-content", b"corrupt-member-content", 1)
    reader = io.BytesIO(data)
    reader.transferred_bytes = len(data)
    reader.etag = "fixture"
    return reader


def test_member_inventory_and_existing_output_preserved(fetcher, monkeypatch, tmp_path):
    monkeypatch.setattr(fetcher, "RangeReader", lambda: archive_reader(fetcher))
    out = tmp_path / "data"
    receipt = fetcher.fetch(out)
    assert receipt["checks_passed"] and receipt["member_crc_verified"]
    assert not receipt["whole_archive_sha256_verified"]
    assert set(receipt["files"]) == set(fetcher.MEMBERS)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        fetcher.fetch(out)
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}


def test_crc_failure_leaves_failed_receipt_and_partial(fetcher, monkeypatch, tmp_path):
    monkeypatch.setattr(fetcher, "RangeReader", lambda: archive_reader(fetcher, corrupt=True))
    out = tmp_path / "data"
    with pytest.raises(zipfile.BadZipFile):
        fetcher.fetch(out)
    receipt = json.loads((out / "download_receipt.json").read_text())
    assert not receipt["checks_passed"] and not receipt["member_crc_verified"]
    assert not (out / fetcher.MEMBERS[0]).exists()
    assert (out / (fetcher.MEMBERS[0] + ".part")).exists()
