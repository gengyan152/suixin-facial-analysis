"""Download the author's unmodified SCUT extended archive and verify ZIP CRCs."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile
import requests


def main():
    root = Path(__file__).resolve().parent
    url = 'https://drive.usercontent.google.com/download?id=1ObjJ8RwYBVI71Eyo_wsM1dWzI0JbhotA&export=download&confirm=t'
    archive = root/'data/SCUT-FBP5500.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    digest = hashlib.sha256()
    size = 0
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        expected = int(response.headers['Content-Length'])
        with archive.with_suffix('.zip.part').open('wb') as handle:
            for chunk in response.iter_content(1024*1024):
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if size % (32*1024*1024) == 0:
                    print(f'Downloaded {size}/{expected} bytes', flush=True)
    assert size == expected, (size, expected)
    archive.with_suffix('.zip.part').rename(archive)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None, 'ZIP CRC failure'
        target = archive.parent.resolve()
        assert all((target/n).resolve().is_relative_to(target) for n in z.namelist())
        z.extractall(target)
        entries = len(z.infolist())
    report = {'source': url, 'started_utc': started, 'bytes': size,
              'sha256': digest.hexdigest(), 'zip_crc_valid': True, 'entries': entries,
              'wall_seconds': time.perf_counter()-start}
    (root/'artifacts/scut_download.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report, flush=True)


if __name__ == '__main__':
    main()
