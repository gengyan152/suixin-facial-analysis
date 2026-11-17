"""Verify extended counts against all 330,000 original volunteer ratings."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    data = root/'data/SCUT-FBP5500/SCUT-FBP5500_v2'
    source = data/'All_Ratings.xlsx'
    ratings = defaultdict(list)
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(source) as archive:
        strings = [''.join(e.itertext()) for e in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si', ns)]
        with archive.open('xl/worksheets/sheet1.xml') as stream:
            for _, row in ET.iterparse(stream, events=('end',)):
                if row.tag == '{'+ns['m']+'}row':
                    values = {}
                    for cell in row:
                        value = cell.find('m:v', ns)
                        if value is not None:
                            values[''.join(c for c in cell.get('r') if c.isalpha())] = strings[int(value.text)] if cell.get('t') == 's' else value.text
                    if values.get('B', '').endswith('.jpg'):
                        ratings[values['B']].append(int(values['C']))
                    row.clear()
    assert len(ratings) == 5500 and all(len(r) == 60 for r in ratings.values())
    seen = set()
    max_mean, max_std = 0., 0.
    for path in data.glob('train_test_files/5_folders_cross_validations_files/*/*_aligned_ext.txt'):
        for line in path.read_text().splitlines():
            name, mean, std, *counts = line.split()
            actual = np.array(ratings[name])
            assert np.array_equal(np.bincount(actual, minlength=6)[1:], np.array(counts, dtype=int)), name
            max_mean = max(max_mean, abs(actual.mean()-float(mean)))
            max_std = max(max_std, abs(actual.std()-float(std)))
            seen.add(name)
    assert len(seen) == 5500 and max_mean <= 5.01e-7 and max_std <= 5.01e-7
    report = {'passed': True, 'images': len(seen), 'individual_ratings': sum(map(len, ratings.values())),
              'ratings_per_image': 60, 'all_five_bin_counts_exact': True, 'std_ddof': 0,
              'max_mean_rounding_error': max_mean, 'max_std_rounding_error': max_std,
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    (root/'artifacts/scut_ratings_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
