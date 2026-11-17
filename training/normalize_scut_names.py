"""Restore original SCUT IDs using paired author-supplied split rows."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parent
    data = root/'data/SCUT-FBP5500/SCUT-FBP5500_v2'
    mapping, files = {}, {}
    with zipfile.ZipFile(root/'data/SCUT-FBP5500.zip') as archive:
        for fold in range(1, 6):
            for split in ['train', 'test']:
                relative = f'train_test_files/5_folders_cross_validations_files/cross_validation_{fold}/{split}_{fold}'
                prefix = 'SCUT-FBP5500/SCUT-FBP5500_v2/'+relative
                original = archive.read(prefix+'.txt').decode().splitlines()
                extended = archive.read(prefix+'_aligned_ext.txt').decode().splitlines()
                mtcnn = archive.read(prefix+'_mtcnn.txt').decode().splitlines()
                assert len(original) == len(extended) == len(mtcnn)
                normalized = []
                for raw, ext, mt in zip(original, extended, mtcnn, strict=True):
                    raw, ext, mt = raw.split(), ext.split(), mt.split()
                    assert len(ext) == 8 and raw[0] == mt[0] and ext[1:] == mt[1:]
                    assert float(raw[1]) == float(ext[1])
                    assert sum(map(int, ext[3:])) == 60
                    assert ext[0] not in mapping or mapping[ext[0]] == raw[0]
                    mapping[ext[0]] = raw[0]
                    normalized.append(' '.join([raw[0], *ext[1:]]))
                files[relative+'_aligned_ext.txt'] = '\n'.join(normalized)+'\n'
        assert len(mapping) == len(set(mapping.values())) == 5500
        assert len(list((data/'aligned').glob('*.jpg'))) == 5500
        for old, new in mapping.items():
            source, target = data/'aligned'/old, data/'aligned'/new
            assert source.is_file() and not target.exists()
        for old, new in mapping.items():
            (data/'aligned'/old).rename(data/'aligned'/new)
        for name, content in files.items():
            (data/name).write_text(content, encoding='utf-8')
        hashes = {name: hashlib.sha256((data/name).read_bytes()).hexdigest() for name in files}
    report = {'mapping_source': 'Paired original, aligned_ext and mtcnn rows in author ZIP',
              'annotation_values_unchanged': True, 'all_rating_counts_sum': 60,
              'image_count': 5500, 'filenames': mapping, 'normalized_split_sha256': hashes}
    (root/'artifacts/scut_name_mapping.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Verified and normalized 5500 IDs and 10 extended splits; annotation values unchanged.')


if __name__ == '__main__':
    main()
