"""Inventory and verify the local RDD2022 split without modifying source data."""
from pathlib import Path
import hashlib, json, collections, concurrent.futures
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'evaluation'
OUT.mkdir(parents=True, exist_ok=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inspect_image(path):
    label = path.parent.parent / 'labels' / (path.stem + '.txt')
    issues = []
    image = cv2.imread(str(path))
    shape = list(image.shape[:2]) if image is not None else None
    if shape is None: issues.append('unreadable image')
    boxes = []
    if not label.exists(): issues.append('missing label')
    else:
        for ln, line in enumerate(label.read_text().splitlines(), 1):
            if not line.strip(): continue
            try:
                a = np.array([float(x) for x in line.split()])
                valid = len(a) == 5 and np.isfinite(a).all()
                valid = valid and a[0] == int(a[0]) and 0 <= a[0] < 5
                valid = valid and (a[1:] >= 0).all() and (a[1:] <= 1).all() and (a[3:] > 0).all()
                if not valid: issues.append(f'invalid label row {ln}')
                else: boxes.append(a.tolist())
            except Exception: issues.append(f'unparseable label row {ln}')
    if len({tuple(x) for x in boxes}) != len(boxes): issues.append('duplicate annotation')
    return {'file':str(path.relative_to(ROOT)), 'name':path.name, 'country':path.stem.split('_')[0],
            'sha256':sha(path), 'label_sha256':sha(label) if label.exists() else None,
            'shape':shape, 'boxes':boxes, 'issues':issues}

def main():
    audit = {'splits':{}, 'overlaps':{}, 'model_sha256':sha(ROOT/'gopher-ai_v1.pt')}
    records = {}
    for split in ['train','val','test']:
        paths = sorted((ROOT/'rdd2022/RDD_SPLIT'/split/'images').glob('*.jpg'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(inspect_image, paths))
        records[split] = rows
        hashes = collections.Counter(x['sha256'] for x in rows)
        audit['splits'][split] = {'images':len(rows), 'objects':sum(len(x['boxes']) for x in rows),
            'backgrounds':sum(not x['boxes'] for x in rows),
            'classes':dict(collections.Counter(int(b[0]) for x in rows for b in x['boxes'])),
            'countries':dict(collections.Counter(x['country'] for x in rows)),
            'issues':[{'file':x['file'],'issues':x['issues']} for x in rows if x['issues']],
            'duplicate_image_excess':sum(n-1 for n in hashes.values() if n>1),
            'orphan_labels':len(set(p.stem for p in (ROOT/'rdd2022/RDD_SPLIT'/split/'labels').glob('*.txt'))-set(p.stem for p in paths))}
        print(split, audit['splits'][split], flush=True)
    for a,b in [('train','val'),('train','test'),('val','test')]:
        shared_names = set(x['name'] for x in records[a]) & set(x['name'] for x in records[b])
        shared_hash = set(x['sha256'] for x in records[a]) & set(x['sha256'] for x in records[b])
        audit['overlaps'][a+'_'+b] = {'shared_names':sorted(shared_names),'shared_content_hashes':sorted(shared_hash)}
    manifest = '\n'.join(f"{s}|{x['name']}|{x['sha256']}|{x['label_sha256']}" for s,rows in records.items() for x in rows)
    audit['manifest_sha256'] = hashlib.sha256(manifest.encode()).hexdigest()
    (OUT/'data_manifest.txt').write_text(manifest, encoding='utf-8')
    (OUT/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    (OUT/'data_records.json').write_text(json.dumps(records), encoding='utf-8')
    print('Overlap summary', {k:{j:len(v) for j,v in x.items()} for k,x in audit['overlaps'].items()}, flush=True)

if __name__ == '__main__': main()
