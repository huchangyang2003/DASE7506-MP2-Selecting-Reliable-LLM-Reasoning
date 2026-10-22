"""Read-only integrity checks for the publication tree and optional supplied data."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((ROOT/'PUBLICATION_MANIFEST.json').read_text())
for rel,entry in manifest['files'].items():
    p=ROOT/rel
    if sha(p)!=entry['sha256'] or p.stat().st_size!=entry['bytes']:raise RuntimeError('Release file differs: '+rel)
frozen=json.loads((ROOT/'FREEZE.json').read_text())
for rel,entry in frozen['files'].items():
    if sha(ROOT/rel)!=entry['sha256']:raise RuntimeError('Frozen inference differs: '+rel)
if sha(ROOT/'FINAL_SELECTION.json')!=frozen['selection_sha256']:raise RuntimeError('Historical selection differs')
data_count=0
if (ROOT/'data').exists():
    data=json.loads((ROOT/'DATA_SOURCES.json').read_text())
    for rel,entry in data['files'].items():
        if sha(ROOT/'data'/rel)!=entry['sha256']:raise RuntimeError('Course data differs: '+rel)
    data_count=len(data['files'])
print(json.dumps({'release_files_verified':len(manifest['files']),'frozen_files_verified':len(frozen['files']),'data_files_verified':data_count,'read_only':True}))

