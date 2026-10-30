"""Read-only verification of release files, relocated inference and supplied data."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
manifest=read(ROOT/'PUBLICATION_MANIFEST.json')
for rel,entry in manifest['files'].items():
    p=ROOT/rel
    if sha(p)!=entry['sha256'] or p.stat().st_size!=entry['bytes']:raise RuntimeError('Release differs: '+rel)
frozen=read(ROOT/'FREEZE.json')
for rel,entry in frozen['files'].items():
    if sha(ROOT/rel)!=entry['sha256']:raise RuntimeError('Inference differs: '+rel)
assert sha(ROOT/'FINAL_SELECTION.json')==frozen['selection_sha256']
assert read(ROOT/'FINAL_RESULT.json')['test_correct']==read(ROOT/'evaluation/final_test/test-score.json')['correct']==393
assert read(ROOT/'evaluation/ce_control/test-score.json')['correct']==384
for rel in ('config.json','model.npz'):
    assert sha(ROOT/'checkpoint/baseline'/rel)==read(ROOT/'evaluation/final_test/PRE_TEST_FREEZE.json')['candidate']['checkpoint_sha256']['baseline\\'+rel]
count=0
if (ROOT/'data').exists():
    for rel,entry in read(ROOT/'DATA_SOURCES.json')['files'].items():
        if sha(ROOT/'data'/rel)!=entry['sha256']:raise RuntimeError('Course data differs: '+rel)
        count+=1
print(json.dumps({'passed':True,'release_files_verified':len(manifest['files']),'frozen_files_verified':len(frozen['files']),
    'data_files_verified':count,'read_only':True,'not_a_new_test_call':True}))
