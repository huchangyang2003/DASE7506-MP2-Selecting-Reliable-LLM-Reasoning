"""Verify the published files and original R1 identity, without prediction or scoring."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    manifest=read(ROOT/'PUBLICATION_MANIFEST.json')
    for rel,entry in manifest['files'].items():
        path=ROOT/rel
        assert path.resolve().is_relative_to(ROOT) and path.is_file(),rel
        assert sha(path)==entry['sha256'] and path.stat().st_size==entry['bytes'],rel
    freeze=read(ROOT/'FREEZE.json')
    assert freeze['not_a_pre_test_freeze']
    for rel,entry in freeze['files'].items():assert sha(ROOT/rel)==entry['sha256'],rel
    assert sha(ROOT/'FINAL_SELECTION.json')==freeze['selection_sha256']
    original=read(ROOT/'evaluation/final_test/PRE_TEST_FREEZE.json')
    assert sha(ROOT/'evaluation/final_test/PRE_TEST_FREEZE.json')==freeze['original_pre_test_freeze_sha256']
    matched={}
    for historical,digest in original['frozen_files'].items():
        name=historical.replace('\\','/')
        if '/R1_portable/' in name:
            rel=name.split('/R1_portable/',1)[1]
            if rel in freeze['files']:
                assert sha(ROOT/rel)==digest,rel
                matched[rel]=digest
    assert set(matched)==set(freeze['files']) and len(matched)==15
    ledger=read(ROOT/'evaluation/final_test/TEST_LEDGER.json')
    assert ledger['prediction_calls']==ledger['score_calls']==1 and ledger['status']=='completed'
    assert ledger['pre_test_freeze_sha256']==sha(ROOT/'evaluation/final_test/PRE_TEST_FREEZE.json')
    assert ledger['frozen_prediction_sha256']==sha(ROOT/'evaluation/final_test/test-predictions.json')
    assert ledger['official_score_sha256']==sha(ROOT/'evaluation/final_test/test-score.json')
    score=read(ROOT/'evaluation/final_test/test-score.json')
    result=read(ROOT/'FINAL_RESULT.json')
    assert score['correct']==result['test_correct']==397 and score['n']==result['test_n']==467
    assert score['accuracy']==result['test_accuracy']
    selection=read(ROOT/'FINAL_SELECTION.json')
    assert selection['candidate']=='R1' and selection['residual_l2']==.03
    assert selection['pooling_alpha']==selection['temperature']==1 and selection['neural_weight']==0
    provenance=read(ROOT/'SOURCE_PROVENANCE.json')
    assert sha(ROOT/'REPORT.pdf')==provenance['approved_report_sha256']
    print(json.dumps({'passed':True,'release_files_verified':len(manifest['files']),
          'original_pre_test_inference_files_verified':len(matched),
          'original_official_test_correct':397,'test_n':467,
          'prediction_calls_added':0,'scoring_calls_added':0,'fitting_calls_added':0,
          'approved_report_unchanged':True,'read_only':True},indent=2))

if __name__=='__main__':main()
