from experiment_io import PROJECT,read,sha
PREVIOUS=PROJECT/"V3_Block_Regularization"
EARLIER=(PREVIOUS,PROJECT/"V3_Relative_Features",PROJECT/"V3_Arithmetic_Reliability",PROJECT/"V3_OOF_Ensemble")
def snapshot(include_validation=False):
    result={}
    for folder in EARLIER:
        m=read(folder/"ARTIFACT_MANIFEST.json");count=0
        for rel,meta in m["files"].items():
            if not include_validation and "validation" in rel.lower():continue
            assert sha(folder/rel)==meta["sha256"],rel
            count+=1
        result[folder.name]={"manifest_sha256":sha(folder/"ARTIFACT_MANIFEST.json"),"files_verified":count}
    return result
