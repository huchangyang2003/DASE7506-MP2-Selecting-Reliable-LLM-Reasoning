from experiment_io import PROJECT,read,sha
EARLIER=tuple(PROJECT/n for n in ("V3_Block_Regularization","V3_Relative_Features","V3_Arithmetic_Reliability","V3_OOF_Ensemble","V4_Reference_Alignment"))
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
