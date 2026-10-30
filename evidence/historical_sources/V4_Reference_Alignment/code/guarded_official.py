"""Guard the unchanged official runners, always on explicit validation."""
import os
from pathlib import Path
import runpy
import sys
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key] = "4"
from experiment_io import guard, save

if __name__ == "__main__":
    runner, arguments = sys.argv[1], sys.argv[2:]
    if runner not in ("predict.py","score_predictions.py") or "--split" not in arguments or arguments[arguments.index("--split")+1] != "validation":
        raise ValueError("Explicit official validation command required")
    role = "validation_predict" if runner == "predict.py" else "validation_score"
    opened, denied = guard(role)
    sys.argv = [runner]+arguments
    runpy.run_path(str(Path(__file__).parent/runner), run_name="__main__")
    out = Path(arguments[arguments.index("--output")+1])
    save(out.with_suffix(".access.json"), {"role":role,"opened":sorted(opened),"denied":denied})
