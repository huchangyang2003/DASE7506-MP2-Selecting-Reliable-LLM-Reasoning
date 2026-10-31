# Compact experimental evidence

`results.json` retains aggregate CE-residual, neural and five-weight fusion results used by the report. It also supplies the report builder. `SELECTION.json` records the primary-only choice, confirmation role, fixed L2 values and final eligibility/ranking rule. These compact exports are made after scoring and do not replace or claim to be original pre-test registrations.

The authentic final R1 pre-test freeze, test-call ledger, predictions, official score and independent verification are in `evaluation/final_test/`. `ACCESS_SUMMARY.json` is explicitly a compact export of the original access receipts, with their original SHA256 values. The original prediction role accessed candidate text/representations only; references were opened by the separate scorer.

The selected R1 uses extra S28/V22 inputs, residual L2=0.03, the fixed CE baseline and the original temperature-one answer probability mean pooling. R0 has little effect, R2 adds no selected improvement, and the small neural/fusion branches do not displace R1. Their aggregate negative results remain in the report without shipping unused weights or all intermediate runs.

Research cost: 10 required CE fold rebuilds + 48 residual fits + 22 neural fits = 80; five new official validation scores; one frozen final R1 test. Fusion and publication add no model fitting. The report and original local archive retain fuller methodological detail. Earlier public history remains in the previous tags.
