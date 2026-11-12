"""Checks of the bulk profiling utilities against pycytominer. Run:  uv run python utils/test_profiling_utils.py"""
import pathlib, sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bulk_utils as b
from pycytominer import normalize


def test_plate_scale_equals_pycytominer_mad_robustize():
    rng = np.random.default_rng(0)
    rows = []
    for plate in ["P1", "P2", "P3"]:
        for well in range(40):
            rows.append({"Metadata_Plate": plate, "Metadata_Well": f"A{well + 1:02d}", "Metadata_Batch_Id": None if well < 12 else f"BRD-K{well}-001",
                         "Metadata_Solvent": "DMSO" if well < 12 else "other",
                         **{f"Cells_F{j}": rng.standard_normal() * (1 + 0.1 * j) + (3 if plate == "P2" else 0) for j in range(8)}})
    df = pd.DataFrame(rows)
    feats = [f"Cells_F{j}" for j in range(8)]
    q = 'Metadata_Batch_Id.isna() and Metadata_Solvent == "DMSO"'
    mine = b.apply_plate_scale(df[feats].to_numpy(), df.Metadata_Plate.to_numpy(), b.plate_scale(df, feats, q), feats)
    ref = pd.concat([normalize(profiles=g.reset_index(drop=True), features=feats, meta_features=[c for c in df.columns if c not in feats],
                               samples=q, method="mad_robustize") for _, g in df.groupby("Metadata_Plate", sort=False)])
    assert np.allclose(mine, ref[feats].to_numpy(), atol=1e-12), "plate scaling differs from pycytominer"


def test_check_plate_set_stops_on_missing_plate_unless_allowed():
    import warnings
    assert b.check_plate_set(["P1", "P2"], ["P1", "P2"]) == []
    assert b.check_plate_set(["P1", "P2", "P3"], ["P1", "P2"]) == []   # extra plates are the pooled step's concern
    try:
        b.check_plate_set(["P1"], ["P1", "P2", "P3"])
    except ValueError as error:
        assert "P2" in str(error) and "P3" in str(error) and "ALLOW_PARTIAL" in str(error)
    else:
        raise AssertionError("a missing plate did not stop the run")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert b.check_plate_set(["P1"], ["P1", "P2"], allow_partial=True) == ["P2"]
    assert any("ALLOW_PARTIAL" in str(w.message) for w in caught)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
