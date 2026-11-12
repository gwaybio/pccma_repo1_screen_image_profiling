"""Checks of the bulk profiling utilities against pycytominer. Run:  uv run python utils/test_profiling_utils.py"""
import pathlib, sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bulk_pipeline as bp
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


REPO = pathlib.Path(__file__).resolve().parent.parent
SCREEN_CONFIGS = [REPO / f"{s}_repo1_screen/3.process_features/screen_config.yaml" for s in ("CHP-134", "SK-N-AS")]


def test_screen_configs_load_and_describe_a_384_well_layout():
    for path in SCREEN_CONFIGS:
        config = bp.load_config(path)
        assert config.screen_name == path.parent.parent.name
        assert (config.plate_rows, config.control_columns, config.outer_columns) == (16, (2, 24), (1, 2, 23, 24))
        assert config.interior_columns == tuple(range(3, 23))
        assert config.thresholds["correlation_threshold"] == 0.9


def test_config_rejects_missing_unknown_and_non_consecutive_keys():
    import yaml
    raw = yaml.safe_load(SCREEN_CONFIGS[0].read_text())
    for bad, message in [({k: v for k, v in raw.items() if k != "plate_rows"}, "missing keys ['plate_rows']"),
                         ({**raw, "plate_colums": 24}, "unknown keys ['plate_colums']"),
                         ({**raw, "interior_columns": [3, 4, 6]}, "consecutive"),
                         ({**raw, "thresholds": {"min_abs_coef_var": 0.001}}, "thresholds must be exactly")]:
        try:
            bp.config_from_dict(bad)
        except ValueError as error:
            assert message in str(error), (message, str(error))
        else:
            raise AssertionError(f"accepted a bad configuration: {message}")


def test_raw_profile_dir_is_the_first_candidate_that_exists():
    import tempfile
    import yaml
    raw = yaml.safe_load(SCREEN_CONFIGS[1].read_text())
    with tempfile.TemporaryDirectory() as tmp:
        present = pathlib.Path(tmp) / "present"
        present.mkdir()
        config = bp.config_from_dict({**raw, "raw_profile_dirs": [str(pathlib.Path(tmp) / "absent"), str(present), str(REPO)]})
        assert config.raw_profile_dir == present
        config = bp.config_from_dict({**raw, "raw_profile_dirs": [str(pathlib.Path(tmp) / "absent"), str(pathlib.Path(tmp) / "absent2")]})
        assert config.raw_profile_dir == pathlib.Path(tmp) / "absent"   # none exists: the first, which the error then names


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
