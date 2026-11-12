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


def test_screen_configs_load():
    for path in SCREEN_CONFIGS:
        config = bp.load_config(path)
        assert config.screen_name == path.parent.parent.name
        assert config.screen_name.startswith(config.cell_line + "_")


def test_config_rejects_missing_and_unknown_keys():
    import yaml
    raw = yaml.safe_load(SCREEN_CONFIGS[0].read_text())
    for bad, message in [({k: v for k, v in raw.items() if k != "cell_line"}, "missing keys ['cell_line']"),
                         ({**raw, "plate_colums": 24}, "unknown keys ['plate_colums']"),
                         ({**raw, "raw_profile_dirs": []}, "raw_profile_dirs is empty")]:
        try:
            bp.config_from_dict(bad)
        except ValueError as error:
            assert message in str(error), (message, str(error))
        else:
            raise AssertionError(f"accepted a bad configuration: {message}")


def _write_platemaps(folder, layouts, rows="ABCDEFGHIJKLMNOP", n_columns=24):
    """Barcode file plus one platemap per layout; layouts maps a name to (compound columns, control columns)."""
    folder = pathlib.Path(folder)
    plates = []
    for i, (name, (compound_columns, control_columns)) in enumerate(layouts.items()):
        wells = [(r, c) for r in rows for c in range(1, n_columns + 1)]
        pd.DataFrame({
            "Well_Position": [f"{r}{c:02d}" for r, c in wells],
            "Batch_Id": [f"BRD-K{i}{c}{r}-001" if c in compound_columns else None for r, c in wells],
            "Solvent": ["DMSO" if c in control_columns else None for r, c in wells],
        }).to_csv(folder / f"{name}.csv", index=False)
        plates.append({"Plate_Barcode": f"P{i}", "File_Name": f"{name}.csv"})
    pd.DataFrame(plates).to_csv(folder / "barcode_platemap.csv", index=False)
    return folder / "barcode_platemap.csv"


def test_design_is_read_from_the_platemaps():
    import tempfile
    for path in SCREEN_CONFIGS:
        config = bp.load_config(path)
        platemaps = path.parent / config.platemap_dir
        design = b.design_from_platemaps(platemaps / config.barcode_platemap_file, platemaps)
        assert (design.n_rows, design.n_columns, design.control_columns, design.outer_columns) == (16, 24, (2, 24), (1, 2, 23, 24))
        assert design.interior_columns == tuple(range(3, 23))
    with tempfile.TemporaryDirectory() as tmp:   # a 96-well design: controls in columns 1 and 12, compounds in 2-11
        csv = _write_platemaps(tmp, {"a": (range(2, 12), (1, 12)), "b": (range(2, 12), (1, 12))}, rows="ABCDEFGH", n_columns=12)
        design = b.design_from_platemaps(csv, tmp)
        assert (design.n_rows, design.control_columns, design.interior_columns, design.outer_columns) == (8, (1, 12), tuple(range(2, 12)), (1, 12))


def test_design_rejects_platemaps_it_cannot_describe():
    import tempfile
    cases = {
        "disagree about which columns hold controls": {"a": (range(3, 23), (2, 24)), "b": (range(3, 23), (1, 24))},
        "hold both controls and compounds": {"a": (range(3, 23), (2, 24))},
        "must be consecutive": {"a": ([3, 4, 6, 7], (2, 24))},
    }
    for message, layouts in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            csv = _write_platemaps(tmp, layouts)
            if "both" in message:   # a control well among the compounds of column 3
                platemap = pd.read_csv(pathlib.Path(tmp) / "a.csv")
                platemap.loc[platemap.Well_Position == "A03", ["Batch_Id", "Solvent"]] = [None, "DMSO"]
                platemap.to_csv(pathlib.Path(tmp) / "a.csv", index=False)
            try:
                b.design_from_platemaps(csv, tmp)
            except ValueError as error:
                assert message in str(error), (message, str(error))
            else:
                raise AssertionError(f"accepted platemaps that {message}")


def test_groups_too_small_to_define_a_distribution_are_errors_not_skips():
    rng = np.random.default_rng(0)
    controls = ["A02"] + [f"{r}24" for r in "BCDEF"]                     # one control well in column 2, five in column 24
    compounds = [f"{r}{c:02d}" for r in "AB" for c in range(3, 23)]
    wells = controls + compounds
    df = pd.DataFrame({"Metadata_Plate": "P1", "Metadata_Well": wells,
                       "Metadata_Batch_Id": [None] * len(controls) + [f"BRD-K{i}-001" for i in range(len(compounds))],
                       "Metadata_Solvent": ["DMSO"] * len(controls) + [None] * len(compounds), "F": rng.standard_normal(len(wells))})
    for fn, kwargs in [(b.align_control_columns, {"control_query": b.NEGATIVE_CONTROL_QUERY}), (b.match_outer_columns, {"layout": ["l1"] * len(df)})]:
        try:
            fn(df, ["F"], **kwargs)
        except ValueError as error:
            assert "need at least" in str(error), str(error)
        else:
            raise AssertionError(f"{fn.__name__} skipped a group that is too small")


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
