"""The corrections and helpers in profiling.bulk_utils."""
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from profiling import bulk_utils as b
from pycytominer import normalize

from conftest import CELL_LINES, REPO


def test_plate_scale_equals_pycytominer_mad_robustize():
    rng = np.random.default_rng(0)
    rows = []
    for plate in ["P1", "P2", "P3"]:
        for well in range(40):
            rows.append({"Metadata_Plate": plate, "Metadata_Well": f"A{well + 1:02d}", "Metadata_Batch_Id": None if well < 12 else f"BRD-K{well}-001",
                         "Metadata_Solvent": "DMSO" if well < 12 else "other",
                         **{f"Cells_F{j}": rng.standard_normal() * (1 + 0.1 * j) + (3 if plate == "P2" else 0) for j in range(8)}})
    df = pd.DataFrame(rows)
    features = [f"Cells_F{j}" for j in range(8)]
    query = 'Metadata_Batch_Id.isna() and Metadata_Solvent == "DMSO"'
    ours = b.apply_plate_scale(df[features].to_numpy(), df.Metadata_Plate.to_numpy(), b.plate_scale(df, features, query), features)
    reference = pd.concat([
        normalize(profiles=g.reset_index(drop=True), features=features, meta_features=[c for c in df.columns if c not in features],
                  samples=query, method="mad_robustize")
        for _, g in df.groupby("Metadata_Plate", sort=False)
    ])
    assert np.allclose(ours, reference[features].to_numpy(), atol=1e-12)


class TestCheckPlateSet:
    def test_complete_or_extra_plates_pass(self):
        assert b.check_plate_set(["P1", "P2"], ["P1", "P2"]) == []
        assert b.check_plate_set(["P1", "P2", "P3"], ["P1", "P2"]) == []   # extra plates are the pooled step's concern

    def test_a_missing_plate_stops_the_run(self):
        with pytest.raises(ValueError, match=r"P2.*P3.*ALLOW_PARTIAL"):
            b.check_plate_set(["P1"], ["P1", "P2", "P3"])

    def test_allow_partial_only_warns(self):
        with pytest.warns(UserWarning, match="ALLOW_PARTIAL"):
            assert b.check_plate_set(["P1"], ["P1", "P2"], allow_partial=True) == ["P2"]


class TestDesignFromPlatemaps:
    def test_both_screens(self, cell_line):
        platemaps = REPO / f"{cell_line}_repo1_screen/0.download_data/metadata"
        design = b.design_from_platemaps(platemaps / "barcode_platemap.csv", platemaps)
        assert (design.n_rows, design.n_columns, design.control_columns, design.outer_columns) == (16, 24, (2, 24), (1, 2, 23, 24))
        assert design.interior_columns == tuple(range(3, 23))

    def test_a_96_well_design(self, write_platemaps):
        csv = write_platemaps({"a": (range(2, 12), (1, 12)), "b": (range(2, 12), (1, 12))}, rows="ABCDEFGH", n_columns=12)
        design = b.design_from_platemaps(csv, csv.parent)
        assert (design.n_rows, design.control_columns, design.interior_columns, design.outer_columns) == (8, (1, 12), tuple(range(2, 12)), (1, 12))

    def test_platemaps_that_disagree_about_the_control_columns(self, write_platemaps):
        csv = write_platemaps({"a": (range(3, 23), (2, 24)), "b": (range(3, 23), (1, 24))})
        with pytest.raises(ValueError, match="disagree about which columns hold controls"):
            b.design_from_platemaps(csv, csv.parent)

    def test_a_control_column_that_also_holds_compounds(self, write_platemaps):
        csv = write_platemaps({"a": (range(3, 23), (2, 24))})
        platemap = pd.read_csv(csv.parent / "a.csv")
        platemap.loc[platemap.Well_Position == "A03", ["Batch_Id", "Solvent"]] = [None, "DMSO"]   # a control among the compounds
        platemap.to_csv(csv.parent / "a.csv", index=False)
        with pytest.raises(ValueError, match="hold both controls and compounds"):
            b.design_from_platemaps(csv, csv.parent)

    def test_compound_columns_that_are_not_consecutive(self, write_platemaps):
        csv = write_platemaps({"a": ([3, 4, 6, 7], (2, 24))})
        with pytest.raises(ValueError, match="must be consecutive"):
            b.design_from_platemaps(csv, csv.parent)


def test_groups_too_small_to_define_a_distribution_are_errors_not_skips():
    rng = np.random.default_rng(0)
    controls = ["A02"] + [f"{r}24" for r in "BCDEF"]                     # one control well in column 2, five in column 24
    compounds = [f"{r}{c:02d}" for r in "AB" for c in range(3, 23)]
    wells = controls + compounds
    df = pd.DataFrame({"Metadata_Plate": "P1", "Metadata_Well": wells,
                       "Metadata_Batch_Id": [None] * len(controls) + [f"BRD-K{i}-001" for i in range(len(compounds))],
                       "Metadata_Solvent": ["DMSO"] * len(controls) + [None] * len(compounds), "F": rng.standard_normal(len(wells))})
    with pytest.raises(ValueError, match="need at least"):
        b.align_control_columns(df, ["F"], b.NEGATIVE_CONTROL_QUERY)
    with pytest.raises(ValueError, match="need at least"):
        b.match_outer_columns(df, ["F"], layout=["l1"] * len(df))
