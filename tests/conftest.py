"""Shared fixtures: the two screens' configuration files and a factory for small synthetic platemaps."""
import pathlib

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
# cell line -> the token used in file names (the notebooks and configuration files)
CELL_LINES = {"CHP-134": "CHP134", "SK-N-AS": "SKNAS"}


@pytest.fixture(params=sorted(CELL_LINES))
def cell_line(request):
    return request.param


@pytest.fixture
def screen_folder(cell_line):
    return REPO / f"{cell_line}_repo1_screen" / "3.process_features"


@pytest.fixture
def config_path(cell_line, screen_folder):
    return screen_folder / f"screen_config_{CELL_LINES[cell_line]}.yaml"


@pytest.fixture
def write_platemaps(tmp_path):
    """Write a barcode-to-platemap file and one platemap per layout; returns the barcode file.

    ``layouts`` maps a layout name to (columns with compounds, columns with controls).
    """

    def write(layouts, rows="ABCDEFGHIJKLMNOP", n_columns=24):
        plates = []
        for i, (name, (compound_columns, control_columns)) in enumerate(layouts.items()):
            wells = [(r, c) for r in rows for c in range(1, n_columns + 1)]
            pd.DataFrame({
                "Well_Position": [f"{r}{c:02d}" for r, c in wells],
                "Batch_Id": [f"BRD-K{i}{c}{r}-001" if c in compound_columns else None for r, c in wells],
                "Solvent": ["DMSO" if c in control_columns else None for r, c in wells],
            }).to_csv(tmp_path / f"{name}.csv", index=False)
            plates.append({"Plate_Barcode": f"P{i}", "File_Name": f"{name}.csv"})
        pd.DataFrame(plates).to_csv(tmp_path / "barcode_platemap.csv", index=False)
        return tmp_path / "barcode_platemap.csv"

    return write
