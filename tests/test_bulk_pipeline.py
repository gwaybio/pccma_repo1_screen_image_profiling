"""The screen configuration and the step functions in profiling.bulk_pipeline."""
import pathlib

import pytest
import yaml
from profiling import bulk_pipeline as bp

from conftest import REPO


def test_screen_config_loads(config_path, cell_line):
    config = bp.load_config(config_path)
    assert config.cell_line == cell_line
    assert config.screen_name == f"{cell_line}_repo1_screen"


@pytest.mark.parametrize("change, message", [
    (lambda raw: raw.pop("cell_line"), "missing keys ['cell_line']"),
    (lambda raw: raw.update(plate_colums=24), "unknown keys ['plate_colums']"),
    (lambda raw: raw.update(raw_profile_dirs=[]), "raw_profile_dirs is empty"),
])
def test_config_rejects_missing_unknown_and_empty_keys(config_path, change, message):
    raw = yaml.safe_load(config_path.read_text())
    change(raw)
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        bp.config_from_dict(raw)


def test_raw_profile_dir_is_the_first_candidate_that_exists(config_path, tmp_path):
    raw = yaml.safe_load(config_path.read_text())
    present = tmp_path / "present"
    present.mkdir()
    config = bp.config_from_dict({**raw, "raw_profile_dirs": [str(tmp_path / "absent"), str(present), str(REPO)]})
    assert config.raw_profile_dir == present
    config = bp.config_from_dict({**raw, "raw_profile_dirs": [str(tmp_path / "absent"), str(tmp_path / "absent2")]})
    assert config.raw_profile_dir == tmp_path / "absent"   # none exists: the first, which the error then names


def test_correct_and_normalize_say_which_step_is_missing(config_path):
    config = bp.load_config(config_path)
    with pytest.raises(ValueError, match="`correct` needs the result of `pool_plates`"):
        bp.correct(config, None)
    with pytest.raises(ValueError, match="`normalize` needs the result of `correct`"):
        bp.normalize(config, None)
