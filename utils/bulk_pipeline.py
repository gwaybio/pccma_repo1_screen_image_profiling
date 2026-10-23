"""Bulk-profile processing steps shared by every screen.

A screen is described once, by a `ScreenConfig` (each screen's `screen_config.py`). The notebooks only load that
configuration and call one function from here, so a new screen needs a configuration, not new processing code:

    annotate_plates(config)             2.annotate                  per plate: platemap and QC flags attached, poor-quality cells dropped
    aggregate_plates(config)            3.aggregate_bulk_profiles   per plate: the median of the cells in each well
    count_cells_per_well(config)        3a.well_cell_counts         whole screen: cells per well (the cell-count covariate)
    run_pooled_bulk_processing(config)  3b.pooled_bulk_processing   whole screen: corrections, normalization, selection (see bulk_utils)

The per-plate steps read PLATE_ID (process one plate per run, which keeps memory down) and OVERWRITE from the environment.
"""

import csv
import gc
import json
import os
import pathlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pycytominer import aggregate, annotate
from pycytominer.cyto_utils import output
from pycytominer.cyto_utils.features import infer_cp_features

import bulk_utils

PLATE_COLUMN = "Metadata_Plate"
WELL_COLUMN = "Metadata_Well"
PLATE_METADATA_PREFIX = "Metadata_"
# Empty wells (no compound and no solvent) are dropped when pooling; wells with no compound that received DMSO are the negative controls
UNTREATED_QUERY = "Metadata_Batch_Id.isna() and Metadata_Solvent.isna()"
NEG_CONTROL_QUERY = 'Metadata_Batch_Id.isna() and Metadata_Solvent == "DMSO"'


@dataclass(frozen=True)
class ScreenConfig:
    """Everything that differs between screens. Paths are relative to the screen's `3.process_features` folder."""

    screen_name: str                      # e.g. "CHP-134_repo1_screen"; prefixes every saved table
    profiles_dir: str                     # the screen's profiles folder on bandicoot (PROFILES_DIR overrides): 3a and 3b read and write here
    raw_profile_dir: pathlib.Path         # one profile per plate before QC (the converted or merged profiles)
    raw_profile_suffix: str               # what follows the plate id in those file names: "_converted.parquet" or ".parquet"
    platemap_columns: tuple               # barcode_platemap.csv columns: (plate barcode, platemap file name)
    platemap_rename: dict                 # renames applied to every platemap before annotation
    annotate_join_on: tuple               # (platemap column, profile column) joined by pycytominer.annotate
    nuclei_location_columns: tuple        # the two nuclei-centre columns as they are named in this screen's annotated profiles
    qc_dir: pathlib.Path = pathlib.Path("./data/qc_results")
    annotated_dir: pathlib.Path = pathlib.Path("./data/annotated_profiles")     # written by 2.annotate
    bulk_dir: pathlib.Path = pathlib.Path("./data/bulk_profiles")               # per-plate aggregates, written by 3.aggregate_bulk_profiles
    platemap_dir: pathlib.Path = pathlib.Path("../0.download_data/metadata")
    annotated_subdir: str = "annotated_profiles"      # under profiles_dir: where 3a looks for the annotated profiles
    annotated_glob: str = "*_annotated.parquet"
    per_plate_glob: str = "*_bulk_aggregated.parquet"  # the per-plate aggregates that 3b pools
    thresholds: dict = field(default_factory=lambda: {"min_abs_coef_var": 1e-3, "correlation_threshold": 0.9})


def plate_names(config: ScreenConfig) -> list[str]:
    """The screen's plates, from the file names in the raw profile folder."""
    if not config.raw_profile_dir.exists():
        raise FileNotFoundError(f"The raw profile folder {config.raw_profile_dir} does not exist.")
    suffix = config.raw_profile_suffix
    return sorted(f.name.removesuffix(suffix) for f in config.raw_profile_dir.glob(f"*{suffix}"))


def _plates_to_run(config: ScreenConfig) -> tuple[list[str], bool]:
    """All plates, or only $PLATE_ID; and whether $OVERWRITE is set (1/true/yes)."""
    plates = plate_names(config)
    only = os.environ.get("PLATE_ID")
    if only:
        if only not in plates:
            raise ValueError(f"Unknown plate_id in PLATE_ID env var: {only}")
        plates = [only]
    overwrite = os.environ.get("OVERWRITE", "").strip().lower() in ("1", "true", "yes")
    return plates, overwrite


def _log_timing(path: pathlib.Path, header: list[str], row: list[str]) -> None:
    """Append one row to a per-step timing log so per-plate performance can be compared across runs."""
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def _platemap_path(config: ScreenConfig, barcode_platemap: pd.DataFrame, plate_id: str):
    plate_col, file_col = config.platemap_columns
    rows = barcode_platemap.loc[barcode_platemap[plate_col] == plate_id, file_col]
    return pathlib.Path(config.platemap_dir) / rows.values[0] if len(rows) else None


def _with_location_columns(df: pd.DataFrame, names: tuple) -> pd.DataFrame:
    """Give the nuclei-centre columns this screen's naming: a few plates' conversion added (or lacks) a ``Metadata_`` prefix."""
    rename = {}
    for name in names:
        other = name[len(PLATE_METADATA_PREFIX):] if name.startswith(PLATE_METADATA_PREFIX) else PLATE_METADATA_PREFIX + name
        if name not in df.columns and other in df.columns:
            rename[other] = name
    return df.rename(columns=rename)


# ---- per plate: 2.annotate ----
def annotate_plate(config: ScreenConfig, plate_id: str, barcode_platemap: pd.DataFrame, overwrite: bool = False) -> float | None:
    """Attach the platemap, drop the poor-quality segmentations (the QC flags) and write one annotated profile for the plate.

    Poor-quality cells are dropped before annotating, which keeps the memory use down (a plate's raw profile is tens of GB).
    Returns the seconds taken, or None if the plate was skipped.
    """
    qc_path = config.qc_dir / f"{plate_id}_qc_annotations.parquet"
    if not qc_path.exists():
        print(f"Skipping {plate_id}: no QC annotations yet (run 1.single_cell_qc.ipynb for this plate first)")
        return None
    platemap_path = _platemap_path(config, barcode_platemap, plate_id)
    if platemap_path is None:
        print(f"Skipping {plate_id}: no platemap found in barcode_platemap.csv")
        return None
    output_file = config.annotated_dir / f"{plate_id}_annotated.parquet"
    if output_file.exists() and not overwrite:
        print(f"Skipping {plate_id}: already annotated (found {output_file})")
        return None

    print(f"Annotating {plate_id}")
    start = time.time()
    profile = pd.read_parquet(config.raw_profile_dir / f"{plate_id}{config.raw_profile_suffix}")
    qc = pd.read_parquet(qc_path)
    platemap = pd.read_csv(platemap_path).rename(columns=config.platemap_rename)
    profile = _with_location_columns(profile, config.nuclei_location_columns)
    qc = _with_location_columns(qc, config.nuclei_location_columns)

    # the QC file's flag columns mark poor-quality segmentations; a cell is joined to its flags by well, site and nucleus centre
    cqc_cols = [c for c in qc.columns if c.startswith("Metadata_cqc_")]
    join_keys = ["Image_Metadata_Well", "Image_Metadata_Site", *config.nuclei_location_columns]
    assert not qc.duplicated(subset=join_keys).any(), f"{plate_id}: QC file has duplicate rows for the same cell"
    assert not profile.duplicated(subset=join_keys).any(), f"{plate_id}: profile file has duplicate rows for the same cell"
    flags = profile[join_keys].merge(qc[join_keys + cqc_cols], on=join_keys, how="left")
    assert flags[cqc_cols].isna().sum().sum() == 0, f"{plate_id}: some cells have no matching QC annotation"
    poor = flags[cqc_cols].any(axis=1)
    print(f"  Dropping {poor.sum()} / {len(profile)} poor-quality segmentations")
    profile = profile.loc[~poor].reset_index(drop=True)
    del qc, flags, poor

    annotated = annotate(profiles=profile, platemap=platemap, join_on=list(config.annotate_join_on))
    annotated.rename(columns={"Image_Metadata_Site": "Metadata_Site"}, inplace=True)   # pycytominer's own name for the site column
    assert "Metadata_Site" in annotated.columns, f"{plate_id}: Metadata_Site column missing after rename()"
    config.annotated_dir.mkdir(parents=True, exist_ok=True)
    output(df=annotated, output_filename=str(output_file), output_type="parquet")
    seconds = time.time() - start
    del profile, platemap, annotated
    gc.collect()
    print(f"Annotation completed for {plate_id}! (annotate={seconds:.1f}s)")
    return seconds


def annotate_plates(config: ScreenConfig) -> None:
    """2.annotate: annotate every plate (or only $PLATE_ID), one after the other."""
    plates, overwrite = _plates_to_run(config)
    barcode_platemap = pd.read_csv(pathlib.Path(config.platemap_dir) / "barcode_platemap.csv")
    barcode_platemap[config.platemap_columns[0]] = barcode_platemap[config.platemap_columns[0]].str.replace(" ", "_", regex=False)
    config.annotated_dir.mkdir(parents=True, exist_ok=True)
    for plate_id in plates:
        seconds = annotate_plate(config, plate_id, barcode_platemap, overwrite)
        if seconds is not None:
            _log_timing(config.annotated_dir / "timing_log.csv", ["plate_id", "annotate_seconds", "timestamp"],
                        [plate_id, f"{seconds:.1f}", datetime.now(timezone.utc).isoformat(timespec="seconds")])


# ---- per plate: 3.aggregate_bulk_profiles ----
def aggregate_plate(config: ScreenConfig, plate_id: str, overwrite: bool = False) -> float | None:
    """The median of the plate's cells in every well, with the platemap metadata that is constant within a well kept."""
    annotated_path = config.annotated_dir / f"{plate_id}_annotated.parquet"
    if not annotated_path.exists():
        print(f"Skipping {plate_id}: not yet annotated (run 2.annotate.ipynb for this plate first)")
        return None
    output_file = config.bulk_dir / f"{plate_id}_bulk_aggregated.parquet"
    if output_file.exists() and not overwrite:
        print(f"Skipping {plate_id}: already processed (found {output_file})")
        return None

    print(f"Now performing pycytominer pipeline for {plate_id}")
    start = time.time()
    annotated = pd.read_parquet(annotated_path)
    # aggregate() keeps only the `strata` columns and the aggregated features. The platemap-derived metadata columns that are
    # constant within a well are put in `strata` so they survive; the raw profile's parquet schema (names only, no data)
    # tells which Metadata_ columns were added by the platemap join.
    per_cell_cols = set(pq.ParquetFile(config.raw_profile_dir / f"{plate_id}{config.raw_profile_suffix}").schema_arrow.names)
    candidates = [c for c in annotated.columns
                  if c.startswith(PLATE_METADATA_PREFIX) and c not in (PLATE_COLUMN, WELL_COLUMN) and c not in per_cell_cols]
    strata = [PLATE_COLUMN, WELL_COLUMN] + [c for c in candidates
                                             if annotated.groupby(WELL_COLUMN)[c].nunique(dropna=False).le(1).all()]
    aggregated = aggregate(population_df=annotated, operation="median", strata=strata)
    config.bulk_dir.mkdir(parents=True, exist_ok=True)
    output(df=aggregated, output_filename=str(output_file), output_type="parquet")
    seconds = time.time() - start
    del annotated, aggregated
    gc.collect()
    print(f"Bulk aggregation completed for {plate_id}! (aggregate={seconds:.1f}s)")
    return seconds


def aggregate_plates(config: ScreenConfig) -> None:
    """3.aggregate_bulk_profiles: aggregate every plate (or only $PLATE_ID)."""
    plates, overwrite = _plates_to_run(config)
    config.bulk_dir.mkdir(parents=True, exist_ok=True)
    for plate_id in plates:
        seconds = aggregate_plate(config, plate_id, overwrite)
        if seconds is not None:
            _log_timing(config.bulk_dir / "timing_log.csv", ["plate_id", "aggregate_seconds", "timestamp"],
                        [plate_id, f"{seconds:.1f}", datetime.now(timezone.utc).isoformat(timespec="seconds")])


# ---- whole screen: 3a.well_cell_counts ----
def count_cells_per_well(config: ScreenConfig) -> pathlib.Path:
    """Count the quality-controlled cells of every well from the annotated profiles (only the plate and well columns are read).

    Plates already in the output are skipped ($OVERWRITE=1 recounts everything); the file is rewritten after every plate.
    ANNOTATED_DIR overrides where the annotated profiles are read from.
    """
    profiles_dir = pathlib.Path(os.environ.get("PROFILES_DIR", config.profiles_dir)).expanduser()
    annotated_dir = pathlib.Path(os.environ.get("ANNOTATED_DIR", str(profiles_dir / config.annotated_subdir))).expanduser()
    output_file = profiles_dir / "bulk_profiles" / f"{config.screen_name}_well_cell_counts.csv"
    overwrite = os.environ.get("OVERWRITE", "").strip().lower() in ("1", "true", "yes")
    files = sorted(annotated_dir.glob(config.annotated_glob))
    if not files:
        raise FileNotFoundError(f"no annotated profiles matching {config.annotated_glob} in {annotated_dir}")
    print(f"{len(files)} annotated profiles in {annotated_dir}")

    columns = [PLATE_COLUMN, WELL_COLUMN, "n_cells"]
    existing = pd.read_csv(output_file) if output_file.exists() and not overwrite else pd.DataFrame(columns=columns)
    done = set(existing[PLATE_COLUMN].astype(str).str.replace(" ", "_"))
    parts = [existing]
    for f in files:
        t0 = time.time()
        table = pq.read_table(f, columns=[PLATE_COLUMN, WELL_COLUMN]).to_pandas()
        plate = str(table[PLATE_COLUMN].iloc[0])
        if plate.replace(" ", "_") in done:
            print(f"Skipping {plate}: already counted")
            continue
        counts = table.groupby([PLATE_COLUMN, WELL_COLUMN]).size().rename("n_cells").reset_index()
        parts.append(counts)
        print(f"{plate}: {int(counts['n_cells'].sum()):,} cells, {len(counts)} wells ({time.time() - t0:.0f}s)")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(parts, ignore_index=True).to_csv(output_file, index=False)   # resumable after every plate

    out = pd.concat(parts, ignore_index=True).drop_duplicates([PLATE_COLUMN, WELL_COLUMN])
    out.to_csv(output_file, index=False)
    print(f"saved {output_file}: {out[PLATE_COLUMN].nunique()} plates, {len(out)} wells")
    return output_file


# ---- whole screen: 3b.pooled_bulk_processing ----
def run_pooled_bulk_processing(config: ScreenConfig) -> None:
    """Pool the plates and run the corrections, normalization, rank transform and feature selection (see bulk_utils).

    Reads the per-plate aggregates and the cell counts from, and writes the bulk profiles and the correction tables to,
    the screen's profiles folder (PROFILES_DIR overrides it; BULK_PROFILES_DIR overrides where the aggregates are read from).
    """
    screen_name = config.screen_name
    profiles_dir = pathlib.Path(os.environ.get("PROFILES_DIR", config.profiles_dir)).expanduser()
    bulk_dir = profiles_dir / "bulk_profiles"
    cell_count_file = bulk_dir / f"{screen_name}_well_cell_counts.csv"                                        # from 3a
    output_file = bulk_dir / f"{screen_name}_pooled_bulk_corrected_feature_selected.parquet"                  # corrected set (main)
    base_output_file = bulk_dir / f"{screen_name}_pooled_bulk_base_feature_selected.parquet"                  # without steps 7-8
    tables_dir = bulk_dir                                                                                      # correction tables
    min_abs_coef_var = config.thresholds["min_abs_coef_var"]
    correlation_threshold = config.thresholds["correlation_threshold"]
    barcode_platemap_file = pathlib.Path(config.platemap_dir) / "barcode_platemap.csv"

    aggregate_dir = pathlib.Path(os.environ["BULK_PROFILES_DIR"]).expanduser() if os.environ.get("BULK_PROFILES_DIR") else bulk_dir
    plate_files = sorted(aggregate_dir.glob(config.per_plate_glob))
    if not plate_files:
        raise FileNotFoundError(f"no per-plate profile files matching {config.per_plate_glob} in {aggregate_dir}")
    print(f"Reading {len(plate_files)} per-plate profiles from {plate_files[0].parent}")

    def report(step, profiles, features):
        n_controls = int(bulk_utils.control_mask(profiles, NEG_CONTROL_QUERY).sum())
        print(f"{step}: {len(profiles)} wells ({n_controls} controls) on {profiles[PLATE_COLUMN].nunique()} plates, {len(features)} features")

    def with_values(frame, features, values):
        """The frame's metadata columns plus ``values`` (n wells x n features) as the feature columns."""
        meta = [c for c in frame.columns if c not in set(infer_cp_features(frame))]
        return pd.concat([frame[meta].reset_index(drop=True), pd.DataFrame(values, columns=features)], axis=1)

    def plate_key(plates):
        """Plate names with spaces replaced by underscores (the pooled table does the same to the per-plate files' names)."""
        return plates.astype(str).str.replace(" ", "_")

    # step 1: pool the plates, drop empty wells, read each plate's layout from the platemap files
    pooled = bulk_utils.pool_plate_profiles(plate_files, PLATE_COLUMN)
    pooled, n_untreated = bulk_utils.drop_wells(pooled, UNTREATED_QUERY)
    print(f"dropped {n_untreated} empty wells (no compound, no solvent)")
    assert not pooled.duplicated([PLATE_COLUMN, WELL_COLUMN]).any(), "a plate/well appears more than once"
    features = infer_cp_features(pooled)
    report("1 pooled", pooled, features)

    layout_by_plate = bulk_utils.layouts_from_platemaps(barcode_platemap_file, pathlib.Path(config.platemap_dir))
    missing_plates = sorted(set(pooled[PLATE_COLUMN]) - set(layout_by_plate.index))
    assert not missing_plates, f"plates without a platemap file in {barcode_platemap_file}: {missing_plates}"
    layout = pooled[PLATE_COLUMN].map(layout_by_plate).to_numpy()
    print(f"{pd.Series(layout).nunique()} layouts; plates per layout: {sorted(layout_by_plate.value_counts().tolist(), reverse=True)}")

    # step 2: drop features with missing or infinite values
    pooled, features, nonfinite = bulk_utils.drop_nonfinite_features(pooled, features)
    print(f"dropped {len(nonfinite)} features with NaN/inf values; examples: {nonfinite[:5]}")
    report("2 finite features", pooled, features)
    original = pooled[features].to_numpy(dtype=float).copy()   # values before any correction (for the per-well shift)

    # steps 3-4: line up the two DMSO columns (E3a), then match the DMSO columns to the interior compound wells (E3)
    pooled = bulk_utils.align_control_columns(pooled, features, NEG_CONTROL_QUERY, PLATE_COLUMN, WELL_COLUMN)
    report("3 control columns aligned", pooled, features)
    pooled = bulk_utils.match_outer_columns(pooled, features, PLATE_COLUMN, WELL_COLUMN, layout=layout)
    shift = pooled[features].to_numpy(dtype=float) - original   # what steps 3-4 added to each well's median
    report("4 matched to interior", pooled, features)
    print(f"wells changed by steps 3-4: {int((np.abs(shift).sum(axis=1) > 0).sum())}")

    # step 5: variant filter
    is_control = bulk_utils.control_mask(pooled, NEG_CONTROL_QUERY)
    stats = bulk_utils.plate_control_stats(pooled, features, is_control, PLATE_COLUMN)
    features_kept, features_dropped = bulk_utils.variant_features(stats, min_abs_coef_var)
    print(f"dropped {len(features_dropped)} of {len(features)} features that do not vary in the DMSO wells of every plate")
    keep = [features.index(f) for f in features_kept]
    pooled = pooled.drop(columns=features_dropped)
    original, shift = original[:, keep], shift[:, keep]
    features = features_kept
    report("5 variant features", pooled, features)

    # step 6: provisional scaling (to estimate the effects on a common scale)
    matched = pooled[features].to_numpy(dtype=float)
    provisional_scale = bulk_utils.plate_scale(pooled, features, NEG_CONTROL_QUERY, PLATE_COLUMN)
    provisional = bulk_utils.apply_plate_scale(matched, pooled[PLATE_COLUMN].to_numpy(), provisional_scale, features, PLATE_COLUMN)
    assert np.isfinite(provisional).all(), "non-finite values after provisional scaling"

    # step 7: the cell-count effect and the position map (one amplitude per layout)
    counts = pd.read_csv(cell_count_file)
    counts[PLATE_COLUMN] = plate_key(counts[PLATE_COLUMN])
    counts = counts.drop_duplicates([PLATE_COLUMN, WELL_COLUMN]).set_index([PLATE_COLUMN, WELL_COLUMN])["n_cells"]
    idx = pd.MultiIndex.from_arrays([plate_key(pooled[PLATE_COLUMN]), pooled[WELL_COLUMN]])
    n_cells = counts.reindex(idx).to_numpy(dtype=float)
    print(f"wells with a cell count: {int(np.isfinite(n_cells).sum())} of {len(n_cells)}")
    assert np.isfinite(n_cells).mean() > 0.99, "cell counts missing for more than 1% of wells"
    log_n = np.log(np.where(n_cells > 0, n_cells, np.nan))
    density, position, layout, amplitudes = bulk_utils.estimate_effects(
        with_values(pooled, features, provisional), features, log_n, NEG_CONTROL_QUERY, PLATE_COLUMN, WELL_COLUMN, layout=layout,
    )
    print(amplitudes.round(3).to_string(index=False))
    print(f"layouts: {layout.nunique()}; wells with a density effect: {int((np.abs(density).sum(axis=1) > 0).sum())}; "
          f"with a position effect: {int((np.abs(position).sum(axis=1) > 0).sum())}")

    # step 8: subtract the effects, in original units
    scale_rows = (provisional_scale[provisional_scale["statistic"] == "scale"].set_index(PLATE_COLUMN)[features]
                  .reindex(pooled[PLATE_COLUMN].to_numpy()).to_numpy())
    effects = (density + position) * scale_rows
    corrected = matched - effects
    print(f"median |effect| / plate scale: {np.median(np.abs(density + position)):.4f}")

    # step 9: normalize each plate on its (corrected) DMSO wells
    final_scale = bulk_utils.plate_scale(with_values(pooled, features, corrected), features, NEG_CONTROL_QUERY, PLATE_COLUMN)
    base_scale = provisional_scale
    plates = pooled[PLATE_COLUMN].to_numpy()
    normalized = bulk_utils.apply_plate_scale(corrected, plates, final_scale, features, PLATE_COLUMN)
    base_normalized = provisional
    assert np.isfinite(normalized).all(), "non-finite values after normalization"

    # steps 10-11: inverse normal transform, feature selection
    transformed = bulk_utils.inverse_normal_transform(with_values(pooled, features, normalized), features)
    base_transformed = bulk_utils.inverse_normal_transform(with_values(pooled, features, base_normalized), features)
    report("10 inverse normal", transformed, features)
    selected = bulk_utils.select_features(transformed, features, correlation_threshold)
    selected_features = infer_cp_features(selected)
    base_selected = bulk_utils.select_features(base_transformed, features, correlation_threshold)
    base_selected_features = infer_cp_features(base_selected)
    report("11 feature selected (corrected)", selected, selected_features)
    report("11 feature selected (base)", base_selected, base_selected_features)
    selected.to_parquet(output_file, index=False)
    base_selected.to_parquet(base_output_file, index=False)
    print(f"saved {output_file}\nsaved {base_output_file}")

    # the per-well shifts and effects and the per-plate scales, so the same constants can be applied to every cell
    keys = pooled[[PLATE_COLUMN, WELL_COLUMN]].reset_index(drop=True)
    pd.concat([keys, pd.DataFrame(shift.astype(np.float32), columns=features)], axis=1).to_parquet(
        tables_dir / f"{screen_name}_correction_well_shift.parquet", index=False)
    pd.concat([keys, pd.DataFrame(effects.astype(np.float32), columns=features)], axis=1).to_parquet(
        tables_dir / f"{screen_name}_correction_well_effects.parquet", index=False)
    final_scale.to_parquet(tables_dir / f"{screen_name}_correction_plate_scale_corrected.parquet", index=False)
    base_scale.to_parquet(tables_dir / f"{screen_name}_correction_plate_scale_base.parquet", index=False)
    (tables_dir / f"{screen_name}_correction_features.json").write_text(json.dumps({
        "variant_features": features,
        "selected_bulk_corrected": selected_features,
        "selected_bulk_base": base_selected_features,
        "plate_layout": layout_by_plate.to_dict(),
        "control_query": NEG_CONTROL_QUERY,
    }, indent=1))
    amplitudes.to_csv(tables_dir / f"{screen_name}_correction_layout_amplitudes.csv", index=False)
    print(f"saved correction tables to {tables_dir}")
