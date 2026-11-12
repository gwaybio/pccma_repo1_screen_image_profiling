"""Helpers for processing pooled bulk (well-level) profiles.

Order: correct, then normalize, then (bulk only) rank-transform, then select features.

    pool wells -> drop NaN/inf features -> align the two DMSO columns of every plate (E3a) ->
    match DMSO columns to the interior compound wells per layout (E3) -> variant filter -> provisional plate scale ->
    cell-count slope + shared position map with one amplitude per layout -> subtract effects ->
    normalize each plate on its corrected DMSO wells -> rank inverse normal transform -> pooled feature selection

Corrections (E3a, E3, the cell-count and position effects) are constants per well, so the same constants apply to a
well's median and to each single cell in it (see `single_cell_utils`). Normalization is a separate, later step.
pycytominer provides the inverse normal transform and feature selection; everything else is here.
"""

import pathlib
import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pycytominer import feature_select, normalize
from pycytominer.operations.transform import RobustMAD
from scipy.stats import median_abs_deviation

PLATE_COLUMN = "Metadata_Plate"
WELL_COLUMN = "Metadata_Well"
BATCH_ID_COLUMN = "Metadata_Batch_Id"


def pool_plate_profiles(
    paths: Iterable[pathlib.Path],
    plate_column: str = PLATE_COLUMN,
    replace_spaces_in_plate_names: bool = True,
) -> pd.DataFrame:
    """Stack the per-plate profile files (one parquet per plate) into one table, one row per well.

    Every file must have the same metadata columns. Feature columns may differ between plates: a feature that is
    missing from some plates is NaN there, is reported with a warning, and is dropped by `drop_nonfinite_features`.
    Spaces in plate names are replaced by underscores by default, so that "Assay Plate_1_3" and "Assay_Plate_1_3"
    (both spellings occur in CHP-134) are the same plate.
    """
    paths = sorted(pathlib.Path(path) for path in paths)
    if not paths:
        raise FileNotFoundError("no per-plate profile files to pool")

    frames = [pd.read_parquet(path) for path in paths]
    metadata = [c for c in frames[0].columns if c.startswith("Metadata_")]
    for path, frame in zip(paths, frames):
        if {c for c in frame.columns if c.startswith("Metadata_")} != set(metadata):
            raise ValueError(f"{path.name} has different metadata columns than {paths[0].name}")

    in_every_plate = set.intersection(*(set(frame.columns) for frame in frames))
    in_any_plate = set().union(*(set(frame.columns) for frame in frames))
    if in_any_plate != in_every_plate:
        warnings.warn(
            f"{len(in_any_plate - in_every_plate)} columns are missing from some plates and will be NaN there: "
            f"{sorted(in_any_plate - in_every_plate)[:5]}"
        )

    pooled = pd.concat(frames, ignore_index=True)
    if replace_spaces_in_plate_names:
        pooled[plate_column] = pooled[plate_column].str.replace(" ", "_", regex=False)
    return pooled


def check_plate_set(found: Iterable[str], expected: Iterable[str], allow_partial: bool = False) -> list[str]:
    """Stop (or warn) if plates that should be pooled are missing; return the missing plates, sorted.

    ``expected`` is every plate the screen's barcode-to-platemap file lists; ``found`` the plates that were pooled. Pooling
    a subset silently changes every whole-screen estimate (normalization, effects, feature selection), so a missing plate
    raises a ValueError unless ``allow_partial`` is set, in which case it only warns (a deliberate run on some plates).
    Plates found but not expected are not checked here (the pooled step reports plates without a platemap).
    """
    missing = sorted(set(expected) - set(found))
    if missing and not allow_partial:
        raise ValueError(
            f"{len(missing)} plates in the platemap file have no per-plate profile and would be left out of the pooled "
            f"processing: {missing}. Fix their aggregation, or set ALLOW_PARTIAL=1 to process the plates found."
        )
    if missing:
        warnings.warn(f"ALLOW_PARTIAL: processing without {len(missing)} plates: {missing}")
    return missing


def split_feature_columns(profiles: pd.DataFrame, features: Sequence[str]) -> list[str]:
    """The metadata columns: everything that is not one of ``features``, in their original order."""
    feature_set = set(features)
    return [column for column in profiles.columns if column not in feature_set]


def drop_nonfinite_features(profiles: pd.DataFrame, features: Sequence[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Drop every feature that has a NaN or an infinite value in any well (as the JUMP recipe does).

    Returns the table without those columns, the features that remain and the features that were dropped.
    """
    features = list(features)
    finite = np.isfinite(profiles[features].to_numpy(dtype=float)).all(axis=0)
    kept = [feature for feature, keep in zip(features, finite) if keep]
    dropped = [feature for feature, keep in zip(features, finite) if not keep]
    return profiles.drop(columns=dropped), kept, dropped


def control_mask(profiles: pd.DataFrame, control_query: str) -> np.ndarray:
    """Boolean array marking the control wells that match a pandas query (the same kind pycytominer takes as ``samples``)."""
    return profiles.index.isin(profiles.query(control_query).index)


def drop_wells(profiles: pd.DataFrame, query: str) -> tuple[pd.DataFrame, int]:
    """Remove the wells that match a pandas query; returns the remaining wells and how many were removed."""
    drop = control_mask(profiles, query)
    return profiles.loc[~drop].reset_index(drop=True), int(drop.sum())


# ---- position correction ----
# The plates share a plate-position pattern (a row effect and a column effect per feature) that is seen in the DMSO
# controls, which contain no compound, so it is a technical effect, not biology. It is estimated once for the screen
# from the replicate-averaged compound wells of all layouts (so it does not just fit noise from the ~30 controls per
# plate), and each layout gets its own amplitude of that shared map. DMSO wells (columns 2 and 24) receive the row
# part of the map. The cell-count effect is estimated alongside. Both are constants per well and feature, and are
# subtracted (in original units) after the variant filter and before the final normalization; see `estimate_effects`.


def split_well(wells: pd.Series) -> pd.DataFrame:
    """Row letter and column number of well names such as A01."""
    parts = wells.str.extract(r"^([A-Za-z]+)(\d+)$")
    return pd.DataFrame({"well_row": parts[0], "well_column": pd.to_numeric(parts[1])})


def extract_compound_id(batch_ids: pd.Series, pattern: str = r"^(BRD-[A-Za-z]\d+)") -> pd.Series:
    """The compound identifier inside a batch identifier (e.g. BRD-K75699339 from BRD-K75699339-001-12-9)."""
    return batch_ids.str.extract(pattern)[0]


def layouts_from_platemaps(
    barcode_platemap_csv: pathlib.Path, platemap_dir: pathlib.Path, plate_column: str = PLATE_COLUMN
) -> pd.Series:
    """The layout of every plate, from the screen's barcode-to-platemap file and the platemap files themselves.

    ``barcode_platemap_csv`` has one row per plate (plate barcode, platemap file name). Two plates have the same
    layout when their platemap files assign the same compound (batch ID) to the same wells, so layouts are read from
    the platemaps' contents, not inferred from the profiles. Plate names have spaces replaced by underscores, as in
    `pool_plate_profiles`. Returns a Series indexed by plate with labels ``layout_01``, ``layout_02``, ...
    """
    table = pd.read_csv(barcode_platemap_csv)
    plates = table.iloc[:, 0].astype(str).str.replace(" ", "_")
    files = table.iloc[:, 1].astype(str)
    signature = {}
    for name in files.unique():
        pm = pd.read_csv(pathlib.Path(platemap_dir) / name)
        well = next(c for c in pm.columns if "well" in c.lower())
        batch = next(c for c in pm.columns if "batch" in c.lower())
        pm = pm.dropna(subset=[batch])
        signature[name] = tuple(sorted(zip(pm[well].astype(str), pm[batch].astype(str))))
    labels = {}
    out = {}
    for plate, name in zip(plates, files):
        sig = signature[name]
        if sig not in labels:
            labels[sig] = f"layout_{len(labels) + 1:02d}"
        out[plate] = labels[sig]
    return pd.Series(out, name="layout")


def plate_control_stats(
    profiles: pd.DataFrame,
    features: Sequence[str],
    is_control: np.ndarray,
    plate_column: str = PLATE_COLUMN,
    min_controls: int = 3,
) -> dict[str, pd.DataFrame]:
    """Median, MAD and |MAD / median| of every feature over each plate's control wells (tables of plates x features).

    The MAD is the raw median absolute deviation (no 1.4826 factor). ``abs_coef_var`` is 0 wherever the median is 0
    or the ratio is not finite, so a feature with a zero median never counts as variable.
    """
    features = list(features)
    controls = profiles.loc[is_control]
    counts = controls.groupby(plate_column).size().reindex(profiles[plate_column].unique(), fill_value=0)
    too_few = counts[counts < min_controls]
    if len(too_few):
        raise ValueError(f"plates with fewer than {min_controls} control wells: {too_few.to_dict()}")

    medians, mads = {}, {}
    for plate, plate_controls in controls.groupby(plate_column, sort=False):
        values = plate_controls[features].to_numpy(dtype=float)
        medians[plate] = np.nanmedian(values, axis=0)
        mads[plate] = median_abs_deviation(values, axis=0, nan_policy="omit")
    median = pd.DataFrame.from_dict(medians, orient="index", columns=features)
    mad = pd.DataFrame.from_dict(mads, orient="index", columns=features)

    with np.errstate(divide="ignore", invalid="ignore"):
        abs_coef_var = (mad / median).abs()
    abs_coef_var = abs_coef_var.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return {"median": median, "mad": mad, "abs_coef_var": abs_coef_var}


def variant_features(stats: dict[str, pd.DataFrame], min_abs_coef_var: float = 1e-3) -> tuple[list[str], list[str]]:
    """Features that vary in the control wells of every plate: MAD is not 0 and |MAD / median| is above ``min_abs_coef_var``.

    A feature that fails on even one plate is dropped. Returns (kept features, dropped features), in column order.
    """
    passes = (stats["mad"] != 0) & (stats["abs_coef_var"] > min_abs_coef_var)
    keep = passes.all(axis=0)
    return keep.index[keep].tolist(), keep.index[~keep].tolist()


def inverse_normal_transform(profiles: pd.DataFrame, features: Sequence[str], random_state: int = 0) -> pd.DataFrame:
    """Rank-based inverse normal transform of every feature over all wells, with pycytominer's quantile method.

    pycytominer builds the quantile landmarks from at most 10,000 randomly chosen rows (scikit-learn's default), which
    matters for screens with more wells than that; seeding numpy's global generator makes that choice reproducible.
    """
    features = list(features)
    np.random.seed(random_state)
    return normalize(
        profiles=profiles,
        features=features,
        meta_features=split_feature_columns(profiles, features),
        samples="all",
        method="inverse_normal",
    )


def select_features(
    profiles: pd.DataFrame,
    features: Sequence[str],
    correlation_threshold: float = 0.9,
    freq_cut: float = 0.05,
    unique_cut: float = 0.01,
) -> pd.DataFrame:
    """Pooled feature selection with pycytominer, in the order the JUMP recipe uses.

    (1) frequency filter (the recipe's "variance" filter), (2) correlation filter, (3) pycytominer's default blocklist
    (the same list the recipe ships), (4) drop any feature with a missing value.
    """
    return feature_select(
        profiles=profiles,
        features=list(features),
        operation=["frequency_threshold", "correlation_threshold", "blocklist", "drop_na_columns"],
        freq_cut=freq_cut,
        unique_cut=unique_cut,
        corr_threshold=correlation_threshold,
        na_cutoff=0,
    )


# ---------------------------------------------------------------------------------------------------------------
# Corrections and normalization
#
# A *correction* removes one specific, named artifact. Every correction is a constant added to a well's values (or a
# monotone map of them), so it can be applied identically to a well's median and to each single cell of that well,
# in the profiles' original units. *Normalization* (centering and scaling each plate on its DMSO wells) is not a
# correction; it always comes after the corrections, with statistics taken from the corrected DMSO wells.
# ---------------------------------------------------------------------------------------------------------------

PLATE_ROWS = 16


def _quantile_match_1d(values: np.ndarray, baseline_sorted: np.ndarray) -> np.ndarray:
    """Map ``values`` onto the empirical distribution ``baseline_sorted``, keeping their own rank order."""
    n = len(values)
    order = np.argsort(values)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n)
    percentile = (ranks + 0.5) / n
    baseline_percentile = (np.arange(len(baseline_sorted)) + 0.5) / len(baseline_sorted)
    return np.interp(percentile, baseline_percentile, baseline_sorted)


def align_control_columns(
    profiles: pd.DataFrame,
    features: Sequence[str],
    control_query: str,
    plate_column: str = PLATE_COLUMN,
    well_column: str = WELL_COLUMN,
    columns: Sequence[int] = (2, 24),
    min_wells: int = 5,
) -> pd.DataFrame:
    """E3a. Within every plate, make the control (DMSO) wells of the two control columns follow the same distribution.

    Both control columns hold the same substance, so within a plate they should be interchangeable, yet each plate's
    column 2 differs from its column 24 by a stable, plate-specific offset (it flips sign between plates, so no shared
    correction can remove it). For each plate and feature the control wells of each column are mapped, rank-preserving,
    onto the pooled distribution of that plate's control wells in both columns. A plate with fewer than ``min_wells``
    control wells in either column is left unchanged.
    """
    features = list(features)
    col = split_well(profiles[well_column])["well_column"].to_numpy()
    is_control = control_mask(profiles, control_query)
    X = profiles[features].to_numpy(dtype=float).copy()
    for _, idx in pd.Series(np.arange(len(profiles))).groupby(profiles[plate_column].to_numpy()).indices.items():
        groups = [idx[is_control[idx] & (col[idx] == c)] for c in columns]
        if any(len(g) < min_wells for g in groups):
            continue
        pooled = np.vstack([X[g] for g in groups])
        pooled_sorted = np.sort(pooled, axis=0)
        for g in groups:
            for j in range(X.shape[1]):
                X[g, j] = _quantile_match_1d(X[g, j], pooled_sorted[:, j])
    out = profiles.copy()
    out[features] = X
    return out


def match_outer_columns(
    profiles: pd.DataFrame,
    features: Sequence[str],
    plate_column: str = PLATE_COLUMN,
    well_column: str = WELL_COLUMN,
    batch_id_column: str = BATCH_ID_COLUMN,
    layout: Sequence | None = None,
    interior_columns: Iterable[int] = range(3, 23),
    outer_columns: Iterable[int] = (1, 2, 23, 24),
    min_interior_wells: int = 20,
    min_outer_wells: int = 5,
) -> pd.DataFrame:
    """E3. Map every outer plate column (the control columns) onto the distribution of the interior compound wells.

    For each layout (``layout``: one label per row, see `layouts_from_platemaps`) and each feature, the interior
    columns' compound wells define a baseline distribution. Every well in an outer column of that group is then
    replaced by the baseline value at the same percentile among that column's wells (rank-preserving), controls
    included. A group with fewer than ``min_interior_wells`` interior compound wells, or an outer column with fewer than
    ``min_outer_wells`` wells, is left unchanged.
    """
    features = list(features)
    if layout is None:
        raise ValueError("layout labels are required (see layouts_from_platemaps)")
    compound_mask = extract_compound_id(profiles[batch_id_column]).notna().to_numpy()
    group = np.asarray(layout)
    col = split_well(profiles[well_column])["well_column"].to_numpy()
    X = profiles[features].to_numpy(dtype=float).copy()
    interior = np.isin(col, list(interior_columns))
    for g in pd.unique(group):
        in_group = group == g
        base_rows = in_group & compound_mask & interior
        if base_rows.sum() < min_interior_wells:
            continue
        baseline_sorted = np.sort(X[base_rows], axis=0)
        for c in outer_columns:
            rows = in_group & (col == c)
            if rows.sum() < min_outer_wells:
                continue
            for j in range(X.shape[1]):
                X[rows, j] = _quantile_match_1d(X[rows, j], baseline_sorted[:, j])
    out = profiles.copy()
    out[features] = X
    return out


def _backfit_median(P: np.ndarray, r: np.ndarray, c: np.ndarray, n_rows: int, n_cols: int, n_iter: int):
    """Additive row + column level vectors per feature by backfitting, each level the MEDIAN of its residuals."""
    resid = P - np.median(P, axis=0)
    row_level = np.zeros((n_rows, P.shape[1]))
    col_level = np.zeros((n_cols, P.shape[1]))
    for _ in range(n_iter):
        res = resid - col_level[c]
        for k in range(n_rows):
            m = r == k
            if m.any():
                row_level[k] = np.median(res[m], axis=0)
        res = resid - row_level[r]
        for k in range(n_cols):
            m = c == k
            if m.any():
                col_level[k] = np.median(res[m], axis=0)
    return row_level, col_level


def _amplitude(profiles: np.ndarray, map_values: np.ndarray) -> float:
    """The scale ``a`` at which ``map_values`` best matches ``profiles`` (1 = the map at its estimated strength).

    Per feature, the least-squares slope of the profiles on the map over the rows; the amplitude is the median of those
    slopes weighted by each feature's map energy (sum of squared map values). A plain least squares over all features and
    wells would be dominated by a few heavy-tailed features.
    """
    energy = (map_values * map_values).sum(axis=0)
    ok = energy > 0
    if not ok.any():
        return 0.0
    slope = (profiles[:, ok] * map_values[:, ok]).sum(axis=0) / energy[ok]
    w = energy[ok]
    order = np.argsort(slope)
    cum = np.cumsum(w[order])
    return float(slope[order][np.searchsorted(cum, cum[-1] / 2)])


def _shrink_amplitude(amplitudes: dict, target) -> float:
    """Shrink one layout's amplitude toward the precision-weighted average of the other layouts' (empirical Bayes).

    ``tau2 = max(0, var(others) - mean(se^2))`` is the layout-to-layout variance left after sampling error; the result is
    ``mean + w * (own - mean)`` with ``w = tau2 / (tau2 + own_se^2)``. The target does not enter the mean or tau2.
    """
    others = [v for k, v in amplitudes.items() if k != target]
    if len(others) < 2:
        return amplitudes[target][0]
    amps = np.array([v[0] for v in others])
    ses = np.array([v[1] for v in others])
    tau2 = max(0.0, amps.var(ddof=1) - np.mean(ses ** 2))
    mean = np.average(amps, weights=1.0 / (ses ** 2 + tau2 + 1e-12))
    own, own_se = amplitudes[target]
    w = tau2 / (tau2 + own_se ** 2) if tau2 + own_se ** 2 > 0 else 0.0
    return float(mean + w * (own - mean))


def estimate_effects(
    profiles: pd.DataFrame,
    features: Sequence[str],
    log_cell_count: np.ndarray,
    control_query: str,
    plate_column: str = PLATE_COLUMN,
    well_column: str = WELL_COLUMN,
    batch_id_column: str = BATCH_ID_COLUMN,
    layout: Sequence | None = None,
    compound_columns: tuple[int, int] = (3, 22),
    control_columns: Sequence[int] = (2, 24),
    winsor: tuple[float, float] = (0.05, 0.95),
    n_backfit_iter: int = 6,
    n_boot: int = 200,
    seed: int = 0,
    n_rows: int = PLATE_ROWS,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.DataFrame]:
    """The cell-count (density) effect and the plate-position effect on every well, in the units of ``profiles``.

    ``profiles`` must already be centred and scaled per plate on its controls, so plates share units. As in the CFReT
    screen, each effect is one shared model plus, for position, one amplitude per plate layout:

    Density: for each feature, one slope against the (winsorized) log cell count, estimated from all control wells'
    variation inside each plate and column (control wells are identical, so count-linked variation there is artifact).
    Applied to compound wells through their plate-centred, winsorized log count, and to control wells through their
    count centred within plate and column (column-level differences are not touched).

    Position: ONE additive row + column map per feature, fitted with medians (values are heavy-tailed) on the
    replicate-averaged compound wells of all layouts after the density effect; and one *amplitude* per layout, the
    least-squares scale at which the map fits that layout's own compound wells. The amplitude's standard error is
    bootstrapped over the layout's compound-position profiles, and the amplitude is shrunk toward the other layouts'
    (empirical Bayes). The position effect of a compound well is ``amplitude[layout] * map(row, column)``; a control
    well in columns 2/24 receives the row part of the map only.

    Wells without a cell count receive no density effect. Returns (density effects, position effects, layout label per
    well, per-layout amplitude table); the first two are (n wells x n features).
    """
    features = list(features)
    n = len(profiles)
    X = profiles[features].to_numpy(dtype=float)
    parts = split_well(profiles[well_column])
    row = parts["well_row"].map({ch: i for i, ch in enumerate("ABCDEFGHIJKLMNOP")}).to_numpy()
    col = parts["well_column"].to_numpy()
    plate = profiles[plate_column].astype(str).to_numpy()
    compound = extract_compound_id(profiles[batch_id_column])
    if layout is None:
        raise ValueError("layout labels are required (see layouts_from_platemaps)")
    layout = np.asarray(layout)
    is_control = control_mask(profiles, control_query)
    lo_c, hi_c = compound_columns
    is_comp = compound.notna().to_numpy() & (col >= lo_c) & (col <= hi_c)
    is_ctrl_col = is_control & np.isin(col, list(control_columns))
    logn = np.asarray(log_cell_count, dtype=float)
    has_n = np.isfinite(logn)
    n_cols = hi_c - lo_c + 1

    # density covariates: compound wells plate-centred and winsorized, control wells centred within plate x column
    z_comp = np.zeros(n)
    sel = is_comp & has_n
    zc = logn[sel] - pd.Series(logn[sel]).groupby(plate[sel]).transform("mean").to_numpy()
    q = pd.Series(zc).groupby(plate[sel]).quantile(list(winsor)).unstack()
    z_comp[sel] = np.clip(zc, pd.Series(plate[sel]).map(q[winsor[0]]).to_numpy(), pd.Series(plate[sel]).map(q[winsor[1]]).to_numpy())
    z_ctrl = np.zeros(n)
    selc = is_ctrl_col & has_n
    key = [plate[selc], col[selc]]
    zd = logn[selc] - pd.Series(logn[selc]).groupby(key).transform("mean").to_numpy()
    qd = pd.Series(zd).groupby(plate[selc]).quantile(list(winsor)).unstack()
    z_ctrl[selc] = np.clip(zd, pd.Series(plate[selc]).map(qd[winsor[0]]).to_numpy(), pd.Series(plate[selc]).map(qd[winsor[1]]).to_numpy())

    # one density slope per feature
    if selc.sum() >= 3:
        Yd = X[selc] - pd.DataFrame(X[selc]).groupby(key).transform("mean").to_numpy()
        zc_d = logn[selc] - pd.Series(logn[selc]).groupby(key).transform("mean").to_numpy()
        beta = (zc_d[:, None] * Yd).sum(axis=0) / (zc_d ** 2).sum()
    else:
        beta = np.zeros(X.shape[1])
    density = np.zeros_like(X)
    density[is_comp & has_n] = np.outer(z_comp[is_comp & has_n], beta)
    density[is_ctrl_col & has_n] = np.outer(z_ctrl[is_ctrl_col & has_n], beta)

    def compound_profiles(wells):
        """Replicate-averaged, plate-centred, density-removed compound-position profiles (+ their row/column)."""
        Y = X[wells] - np.outer(z_comp[wells], beta)
        Y = Y - pd.DataFrame(Y).groupby(plate[wells]).transform("mean").to_numpy()
        gid = pd.DataFrame({"c": compound.to_numpy()[wells], "r": row[wells], "k": col[wells]}).groupby(
            ["c", "r", "k"], sort=False).ngroup().to_numpy()
        m = gid.max() + 1
        P = np.zeros((m, Y.shape[1]))
        np.add.at(P, gid, Y)
        P /= np.bincount(gid, minlength=m)[:, None]
        first = pd.DataFrame({"g": gid, "r": row[wells], "k": col[wells]}).drop_duplicates("g").sort_values("g")
        return P, first["r"].to_numpy().astype(int), (first["k"].to_numpy() - lo_c).astype(int)

    # one shared position map from all compound wells
    all_wells = np.where(is_comp & has_n)[0]
    P, r, c = compound_profiles(all_wells)
    row_level, col_level = _backfit_median(P, r, c, n_rows, n_cols, n_backfit_iter)
    position_map = np.zeros_like(X)
    tg = np.where(is_comp)[0]
    position_map[tg] = row_level[row[tg].astype(int)] + col_level[(col[tg] - lo_c).astype(int)]

    # one amplitude per layout
    rng = np.random.default_rng(seed)
    raw = {}
    for lay in pd.unique(layout):
        wells = np.where((layout == lay) & is_comp & has_n)[0]
        if len(wells) == 0:
            continue
        P, r, c = compound_profiles(wells)
        T = row_level[r] + col_level[c]
        if not (T * T).sum() > 0:
            continue
        boots = [_amplitude(P[i], T[i]) for i in (rng.integers(0, len(P), len(P)) for _ in range(n_boot))]
        raw[lay] = (_amplitude(P, T), float(np.std(boots)), len(P))
    position = np.zeros_like(X)
    rows_out = []
    for lay in pd.unique(layout):
        in_lay = (layout == lay) & is_comp
        if lay in raw:
            a = _shrink_amplitude({k: (v[0], v[1]) for k, v in raw.items()}, lay)
            position[in_lay] = a * position_map[in_lay]
            # control wells: the row part of the map only (no compound sits in their columns, so the column level of
            # columns 2/24 is not estimated; E3 already set their column-level distribution)
            ctrl = (layout == lay) & is_ctrl_col
            position[ctrl] = a * row_level[row[ctrl].astype(int)]
            rows_out.append({"layout": lay, "n_profiles": raw[lay][2], "raw_amplitude": raw[lay][0],
                             "se": raw[lay][1], "amplitude": a})
        else:
            rows_out.append({"layout": lay, "n_profiles": 0, "raw_amplitude": np.nan, "se": np.nan, "amplitude": 0.0})
    return density, position, pd.Series(layout, index=profiles.index, name="layout"), pd.DataFrame(rows_out)


def plate_scale(
    profiles: pd.DataFrame,
    features: Sequence[str],
    control_query: str,
    plate_column: str = PLATE_COLUMN,
) -> pd.DataFrame:
    """Each plate's centre and scale per feature, from its control wells, with pycytominer's ``RobustMAD``.

    ``RobustMAD`` is what ``normalize(method="mad_robustize")`` fits: the median and the MAD (scaled by pycytominer's
    own 1.4826 convention). Returned long: one row per (plate, statistic in {"median", "scale"}), one column per feature,
    so the same constants can be applied to wells and to single cells (`apply_plate_scale`).
    """
    features = list(features)
    rows = []
    for plate, frame in profiles.groupby(plate_column, sort=False):
        controls = frame.loc[control_mask(frame, control_query), features]
        if len(controls) < 3:
            raise ValueError(f"plate {plate} has fewer than 3 control wells")
        fitted = RobustMAD().fit(controls)
        rows.append(pd.concat([pd.Series({plate_column: plate, "statistic": "median"}), fitted.median]))
        rows.append(pd.concat([pd.Series({plate_column: plate, "statistic": "scale"}), fitted.mad]))
    out = pd.DataFrame(rows).reset_index(drop=True)
    out[features] = out[features].astype(float)
    return out


def apply_plate_scale(
    values: np.ndarray, plates: np.ndarray, scale_table: pd.DataFrame, features: Sequence[str],
    plate_column: str = PLATE_COLUMN, epsilon: float = 1e-18,
) -> np.ndarray:
    """``RobustMAD.transform`` with each row's own plate constants: (values - median) / (scale + epsilon)."""
    features = list(features)
    med = scale_table[scale_table["statistic"] == "median"].set_index(plate_column)[features]
    sca = scale_table[scale_table["statistic"] == "scale"].set_index(plate_column)[features]
    plates = np.asarray(plates)
    out = np.empty(values.shape, dtype=np.float64)
    for p in pd.unique(plates):
        r = plates == p
        scaler = RobustMAD(epsilon=epsilon)
        scaler.median, scaler.mad = med.loc[p], sca.loc[p]
        out[r] = scaler.transform(pd.DataFrame(values[r], columns=features)).to_numpy()
    return out
