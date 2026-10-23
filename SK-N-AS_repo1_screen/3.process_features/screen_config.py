"""SK-N-AS: what differs from the other screens in the shared bulk-processing steps (see utils/bulk_pipeline.py)."""
import pathlib

from bulk_pipeline import ScreenConfig

# Converted single-cell profiles (one per plate, from 0.convert_cytotable.ipynb). On Alpine (HPC) they live on the
# PetaLibrary "koala" mount, falling back to the data/converted_profiles folder in the checkout on scratch; locally they
# live on the external drive, since each plate's profile is tens of GB.
if pathlib.Path("/scratch/alpine").exists():
    _koala = pathlib.Path("/pl/active/koala/ALSF_screen_data/SK-N-AS_repo1_profiles/converted_profiles")
    _raw_profile_dir = _koala if _koala.exists() else pathlib.Path("data/converted_profiles")
else:
    _raw_profile_dir = pathlib.Path("/media/18tbdrive2/SK-N-AS_repo1_profiles/converted_profiles")

config = ScreenConfig(
    screen_name="SK-N-AS_repo1_screen",
    profiles_dir="~/mnt/bandicoot/PCCMA_data/SK-N-AS_repo1_profiles",
    raw_profile_dir=_raw_profile_dir,
    raw_profile_suffix="_converted.parquet",
    platemap_columns=("Plate_Barcode", "File_Name"),
    platemap_rename={},
    annotate_join_on=("Metadata_Well_Position", "Image_Metadata_Well"),
    nuclei_location_columns=("Nuclei_Location_Center_X", "Nuclei_Location_Center_Y"),
)
