"""CHP-134: what differs from the other screens in the shared bulk-processing steps (see utils/bulk_pipeline.py)."""
import pathlib

from bulk_pipeline import ScreenConfig

config = ScreenConfig(
    screen_name="CHP-134_repo1_screen",
    profiles_dir="~/mnt/bandicoot/PCCMA_data/CHP-134_repo1_profiles/profiles",
    raw_profile_dir=pathlib.Path("./data/merged_profiles"),     # from 0b.merge_profiles.ipynb
    raw_profile_suffix=".parquet",
    platemap_columns=("Plate Barcode", "File Name"),
    platemap_rename={"Well_Position": "Well"},
    annotate_join_on=("Metadata_Well", "Image_Metadata_Well"),
    nuclei_location_columns=("Metadata_Nuclei_Location_Center_X", "Metadata_Nuclei_Location_Center_Y"),
)
