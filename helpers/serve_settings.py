import os

# +++++++++++++++++++++++++++++++++++++++++++++ REST API request Setting +++++++++++++++++++++++++++++++++++++++++++++ #
mandatory_fields = [
    "file_path"
]
optional_fields = {
    "environment": "development",
    "confidence_threshold": 0.5,
    "iou_threshold": 0.5
}

load_image_from = "masked"

# TO-DO
headers = []

# ++++++++++++++++++++++++++++++++++++++++++++++ Model Related Settings ++++++++++++++++++++++++++++++++++++++++++++++ #
num_classes = 16
base_config = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"
model_level_confidence_threshold = 0.1
class_mapping = {
    0: "wheel", 
    1: "door", 
    2: "fender", 
    3: "bonnet", 
    4: "bumper", 
    5: "grill", 
    6: "light", 
    7: "rocker_panel", 
    8: "door_knob", 
    9: "wind_shield", 
    10: "roof", 
    11: "mirror", 
    12: "window_panel", 
    13: "top_fender", 
    14: "boot", 
    15: "number_plate"
}


# ++++++++++++++++++++++++++++++++++++++++++++ REST API Response Settings ++++++++++++++++++++++++++++++++++++++++++++ #
approx_coef = 0.003

# ++++++++++++++++++++++++++++++++++++++++++++++++ S3 Bucket Settings ++++++++++++++++++++++++++++++++++++++++++++++++ #
temp_folder_path = "temp_storage"

# V2 integration variables (dev)
DEV_BUCKET = 00
DEV_ACCESS_KEY = 00
DEV_SECRET_ACCESS_KEY = 00

# V2 integration variables (Staging)
STAGING_BUCKET = 00
STAGING_ACCESS_KEY = 00
STAGING_SECRET_ACCESS_KEY = 00

# V2 integration variables (production)
PRODUCTION_BUCKET = 00
PRODUCTION_ACCESS_KEY = 00
PRODUCTION_SECRET_ACCESS_KEY = 00


