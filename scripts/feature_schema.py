"""Shared column definitions for handcrafted-feature generation and modeling."""

FACIAL_FEATURE_COLUMNS = [
    "eye_open_mean",
    "eye_open_std",
    "eye_open_min",
    "eye_open_range",
    "eye_motion_mean",
    "blink_count",
    "blink_closed_ratio",
    "mouth_open_mean",
    "mouth_open_std",
    "mouth_open_max",
    "mouth_open_range",
    "mouth_open_motion",
    "mouth_corner_mean",
    "mouth_corner_std",
    "mouth_corner_range",
    "mouth_corner_motion",
    "mouth_corner_asymmetry_mean",
    "brow_eye_mean",
    "brow_eye_std",
    "brow_eye_range",
    "brow_eye_motion",
]

METADATA_COLUMNS = [
    "trial_id",
    "participant_id",
    "drink_name",
    "drink_label",
]

QUESTIONNAIRE_COLUMNS = [
    "delicious",
    "like",
    "sweet",
    "bitter",
    "stimulus",
    "surprise",
    "easy_to_drink",
]

FEATURE_OUTPUT_COLUMNS = (
    METADATA_COLUMNS + FACIAL_FEATURE_COLUMNS + QUESTIONNAIRE_COLUMNS
)
