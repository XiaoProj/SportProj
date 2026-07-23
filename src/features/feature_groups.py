"""Feature group definitions used by training and ablation experiments."""

JOINT_ANGLE_FEATURES = [
    "left_knee_flexion_angle",
    "right_knee_flexion_angle",
    "left_hip_flexion_angle",
    "right_hip_flexion_angle",
    "left_ankle_angle",
    "right_ankle_angle",
]

SYMMETRY_FEATURES = [
    "left_right_knee_angle_difference",
    "left_right_ankle_angle_difference",
]

STABILITY_FEATURES = [
    "left_knee_valgus_variation",
    "right_knee_valgus_variation",
    "landing_phase_left_knee_angle_fluctuation",
    "landing_phase_right_knee_angle_fluctuation",
]

TEMPORAL_FEATURES = [
    "takeoff_to_landing_duration",
    "lowest_center_of_mass_timing",
]

LOAD_AND_SUBJECTIVE_FEATURES = [
    "training_load_srpe_duration",
    "fatigue_score",
    "sleep_score",
    "muscle_soreness_score",
]

POSE_FEATURES = (
    JOINT_ANGLE_FEATURES
    + SYMMETRY_FEATURES
    + STABILITY_FEATURES
    + TEMPORAL_FEATURES
)

ALL_FEATURES = POSE_FEATURES + LOAD_AND_SUBJECTIVE_FEATURES

FEATURE_GROUPS = {
    "joint_angles": JOINT_ANGLE_FEATURES,
    "symmetry": SYMMETRY_FEATURES,
    "stability": STABILITY_FEATURES,
    "temporal": TEMPORAL_FEATURES,
    "load_subjective": LOAD_AND_SUBJECTIVE_FEATURES,
    "pose_all": POSE_FEATURES,
    "all": ALL_FEATURES,
}


def available_feature_groups() -> dict[str, list[str]]:
    """Return a copy of the named ablation feature groups."""

    return {name: list(features) for name, features in FEATURE_GROUPS.items()}


def filter_existing_features(columns: list[str], requested: list[str]) -> list[str]:
    """Keep requested features that are actually present in a feature table."""

    column_set = set(columns)
    return [feature for feature in requested if feature in column_set]
