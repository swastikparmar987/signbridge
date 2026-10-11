import numpy as np


def build_xyz_velocity_features(sequence: np.ndarray) -> np.ndarray:
    """Build normalized XYZ position plus temporal velocity features.

    Args:
        sequence: Normalized landmarks with shape (T, 42, 3).

    Returns:
        Float32 features with shape (T, 252).
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[1:] != (42, 3):
        raise ValueError(f"Expected sequence shape (T, 42, 3), got {sequence.shape}")

    positions = sequence.reshape(sequence.shape[0], 126)
    velocity = np.zeros_like(positions, dtype=np.float32)
    if positions.shape[0] > 1:
        velocity[1:] = positions[1:] - positions[:-1]
    return np.concatenate((positions, velocity), axis=1).astype(np.float32)


def build_interhand_displacement(raw_sequence: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute palm-scale-normalized 3D inter-hand displacement (T, 3).

    Args:
        raw_sequence: Raw landmarks with shape (T, 42, 3).

    Returns:
        Float32 displacement vectors with shape (T, 3).
    """
    raw_sequence = np.asarray(raw_sequence, dtype=np.float32)
    num_frames = raw_sequence.shape[0]
    displacement = np.zeros((num_frames, 3), dtype=np.float32)

    for t in range(num_frames):
        h0 = raw_sequence[t, :21]
        h1 = raw_sequence[t, 21:42]

        h0_active = not np.allclose(h0, 0)
        h1_active = not np.allclose(h1, 0)

        if h0_active and h1_active:
            wrist0 = h0[0]
            wrist1 = h1[0]
            s0 = np.linalg.norm(h0[9] - wrist0)
            s1 = np.linalg.norm(h1[9] - wrist1)
            mean_scale = (s0 + s1) / 2.0
            if mean_scale > eps:
                displacement[t] = (wrist1 - wrist0) / mean_scale

    return displacement


def build_palm_normals(raw_sequence: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute 3D unit palm normal vectors for Hand 0 and Hand 1 (T, 6).

    Args:
        raw_sequence: Raw landmarks with shape (T, 42, 3).

    Returns:
        Float32 unit palm normals with shape (T, 6).
    """
    raw_sequence = np.asarray(raw_sequence, dtype=np.float32)
    num_frames = raw_sequence.shape[0]
    normals = np.zeros((num_frames, 6), dtype=np.float32)

    for t in range(num_frames):
        h0 = raw_sequence[t, :21]
        h1 = raw_sequence[t, 21:42]

        # Hand 0 normal
        if not np.allclose(h0, 0):
            v1_0 = h0[5] - h0[0]
            v2_0 = h0[17] - h0[0]
            n_raw0 = np.cross(v1_0, v2_0)
            norm0 = np.linalg.norm(n_raw0)
            if norm0 > eps:
                normals[t, :3] = n_raw0 / norm0

        # Hand 1 normal
        if not np.allclose(h1, 0):
            v1_1 = h1[5] - h1[0]
            v2_1 = h1[17] - h1[0]
            n_raw1 = np.cross(v1_1, v2_1)
            norm1 = np.linalg.norm(n_raw1)
            if norm1 > eps:
                normals[t, 3:6] = n_raw1 / norm1

    return normals


def build_exp14_features(raw_sequence: np.ndarray, normalized_sequence: np.ndarray, feature_mode: str = "c") -> np.ndarray:
    """
    Build EXP14 feature set (255D, 258D, or 261D).

    Args:
        raw_sequence: Raw landmarks (T, 42, 3).
        normalized_sequence: Wrist-normalized landmarks (T, 42, 3).
        feature_mode: "a" (255D), "b" (258D), or "c" (261D).

    Returns:
        Float32 feature tensor (T, D).
    """
    base_252 = build_xyz_velocity_features(normalized_sequence) # (T, 252)

    mode = feature_mode.lower()
    if mode == "a":
        disp = build_interhand_displacement(raw_sequence) # (T, 3)
        return np.concatenate((base_252, disp), axis=1).astype(np.float32)
    elif mode == "b":
        normals = build_palm_normals(raw_sequence) # (T, 6)
        return np.concatenate((base_252, normals), axis=1).astype(np.float32)
    elif mode == "c":
        disp = build_interhand_displacement(raw_sequence) # (T, 3)
        normals = build_palm_normals(raw_sequence) # (T, 6)
        return np.concatenate((base_252, disp, normals), axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown EXP14 feature mode: {feature_mode}. Expected 'a', 'b', or 'c'.")


def build_body_aware_features(
    raw_sequence: np.ndarray,
    pose_sequence: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Compute compact 31D body/spatial relative features per frame (T, 31).

    Feature Layout per frame (31D total):
    - 3D: Hand 0 wrist to Torso center (normalized by shoulder width)
    - 3D: Hand 1 wrist to Torso center (normalized by shoulder width)
    - 3D: Hand 0 wrist to Nose/Head center (normalized by shoulder width)
    - 3D: Hand 1 wrist to Nose/Head center (normalized by shoulder width)
    - 3D: Hand 0 wrist to Left Shoulder (normalized by shoulder width)
    - 3D: Hand 1 wrist to Right Shoulder (normalized by shoulder width)
    - 3D: Hand 0 wrist to Left Ear (normalized by shoulder width)
    - 3D: Hand 1 wrist to Right Ear (normalized by shoulder width)
    - 3D: Shoulder vector (Right Shoulder - Left Shoulder) / shoulder_width
    - 3D: Head-Torso vector (Nose - Torso center) / shoulder_width
    - 1D: Pose presence flag (1.0 if pose valid, 0.0 otherwise)
    """
    raw_sequence = np.asarray(raw_sequence, dtype=np.float32)
    pose_sequence = np.asarray(pose_sequence, dtype=np.float32)
    num_frames = raw_sequence.shape[0]

    body_features = np.zeros((num_frames, 31), dtype=np.float32)

    for t in range(num_frames):
        h0 = raw_sequence[t, :21]
        h1 = raw_sequence[t, 21:42]
        pose = pose_sequence[t]

        h0_active = not np.allclose(h0, 0)
        h1_active = not np.allclose(h1, 0)
        pose_active = not np.allclose(pose, 0)

        if not pose_active:
            continue

        p_nose = pose[0]
        p_ear_L = pose[7]
        p_ear_R = pose[8]
        p_sh_L = pose[11]
        p_sh_R = pose[12]
        p_hip_L = pose[23]
        p_hip_R = pose[24]

        p_torso = (p_sh_L + p_sh_R + p_hip_L + p_hip_R) / 4.0

        sh_dist = np.linalg.norm(p_sh_R - p_sh_L)
        scale = sh_dist if sh_dist > eps else 1.0

        if h0_active:
            w0 = h0[0]
            w0_torso = (w0 - p_torso) / scale
            w0_head = (w0 - p_nose) / scale
            w0_sh_L = (w0 - p_sh_L) / scale
            w0_ear_L = (w0 - p_ear_L) / scale
        else:
            w0_torso = np.zeros(3, dtype=np.float32)
            w0_head = np.zeros(3, dtype=np.float32)
            w0_sh_L = np.zeros(3, dtype=np.float32)
            w0_ear_L = np.zeros(3, dtype=np.float32)

        if h1_active:
            w1 = h1[0]
            w1_torso = (w1 - p_torso) / scale
            w1_head = (w1 - p_nose) / scale
            w1_sh_R = (w1 - p_sh_R) / scale
            w1_ear_R = (w1 - p_ear_R) / scale
        else:
            w1_torso = np.zeros(3, dtype=np.float32)
            w1_head = np.zeros(3, dtype=np.float32)
            w1_sh_R = np.zeros(3, dtype=np.float32)
            w1_ear_R = np.zeros(3, dtype=np.float32)

        sh_vec = (p_sh_R - p_sh_L) / scale
        head_torso_vec = (p_nose - p_torso) / scale
        pose_valid = np.array([1.0], dtype=np.float32)

        feat_vec = np.concatenate([
            w0_torso, w1_torso,
            w0_head, w1_head,
            w0_sh_L, w1_sh_R,
            w0_ear_L, w1_ear_R,
            sh_vec, head_torso_vec,
            pose_valid
        ], axis=0).astype(np.float32)

        body_features[t] = feat_vec

    return body_features


def build_exp15_features(
    raw_sequence: np.ndarray,
    normalized_sequence: np.ndarray,
    pose_sequence: np.ndarray
) -> np.ndarray:
    """
    Build EXP15 feature set (292D: 261D EXP14-C + 31D Body Features).
    """
    exp14c_261 = build_exp14_features(raw_sequence, normalized_sequence, feature_mode="c")
    body_31 = build_body_aware_features(raw_sequence, pose_sequence)
    return np.concatenate((exp14c_261, body_31), axis=1).astype(np.float32)