import numpy as np

def normalize_hand(hand: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize coordinates for a single hand of shape (21, 3).
    
    Steps:
    1. Check for missing hand (all zeros) or invalid inputs.
    2. Subtract wrist coordinate (Landmark 0) to make hand wrist-relative.
    3. Scale normalize by the distance between wrist (Landmark 0) and middle MCP (Landmark 9).
    
    Args:
        hand: np.ndarray of shape (21, 3)
        eps: Small epsilon value to prevent zero division
        
    Returns:
        np.ndarray of shape (21, 3), normalized float32.
    """
    hand = np.nan_to_num(hand, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32).copy()

    # Missing hand: keep all zeros
    if np.allclose(hand, 0):
        return hand

    # Wrist (Landmark 0) relative
    wrist = hand[0].copy()
    normalized = hand - wrist

    # Scale by distance from wrist (0) to middle finger MCP (9)
    middle_mcp = hand[9]
    scale = np.linalg.norm(middle_mcp - wrist)

    if scale < eps:
        return normalized

    normalized = normalized / scale
    return normalized.astype(np.float32)


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Normalize landmark sequence.
    
    Input shape can be (32, 42, 3) or (32, 126).
    Output shape will match input structure (32, 42, 3).
    
    Args:
        sequence: np.ndarray of shape (32, 42, 3) or (32, 126)
        
    Returns:
        np.ndarray of shape (32, 42, 3), float32.
    """
    sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32).copy()
    
    original_shape = sequence.shape
    if sequence.ndim == 2 and sequence.shape[1] == 126:
        sequence = sequence.reshape(sequence.shape[0], 42, 3)
    elif sequence.ndim == 3 and sequence.shape[1:] == (42, 3):
        pass
    else:
        raise ValueError(f"Invalid sequence shape: {original_shape}. Expected (T, 42, 3) or (T, 126)")

    num_frames = sequence.shape[0]
    normalized_sequence = np.zeros_like(sequence, dtype=np.float32)

    for frame_idx in range(num_frames):
        frame = sequence[frame_idx]
        hand1 = frame[:21]
        hand2 = frame[21:42]

        normalized_sequence[frame_idx, :21] = normalize_hand(hand1)
        normalized_sequence[frame_idx, 21:42] = normalize_hand(hand2)

    return normalized_sequence

