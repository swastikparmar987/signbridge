import torch
from pathlib import Path

# --------------------------------------------------
# PATH
# --------------------------------------------------

CHECKPOINT_PATH = Path(
    "/Users/pro/Desktop/projects/signbridge/trained_models/best_gru_normalized.pth"
)

print("=" * 70)
print("🔍 SIGNBRIDGE CHECKPOINT INSPECTOR")
print("=" * 70)

print("\nCheckpoint path:")
print(CHECKPOINT_PATH)

print("\nExists:", CHECKPOINT_PATH.exists())

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"\n❌ Checkpoint not found:\n{CHECKPOINT_PATH}"
    )

# --------------------------------------------------
# LOAD CHECKPOINT
# --------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

print("\nCheckpoint type:")
print(type(checkpoint))

# --------------------------------------------------
# INSPECT CONTENTS
# --------------------------------------------------

if isinstance(checkpoint, dict):

    print("\n📦 Checkpoint keys:")

    for key in checkpoint.keys():
        print(" -", key)

    # Detect state_dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("\n✅ Found: model_state_dict")

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print("\n✅ Found: state_dict")

    else:
        state_dict = checkpoint
        print("\n✅ Assuming checkpoint itself is state_dict")

else:
    state_dict = checkpoint

# --------------------------------------------------
# PARAMETER INSPECTION
# --------------------------------------------------

print("\n" + "=" * 70)
print("🧠 MODEL PARAMETERS")
print("=" * 70)

for name, tensor in state_dict.items():

    if hasattr(tensor, "shape"):
        print(f"{name:<35} {tuple(tensor.shape)}")

# --------------------------------------------------
# AUTOMATIC ARCHITECTURE CLUES
# --------------------------------------------------

print("\n" + "=" * 70)
print("🔎 ARCHITECTURE CLUES")
print("=" * 70)

if "gru.weight_ih_l0" in state_dict:

    weight = state_dict["gru.weight_ih_l0"]

    print("\nGRU detected ✅")

    print("Input size:",
          weight.shape[1])

    hidden_size = weight.shape[0] // 3

    print("Hidden size:",
          hidden_size)

    gru_layers = len([
        key for key in state_dict.keys()
        if "weight_ih_l" in key
    ])

    print("GRU layers:",
          gru_layers)

if "classifier.weight" in state_dict:

    classifier = state_dict["classifier.weight"]

    print("\nClassifier detected ✅")

    print("Number of classes:",
          classifier.shape[0])

    print("Classifier input:",
          classifier.shape[1])

print("\n" + "=" * 70)
print("✅ CHECKPOINT INSPECTION COMPLETE")
print("=" * 70)