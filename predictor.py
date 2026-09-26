import torch
import numpy as np
from scipy.io import loadmat
from NTU_Fi_model import NTU_Fi_ResNet18


# --------------------------------------------------
# Load model
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NTU_Fi_ResNet18(num_classes=6)

checkpoint = torch.load(
    "NTU-Fi-HAR_ResNet18.pt",
    map_location="cpu",
    weights_only=False
)

model.load_state_dict(checkpoint)
model.eval()
model = model.to(device)


model_classes = [
    "clean",
    "run",
    "box",
    "walk",
    "circle",
    "fall"
]


# --------------------------------------------------
# Presence / motion gate
# --------------------------------------------------

MOTION_THRESHOLD = 0.10


def presence_gate(raw_csi):
    # Official NTU-Fi preprocessing
    processed = (raw_csi - 42.3199) / 4.9802
    processed = processed[:, ::4]
    processed = processed.reshape(3, 114, 500)

    # Variance-based motion score
    motion_score = np.var(processed, axis=2).mean()

    # Presence/motion decision
    detected = motion_score >= MOTION_THRESHOLD

    return detected, motion_score, processed


# --------------------------------------------------
# Full CSI prediction
# --------------------------------------------------

def predict_csi(raw_csi):
    detected, motion_score, processed = presence_gate(raw_csi)

    if not detected:
        return {
            "presence": False,
            "activity": "none",
            "confidence": 0.0,
            "motion_score": float(motion_score)
        }

    input_tensor = torch.FloatTensor(processed)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)

    predicted_index = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, predicted_index].item()

    activity = model_classes[predicted_index]

    return {
        "presence": True,
        "activity": activity,
        "confidence": float(confidence),
        "motion_score": float(motion_score)
    }


# --------------------------------------------------
# Six real test samples for replay
# --------------------------------------------------

SAMPLE_PATHS = [
    "six_test_samples/box/box186.mat",
    "six_test_samples/circle/circle189.mat",
    "six_test_samples/clean/clean195.mat",
    "six_test_samples/fall/fall174.mat",
    "six_test_samples/run/run17.mat",
    "six_test_samples/walk/walk17.mat",
]

sample_index = 0


# --------------------------------------------------
# Backend prediction
# --------------------------------------------------

def get_prediction():
    global sample_index

    path = SAMPLE_PATHS[sample_index]

    sample_index = (sample_index + 1) % len(SAMPLE_PATHS)

    raw_csi = loadmat(path)["CSIamp"]

    return predict_csi(raw_csi)