import numpy as np
from scipy.io import loadmat
import onnxruntime as ort


model_classes = [
    "clean",
    "run",
    "box",
    "walk",
    "circle",
    "fall"
]

MOTION_THRESHOLD = 0.10

session = ort.InferenceSession(
    "NTU-Fi-HAR_ResNet18.onnx",
    providers=["CPUExecutionProvider"]
)


def presence_gate(raw_csi):
    processed = (raw_csi - 42.3199) / 4.9802
    processed = processed[:, ::4]
    processed = processed.reshape(3, 114, 500)

    motion_score = np.var(processed, axis=2).mean()
    detected = motion_score >= MOTION_THRESHOLD

    return detected, motion_score, processed


def predict_csi(raw_csi):
    detected, motion_score, processed = presence_gate(raw_csi)

    if not detected:
        return {
            "presence": False,
            "activity": "none",
            "confidence": 0.0,
            "motion_score": float(motion_score)
        }

    input_tensor = processed.astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    output = session.run(
        None,
        {"input": input_tensor}
    )[0]

    scores = output[0]

    exp_scores = np.exp(scores - np.max(scores))
    probabilities = exp_scores / exp_scores.sum()

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    activity = model_classes[predicted_index]

    return {
        "presence": True,
        "activity": activity,
        "confidence": confidence,
        "motion_score": float(motion_score)
    }


SAMPLE_PATHS = [
    "six_test_samples/box/box186.mat",
    "six_test_samples/circle/circle189.mat",
    "six_test_samples/clean/clean195.mat",
    "six_test_samples/fall/fall174.mat",
    "six_test_samples/run/run17.mat",
    "six_test_samples/walk/walk17.mat",
]

sample_index = 0


def get_prediction():
    global sample_index

    path = SAMPLE_PATHS[sample_index]
    sample_index = (sample_index + 1) % len(SAMPLE_PATHS)

    raw_csi = loadmat(path)["CSIamp"]

    return predict_csi(raw_csi)