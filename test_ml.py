import torch
from scipy.io import loadmat
from NTU_Fi_model import NTU_Fi_ResNet18


# 1. Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# 2. Create model
model = NTU_Fi_ResNet18(num_classes=6)


# 3. Load pretrained weights
checkpoint = torch.load(
    "NTU-Fi-HAR_ResNet18.pt",
    map_location="cpu",
    weights_only=False
)

model.load_state_dict(checkpoint)
model.eval()
model = model.to(device)

print("Model loaded successfully!")


# 4. Prediction function
def predict_activity(raw_csi):

    processed = (raw_csi - 42.3199) / 4.9802

    processed = processed[:, ::4]

    processed = processed.reshape(3, 114, 500)

    input_tensor = torch.FloatTensor(processed)

    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)

    predicted_index = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0, predicted_index].item()

    model_classes = [
        "clean",
        "run",
        "box",
        "walk",
        "circle",
        "fall"
    ]

    predicted_label = model_classes[predicted_index]

    return {
        "activity": predicted_label,
        "confidence": float(confidence)
    }


# 5. Load an actual NTU-Fi test sample
path = "Data/NTU-Fi_HAR/test_amp/circle/circle189.mat"

raw_csi = loadmat(path)["CSIamp"]

print("CSI shape:", raw_csi.shape)


# 6. Run prediction
result = predict_activity(raw_csi)

print("Prediction:", result)