# UNSEEN — WiFi CSI Human Sensing

> Privacy-aware human presence and activity sensing using wireless signals.

UNSEEN is a prototype for detecting human motion and activity from **WiFi Channel State Information (CSI)** instead of relying on cameras or wearable devices.

The project combines a research-backed WiFi-CSI activity-recognition model with a web-based inference and monitoring pipeline.

## Demo

- **Live Dashboard:** https://wifi-csi-human-sensing.onrender.com/
- **GitHub Repository:** https://github.com/r7dhi/wifi-csi-human-sensing

---

## What is the idea?

When a person moves through a wireless environment, the propagation of WiFi signals changes.

These changes are represented in **Channel State Information (CSI)**.

Our basic pipeline is:

**WiFi CSI → Preprocessing → Motion/Activity Gate → ML Inference → Prediction → FastAPI/WebSocket → Dashboard**

The current activity-recognition model contains six classes:

- Clean
- Run
- Box
- Walk
- Circle
- Fall

---

## Why UNSEEN?

Traditional monitoring systems often rely on cameras or wearable devices.

UNSEEN explores a different sensing modality: **wireless signals**.

Instead of capturing a conventional image of a person, the system analyzes changes in the wireless channel caused by movement.

This creates a potential privacy-preserving alternative for environments where activity information is useful but continuous visual monitoring may be undesirable.

---

## Current Prototype

The current public prototype uses **recorded/benchmark CSI samples** rather than a live physical WiFi-router-to-dashboard CSI acquisition setup.

The backend replays sample CSI data, preprocesses it, runs the trained model, and sends the resulting prediction to the browser through a WebSocket.

Some dashboard presentation elements are also designed for demonstration purposes.

**The current deployment should not be interpreted as a fully live RF sensing system.**

### Current limitations

- Live CSI acquisition from dedicated WiFi sensing hardware is not implemented yet.
- The current motion/presence gate is a prototype motion/activity filter, not a scientifically validated general-purpose human-presence detector.
- The current classifier is not designed or validated for separating multiple people and attributing activities to individual people.
- Testing is currently based on benchmark/recorded CSI data.
- Broader testing is needed across different people, rooms, layouts, and device positions.

---

## Machine Learning

The ML foundation is based on the **NTU-Fi-HAR dataset/model and ResNet18 architecture** from the WiFi-CSI-Sensing-Benchmark / SenseFi research ecosystem.

We use the trained model for inference rather than claiming to have trained a new activity-recognition model from scratch.

### Preprocessing

The CSI input is processed using the preprocessing used for the NTU-Fi-HAR model:

```python
x = (x - 42.3199) / 4.9802
x = x[:, ::4]
x = x.reshape(3, 114, 500)
