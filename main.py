from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
import asyncio
from predictor import get_prediction

app = FastAPI()


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        prediction = get_prediction()
        await websocket.send_json(prediction)
        await asyncio.sleep(2)