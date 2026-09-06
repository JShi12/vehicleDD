"""FastAPI inference service.

`ultralytics` is AGPL-3.0-licensed; this service is a public network-accessible use of it, which
AGPL's network-use clause reaches. Since the full source is already public, `/health` surfaces the
repo URL (also in HealthResponse's default) to keep that obligation trivially satisfied.
"""
from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
from ultralytics import YOLO

from cardd import __version__
from cardd.serving.schemas import BoundingBox, Detection, HealthResponse, PredictResponse
from cardd.serving.weights import resolve_and_download_champion
from cardd.util import git_sha


@asynccontextmanager
async def lifespan(app: FastAPI):
    weights_path = resolve_and_download_champion()
    app.state.model = YOLO(weights_path)
    app.state.model_source = weights_path
    yield


app = FastAPI(title="CarDD vehicle damage detector", version=__version__, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model = getattr(app.state, "model", None)
    return HealthResponse(
        status="ok" if model is not None else "loading",
        model_loaded=model is not None,
        model_source=getattr(app.state, "model_source", ""),
        git_sha=git_sha(),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.7, ge=0.0, le=1.0),
) -> PredictResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image upload, got content-type {file.content_type!r}",
        )

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}") from e

    model: YOLO = app.state.model
    start = time.monotonic()
    results = model.predict(source=image, conf=conf, iou=iou, verbose=False)
    inference_ms = (time.monotonic() - start) * 1000

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        detections.append(Detection(
            class_id=class_id,
            class_name=model.names[class_id],
            confidence=float(box.conf[0]),
            box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        ))

    return PredictResponse(
        detections=detections,
        image_width=image.width,
        image_height=image.height,
        inference_ms=inference_ms,
        model_source=app.state.model_source,
    )
