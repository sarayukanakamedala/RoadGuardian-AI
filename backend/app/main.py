from pathlib import Path
from uuid import uuid4
import logging
import traceback

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ml.inference.predict import predict_video


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("roadguardian")


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

UPLOAD_DIR = PROJECT_ROOT / "backend" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="RoadGuardian AI",
    description="AI-Based Road Accident Detection & Emergency Response",
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Health routes
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "RoadGuardian AI backend is running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "RoadGuardian AI",
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
    }

    original_filename = file.filename or "uploaded_video"
    extension = Path(original_filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                "Please upload MP4, AVI, MOV, or MKV."
            ),
        )

    temp_filename = f"{uuid4().hex}{extension}"
    temp_path = UPLOAD_DIR / temp_filename

    try:
        logger.info("Receiving video: %s", original_filename)

        # Save uploaded video
        contents = await file.read()

        with open(temp_path, "wb") as buffer:
            buffer.write(contents)

        logger.info(
            "Video saved temporarily: %s (%d bytes)",
            temp_path,
            len(contents),
        )

        # Run ML inference
        logger.info("Starting RoadGuardian inference...")

        prediction_result = predict_video(temp_path)

        logger.info(
            "Raw prediction result: %r",
            prediction_result,
        )

        # -------------------------------------------------
        # Normalize prediction output
        # -------------------------------------------------

        if isinstance(prediction_result, dict):

            prediction = prediction_result.get("prediction")

            accident_probability = prediction_result.get(
                "accident_probability"
            )

            normal_probability = prediction_result.get(
                "normal_probability"
            )

            decision_threshold = prediction_result.get(
                "decision_threshold",
                0.5,
            )

            device = prediction_result.get(
                "device",
                "cpu",
            )

        else:
            raise RuntimeError(
                "predict_video() returned an unexpected result type: "
                f"{type(prediction_result).__name__}. "
                f"Raw result: {prediction_result!r}"
            )

        # Validate required prediction values
        if prediction is None:
            raise RuntimeError(
                "Prediction result does not contain 'prediction'. "
                f"Raw result: {prediction_result!r}"
            )

        if accident_probability is None:
            raise RuntimeError(
                "Prediction result does not contain "
                "'accident_probability'. "
                f"Raw result: {prediction_result!r}"
            )

        if normal_probability is None:
            raise RuntimeError(
                "Prediction result does not contain "
                "'normal_probability'. "
                f"Raw result: {prediction_result!r}"
            )

        logger.info(
            "Prediction successful: %s | accident=%.6f | normal=%.6f",
            prediction,
            float(accident_probability),
            float(normal_probability),
        )

        return {
            "status": "success",
            "filename": original_filename,
            "prediction": prediction,
            "accident_probability": float(accident_probability),
            "normal_probability": float(normal_probability),
            "decision_threshold": float(decision_threshold),
            "device": device,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.error("Prediction failed.")
        logger.error("%s", exc)
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "RoadGuardian prediction failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    finally:
        # Always remove temporary uploaded video
        if temp_path.exists():
            try:
                temp_path.unlink()
                logger.info("Temporary video removed.")
            except Exception as cleanup_error:
                logger.warning(
                    "Could not remove temporary file: %s",
                    cleanup_error,
                )


# ---------------------------------------------------------
# Start with:
#
# uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
# ---------------------------------------------------------