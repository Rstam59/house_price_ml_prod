import logging

from fastapi import APIRouter, HTTPException, Request, status

from house_ml.inference.schemas import (
    HousingPredictionRequest,
    HousingPredictionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def readiness(request: Request) -> dict[str, str]:
    predictor = getattr(request.app.state, "predictor", None)

    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready"
        )
    return {"status": "ready"}


@router.post("/predict", response_model=HousingPredictionResponse)
def predict(payload: HousingPredictionRequest, request: Request):
    predictor = getattr(request.app.state, "predictor", None)

    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready"
        )

    try:
        return predictor.predict(payload)
    except Exception as err:
        logger.exception("Prediction Failed")

        raise HTTPException(status_code=500, detail="Prediction failed") from err
