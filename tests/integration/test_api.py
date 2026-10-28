from fastapi.testclient import TestClient

from house_ml.api.app import create_app
from house_ml.inference.schemas import HousingPredictionResponse


class FakePredictor:
    def predict(self, request) -> HousingPredictionResponse:
        return HousingPredictionResponse(predicted_median_house_value=250000.0)


def fake_predictor_factory():
    return FakePredictor()


def valid_payload():
    return {
        "longitude": -122,
        "latitude": -30,
        "housing_median_age": 20,
        "total_rooms": 10000,
        "total_bedrooms": 10,
        "population": 300,
        "households": 100,
        "median_income": 4,
        "ocean_proximity": "1<H OCEAN",
    }


def test_health_endpoint():
    app = create_app(predictor_factory=fake_predictor_factory)

    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200

        assert response.json() == {"status": "ok"}


def test_ready_endpoint():
    app = create_app(predictor_factory=fake_predictor_factory)

    with TestClient(app) as client:
        response = client.get("/ready")

        assert response.status_code == 200

        assert response.json() == {"status": "ready"}


def test_predict_endpoint():
    app = create_app(predictor_factory=fake_predictor_factory)

    with TestClient(app) as client:
        response = client.post("/predict", json=valid_payload())

        assert response.status_code == 200

        assert response.json() == {"predicted_median_house_value": (250000.0)}


def test_predict_rejects_invalid_latitude():
    app = create_app(predictor_factory=fake_predictor_factory)

    payload = valid_payload()

    payload["latitude"] = 500

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

        assert response.status_code == 422


class BrokenPredictor:
    def predict(self, request):
        raise RuntimeError("Something exploded")


def broken_predictor_factory():
    return BrokenPredictor()


def test_predict_handles_model_failure():
    app = create_app(predictor_factory=broken_predictor_factory)

    with TestClient(app) as client:
        response = client.post("/predict", json=valid_payload())

    assert response.status_code == 500

    assert response.json() == {"detail": "Prediction failed"}
