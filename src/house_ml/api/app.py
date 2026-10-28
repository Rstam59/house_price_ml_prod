import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from house_ml.api.routes import router
from house_ml.config import load_config
from house_ml.inference.predictor import HousingPredictor
from house_ml.logging_config import configure_logging

logger = logging.getLogger(__name__)


def create_app(
    predictor_factory: Callable[[], HousingPredictor] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ) -> AsyncIterator[None]:
        config = load_config()

        configure_logging(config.logging.level)

        logger.info("Starting housing prediction API")

        logger.info("Loading model artifact")

        if predictor_factory is None:
            predictor = HousingPredictor.from_config(config)

        else:
            predictor = predictor_factory()

        app.state.predictor = predictor

        logger.info("Model loaded successfully")

        yield

        logger.info("Shutting down housing prediction API")

        app.state.predictor = None

    app = FastAPI(title="Housing Price api", version="1.0.0", lifespan=lifespan)

    app.include_router(router)

    return app


app = create_app()
