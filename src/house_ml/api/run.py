import uvicorn

from house_ml.config import load_config


def main() -> None:
    config = load_config()

    uvicorn.run(
        "house_ml.api.app:app", host=config.api.host, port=config.api.port, reload=False
    )


if __name__ == "__main__":
    main()
