from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class DataConfig:
    url: str
    directory: str
    tarball_name: str
    extracted_directory: str
    csv_name: str


@dataclass(frozen=True)
class IncomeCategoryConfig:
    bins: list[float]
    labels: list[float]


@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    random_state: int
    income_category: IncomeCategoryConfig


@dataclass(frozen=True)
class TargetConfig:
    column: str


@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int
    max_depth: int | None
    random_state: int
    n_jobs: int


@dataclass(frozen=True)
class ModelConfig:
    type: str
    random_forest: RandomForestConfig


@dataclass(frozen=True)
class EvaluationConfig:
    cv_folds: int
    scoring: str


@dataclass(frozen=True)
class ArtifactConfig:
    directory: str
    model_filename: str
    metadata_filename: str


@dataclass(frozen=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True)
class ApiConfig:
    host: str
    port: int


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    split: SplitConfig
    target: TargetConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    artifacts: ArtifactConfig
    logging: LoggingConfig
    api: ApiConfig


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """
    Load the configuration from a YAML file.
    """

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r") as file:
        raw = yaml.safe_load(file)

    income_config = IncomeCategoryConfig(**raw["split"]["income_category"])

    split_config = SplitConfig(
        test_size=raw["split"]["test_size"],
        random_state=raw["split"]["random_state"],
        income_category=income_config,
    )
    rf_config = RandomForestConfig(**raw["model"]["random_forest"])

    model_config = ModelConfig(type=raw["model"]["type"], random_forest=rf_config)

    return AppConfig(
        data=DataConfig(**raw["data"]),
        split=split_config,
        target=TargetConfig(**raw["target"]),
        model=model_config,
        evaluation=EvaluationConfig(**raw["evaluation"]),
        artifacts=ArtifactConfig(**raw["artifacts"]),
        logging=LoggingConfig(**raw["logging"]),
        api=ApiConfig(**raw["api"]),
    )


if __name__ == "__main__":
    config = load_config()

    print(config)

    print(config.model.random_forest)
