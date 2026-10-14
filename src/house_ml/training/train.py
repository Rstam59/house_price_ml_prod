from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline

import logging

from house_ml.artifacts.model_io import save_metadata, save_model
from house_ml.config import AppConfig, load_config
from house_ml.data.dataset import split_features_target
from house_ml.data.load import load_housing_data
from house_ml.data.split import stratified_split
from house_ml.features.preprocessing import build_preprocessor
from house_ml.models.model import build_model
from house_ml.logging_config import configure_logging
from house_ml.evaluation.cross_validation import run_cross_validation
from house_ml.evaluation.metrics import validate_predictions, calculate_regression_metrics

logger = logging.getLogger(__name__)

def build_training_pipeline(
    config: AppConfig
) -> Pipeline:
    '''
    Build complete preprocessing + model pipeline
    '''
    preprocessor = build_preprocessor()

    model = build_model(config.model)

    pipeline = Pipeline(
        steps = [
            ('preprocessing',
            preprocessor
            ),
            (
                'model',
                model
            )

        ]
    )

    return pipeline


def main() -> None:

    config = load_config()

    configure_logging(config.logging.level)
    
    logger.info(
        'Starting training pipeline'
    )
    #Load data
    housing = load_housing_data(config.data)

    #Split train/test
    train_set, test_set = stratified_split(housing, config.split)

    X_train, y_train = split_features_target(train_set, config.target)
    X_test, y_test = split_features_target(test_set, config.target)

    logger.info(
        'Training rows: %d',
        len(X_train)
    )

    logger.info(
        'Test rows: %d',
        len(X_test)
    )

    # Build pipeline
    pipeline = build_training_pipeline(config)

    cv_metrics = run_cross_validation(pipeline, X_train, y_train, config.evaluation)

    #Final training
    logger.info(
        'Training final model'
        "on full training dataset"
    )

    pipeline.fit(X_train, y_train)

    # Final test evaluation
    predictions = pipeline.predict(X_test)

    validate_predictions(predictions)

    test_metrics = calculate_regression_metrics(y_test, predictions)

    logger.info(
        'Test RMSE: %.2f',
        test_metrics['rmse']
    )

    logger.info(
        'Test MAE: %.2f',
        test_metrics['mae']
    )

    logger.info(
        'Test R^2: %.4f',
        test_metrics['r2']
    )

    # Artifact persistance

    all_metrics = {
        **cv_metrics,
        **test_metrics
    }

    save_model(pipeline, config.artifacts)
    save_metadata(all_metrics, config.artifacts)

    logger.info('Training pipeline completed')


if __name__ == '__main__':
    main()