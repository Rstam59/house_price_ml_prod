from house_ml.config import load_config


def test_load_config():
    config = load_config()

    assert config.data.csv_name == "housing.csv"

    assert config.split.test_size == 0.2

    assert config.target.column == "median_house_value"

    assert config.model.type == "random_forest"
