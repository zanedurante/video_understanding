from video.utils.config_manager import get_config, get_num_workers, get_val_from_config
import os


def test_get_num_workers():
    assert get_num_workers("2") == 2
    num_cpus = os.cpu_count()
    assert get_num_workers("cpus-2") == num_cpus - 2


def test_load_base_config():
    config = get_config("configs/base.yaml")
    assert config is not None

    config_name = config.logger.run_name
    assert config_name is not None


def test_edit_config():
    config = get_config("tests/test_config.yaml")
    assert config.logger.log_every_n_steps == 20
    assert config.model.type == "test_model"


def test_get_val_from_config():
    config = get_config("tests/test_config.yaml")
    val = get_val_from_config(config, "model.type")
    assert val == "test_model"
    val = get_val_from_config(config, "model.backbone_name", "hello world")
    assert val == "hello world"


if __name__ == "__main__":
    test_get_num_workers()
    test_load_base_config()
    test_edit_config()
    test_get_val_from_config()
    print("All tests passed!")
