import pytest
import pyrootutils
from omegaconf import open_dict
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig

from src.eval import evaluate

@pytest.mark.slow
def test_metric_learning(tmp_path):
    with initialize(version_base="1.2", config_path="../configs"):
        cfg_eval = compose(config_name="eval.yaml", return_hydra_config=True)

    with open_dict(cfg_eval):
        cfg_eval.paths.output_dir = tmp_path
        cfg_eval.paths.root_dir = str(pyrootutils.find_root())
        cfg_eval.extras.print_config = False
        cfg_eval.extras.enforce_tags = False

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = evaluate(cfg_eval)
    acc = test_metric_dict["test/acc"].item()
    assert abs(acc - 0.926) < 0.001
