import unittest

import torch

from node_hydraulic_task_sharing.config import ConfigNode
from node_hydraulic_task_sharing.node_metrics import configuration_validation_loss
from node_hydraulic_task_sharing.run_node_mtl_combinations import define_experiments
from node_hydraulic_task_sharing.utils import GraphWaveNetMT


class NodeSmokeTest(unittest.TestCase):
    def test_model_forward_shape(self) -> None:
        model = GraphWaveNetMT(
            N=4,
            C_in=7,
            C_out=2,
            C_res=16,
            C_skip=16,
            K=2,
            dilations=(1, 2),
            use_adaptive=False,
            num_static_supports=1,
            horizon=6,
        )
        model.eval()
        with torch.no_grad():
            output = model(
                torch.randn(2, 7, 4, 6),
                [torch.eye(4)],
            )
        self.assertEqual(tuple(output.shape), (2, 2, 4, 6))

    def test_node_targets_and_all_task_subsets(self) -> None:
        config = ConfigNode()
        self.assertEqual(config.target_names, ["Depth", "Head", "LatIn", "TotIn", "Flood"])
        experiments = define_experiments()
        self.assertEqual(len(experiments), 31)
        self.assertEqual(len({item["name"] for item in experiments}), 31)

    def test_configuration_validation_loss_sums_all_included_tasks(self) -> None:
        self.assertAlmostEqual(configuration_validation_loss([0.25, 0.50, 0.75]), 1.50)


if __name__ == "__main__":
    unittest.main()
