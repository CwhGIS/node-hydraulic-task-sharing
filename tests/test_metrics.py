import unittest

import numpy as np

from node_hydraulic_task_sharing.node_metrics import validation_loss_by_task


class ValidationLossTest(unittest.TestCase):
    def test_matches_elementwise_mse_per_task(self) -> None:
        predictions = np.array(
            [
                [[[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[2.0, 4.0], [6.0, 8.0]], [[1.0, 1.0], [1.0, 1.0]]],
            ]
        )
        targets = np.zeros_like(predictions)
        losses = validation_loss_by_task(predictions, targets)
        self.assertAlmostEqual(losses[0], np.square(predictions[:, 0]).mean())
        self.assertAlmostEqual(losses[1], np.square(predictions[:, 1]).mean())

    def test_ignores_non_finite_entries(self) -> None:
        predictions = np.array([[[[1.0, np.nan]]]])
        targets = np.zeros_like(predictions)
        losses = validation_loss_by_task(predictions, targets)
        self.assertAlmostEqual(losses[0], 1.0)

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validation_loss_by_task(np.zeros((1, 2, 3, 4)), np.zeros((1, 2, 3, 5)))


if __name__ == "__main__":
    unittest.main()