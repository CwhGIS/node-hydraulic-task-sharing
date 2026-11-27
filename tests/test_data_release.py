import csv
import math
import unittest
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "pooled_metrics"
SEED_VIEW_DIR = DATA_DIR / "by_seed"
SEEDS = ("3407", "3408", "3409", "3410", "3411")
METHODS = ("fixed_loss", "gradnorm")
CONFIG_IDS = {f"M{index:02d}" for index in range(1, 32)}
TARGETS = {"Depth", "Head", "LatIn", "TotIn", "Flood"}
EXPECTED_COLUMNS = [
    "seed",
    "method",
    "config_id",
    "configuration",
    "target",
    "horizon",
    "val_r2",
    "val_rmse",
    "val_mae",
    "test_r2",
    "test_rmse",
    "test_mae",
    "500A_r2",
    "500A_rmse",
    "500A_mae",
]
SPLIT_METRICS = tuple(
    f"{split}_{metric}"
    for split in ("val", "test", "500A")
    for metric in ("r2", "rmse", "mae")
)


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


class DataReleaseTest(unittest.TestCase):
    def test_only_by_seed_release_is_present(self):
        self.assertTrue(SEED_VIEW_DIR.is_dir())
        self.assertEqual(
            sorted(path.name for path in DATA_DIR.iterdir() if path.is_file()),
            [],
        )
        self.assertEqual(
            sorted(path.name for path in DATA_DIR.iterdir() if path.is_dir()),
            ["by_seed"],
        )

    def test_seed_and_model_coverage(self):
        self.assertEqual(
            sorted(path.name for path in SEED_VIEW_DIR.iterdir() if path.is_dir()),
            list(SEEDS),
        )
        expected_files = {
            f"{config_id}_{method}.csv"
            for config_id in CONFIG_IDS
            for method in METHODS
        }
        self.assertEqual(
            len(list(SEED_VIEW_DIR.rglob("*.csv"))),
            len(SEEDS) * len(expected_files),
        )
        for seed in SEEDS:
            with self.subTest(seed=seed):
                paths = list((SEED_VIEW_DIR / seed).glob("*.csv"))
                self.assertEqual({path.name for path in paths}, expected_files)

    def test_each_table_contains_all_split_metrics(self):
        target_sets = {}
        configuration_names = {}
        for path in sorted(SEED_VIEW_DIR.rglob("*.csv")):
            columns, rows = read_rows(path)
            with self.subTest(file=path.relative_to(SEED_VIEW_DIR).as_posix()):
                self.assertEqual(columns, EXPECTED_COLUMNS)
                self.assertGreater(len(rows), 0)
                seed = path.parent.name
                config_id, method = path.stem.split("_", 1)
                self.assertEqual({row["seed"] for row in rows}, {seed})
                self.assertEqual({row["method"] for row in rows}, {method})
                self.assertEqual({row["config_id"] for row in rows}, {config_id})
                self.assertIn(config_id, CONFIG_IDS)
                self.assertIn(method, METHODS)
                self.assertEqual(
                    len({(row["target"], row["horizon"]) for row in rows}),
                    len(rows),
                )
                targets = {row["target"] for row in rows}
                self.assertTrue(targets <= TARGETS)
                self.assertEqual(len(rows), len(targets) * 6)
                self.assertEqual(
                    {(row["target"], int(row["horizon"])) for row in rows},
                    {(target, horizon) for target in targets for horizon in range(1, 7)},
                )
                target_sets.setdefault(config_id, targets)
                self.assertEqual(target_sets[config_id], targets)
                names = {row["configuration"] for row in rows}
                self.assertEqual(len(names), 1)
                configuration_names.setdefault(config_id, names)
                self.assertEqual(configuration_names[config_id], names)
                for row in rows:
                    for metric in SPLIT_METRICS:
                        value = float(row[metric])
                        self.assertTrue(math.isfinite(value), f"non-finite {metric}")

        self.assertEqual(set(target_sets), CONFIG_IDS)
        self.assertEqual(set(configuration_names), CONFIG_IDS)


if __name__ == "__main__":
    unittest.main()
