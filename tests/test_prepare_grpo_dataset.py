import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.prepare_grpo_dataset import audit_dataset, prepare_dataset


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path, task_ids):
    path.write_text(
        "".join(json.dumps({"task_id": task_id}) + "\n" for task_id in task_ids),
        encoding="utf-8",
    )


def _write_parquet(path, task_ids, split):
    rows = [
        {
            "data_source": "shopsimulator",
            "prompt": [{"role": "user", "content": f"task {task_id}"}],
            "ability": "shopping",
            "reward_model": {"style": "rule", "ground_truth": None},
            "extra_info": {"split": split, "index": index, "task_id": task_id},
        }
        for index, task_id in enumerate(task_ids)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_sft(path, manifest_path, task_ids):
    _write_jsonl(path, task_ids)
    manifest_path.write_text(
        json.dumps(
            {
                "buckets": {
                    "foundation": {
                        "train_task_ids": task_ids[:-1],
                        "validation_task_ids": task_ids[-1:],
                    }
                }
            }
        ),
        encoding="utf-8",
    )


class PrepareGrpoDatasetTests(unittest.TestCase):
    def test_filters_both_formats_reindexes_and_writes_auditable_metadata(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            train_jsonl = source / "train.jsonl"
            train_parquet = source / "train.parquet"
            validation_jsonl = source / "validation.jsonl"
            validation_parquet = source / "validation.parquet"
            sft_source = root / "sft.jsonl"
            sft_manifest = root / "manifest.json"
            evaluation = root / "evaluation.jsonl"
            metadata_path = source / "metadata.json"
            _write_jsonl(train_jsonl, [1, 2, 3, 4])
            _write_parquet(train_parquet, [1, 2, 3, 4], "train")
            _write_jsonl(validation_jsonl, [5])
            _write_parquet(validation_parquet, [5], "validation")
            _write_sft(sft_source, sft_manifest, [2, 9])
            _write_jsonl(evaluation, [3])

            metadata = prepare_dataset(
                train_jsonl=train_jsonl,
                train_parquet=train_parquet,
                validation_jsonl=validation_jsonl,
                validation_parquet=validation_parquet,
                sft_source=sft_source,
                sft_manifest=sft_manifest,
                evaluation=evaluation,
                metadata_path=metadata_path,
                output_dir=output,
                root=root,
            )

            self.assertEqual(
                [json.loads(line)["task_id"] for line in (output / "train.jsonl").read_text().splitlines()],
                [1, 4],
            )
            extras = pq.read_table(output / "train.parquet").column("extra_info").to_pylist()
            self.assertEqual([row["task_id"] for row in extras], [1, 4])
            self.assertEqual([row["index"] for row in extras], [0, 1])
            self.assertEqual(metadata["provenance"]["removed_train_task_ids"], [2, 3])
            self.assertEqual(metadata["train"]["tasks"], 2)
            self.assertEqual(metadata["validation"]["tasks"], 1)
            self.assertEqual(
                metadata["train"]["parquet_sha256"],
                hashlib.sha256((output / "train.parquet").read_bytes()).hexdigest(),
            )
            self.assertEqual(metadata["audit"]["active_sft_overlap"], 0)
            self.assertEqual(metadata["audit"]["sft_evaluation_overlap"], 0)
            self.assertEqual(metadata["audit"]["final_evaluation_overlap"], 0)

            rerun = prepare_dataset(
                train_jsonl=train_jsonl,
                train_parquet=train_parquet,
                validation_jsonl=validation_jsonl,
                validation_parquet=validation_parquet,
                sft_source=sft_source,
                sft_manifest=sft_manifest,
                evaluation=evaluation,
                metadata_path=metadata_path,
                output_dir=output,
                root=root,
            )
            self.assertEqual(rerun, metadata)

    def test_rejects_jsonl_parquet_order_mismatch(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_jsonl(root / "train.jsonl", [1, 2])
            _write_parquet(root / "train.parquet", [2, 1], "train")
            _write_jsonl(root / "validation.jsonl", [3])
            _write_parquet(root / "validation.parquet", [3], "validation")
            _write_sft(root / "sft.jsonl", root / "manifest.json", [8, 9])
            _write_jsonl(root / "evaluation.jsonl", [10])

            with self.assertRaisesRegex(ValueError, "order-aligned"):
                prepare_dataset(
                    train_jsonl=root / "train.jsonl",
                    train_parquet=root / "train.parquet",
                    validation_jsonl=root / "validation.jsonl",
                    validation_parquet=root / "validation.parquet",
                    sft_source=root / "sft.jsonl",
                    sft_manifest=root / "manifest.json",
                    evaluation=root / "evaluation.jsonl",
                    metadata_path=root / "metadata.json",
                    output_dir=root / "output",
                    root=root,
                )

    def test_checked_in_grpo_dataset_is_disjoint_and_matches_metadata(self):
        data = ROOT / "data/grpo"
        audit = audit_dataset(
            train_jsonl=data / "train.jsonl",
            train_parquet=data / "train.parquet",
            validation_jsonl=data / "validation.jsonl",
            validation_parquet=data / "validation.parquet",
            sft_source=ROOT / "data/sft_pure_v4/all.jsonl",
            sft_manifest=ROOT / "data/sft_curriculum/manifest.json",
            evaluation=ROOT / "data/evaluation/tasks.jsonl",
        )
        metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(audit, metadata["audit"])
        self.assertEqual(metadata["train"]["tasks"], 994)
        self.assertEqual(metadata["validation"]["tasks"], 50)
        for split in ("train", "validation"):
            for kind in ("jsonl", "parquet"):
                path = ROOT / metadata[split][kind]
                self.assertEqual(
                    metadata[split][f"{kind}_sha256"],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
