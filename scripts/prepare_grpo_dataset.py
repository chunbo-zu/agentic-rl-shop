#!/usr/bin/env python3
"""Filter and audit the canonical GRPO dataset against active held-out sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


SCHEMA_VERSION = "shopping-grpo-dataset-v1"
EXCLUSION_POLICY = "pure-v4-and-final-evaluation-disjoint-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict) or "task_id" not in row:
            raise ValueError(f"{path}:{line_number}: row must contain task_id")
        rows.append(row)
    return rows


def task_ids_from_jsonl(path: Path) -> list[int]:
    return [int(row["task_id"]) for row in read_jsonl(path)]


def task_ids_from_parquet(table) -> list[int]:
    if "extra_info" not in table.column_names:
        raise ValueError("GRPO Parquet must contain extra_info")
    values = table.column("extra_info").to_pylist()
    try:
        return [int(value["task_id"]) for value in values]
    except (KeyError, TypeError) as exc:
        raise ValueError("GRPO Parquet extra_info must contain task_id") from exc


def _require_unique(ids: list[int], label: str) -> None:
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains duplicate task IDs")


def _portable_path(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(Path(path).resolve())


def _load_active_sft_ids(source: Path, manifest_path: Path) -> set[int]:
    source_ids = set(task_ids_from_jsonl(source))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest_ids = {
        int(task_id)
        for bucket in manifest.get("buckets", {}).values()
        for split in ("train_task_ids", "validation_task_ids")
        for task_id in bucket.get(split, [])
    }
    if source_ids != manifest_ids:
        raise ValueError("active SFT source and curriculum manifest task IDs differ")
    return source_ids


def _load_parquet(path: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("preparing GRPO Parquet requires pyarrow from the grpo extra") from exc
    return pq.read_table(path)


def _filtered_split(jsonl_path: Path, parquet_path: Path, split: str, excluded: set[int]):
    json_rows = read_jsonl(jsonl_path)
    table = _load_parquet(parquet_path)
    json_ids = [int(row["task_id"]) for row in json_rows]
    parquet_ids = task_ids_from_parquet(table)
    _require_unique(json_ids, f"{split} JSONL")
    _require_unique(parquet_ids, f"{split} Parquet")
    if json_ids != parquet_ids:
        raise ValueError(f"{split} JSONL and Parquet task IDs are not order-aligned")

    parquet_rows = table.to_pylist()
    for index, extra in enumerate(row["extra_info"] for row in parquet_rows):
        if str(extra.get("split")) != split:
            raise ValueError(f"{split} Parquet row {index} has split={extra.get('split')!r}")
        if int(extra.get("index", -1)) != index:
            raise ValueError(f"{split} Parquet extra_info.index is not contiguous")

    keep_indices = [index for index, task_id in enumerate(json_ids) if task_id not in excluded]
    removed = sorted(set(json_ids) & excluded)
    kept_json = [json_rows[index] for index in keep_indices]
    kept_parquet = [parquet_rows[index] for index in keep_indices]
    for index, (json_row, parquet_row) in enumerate(zip(kept_json, kept_parquet)):
        if isinstance(json_row.get("extra_info"), dict):
            json_row["extra_info"]["index"] = index
            json_row["extra_info"]["split"] = split
        parquet_row["extra_info"]["index"] = index
        parquet_row["extra_info"]["split"] = split

    if not removed:
        return kept_json, table, removed

    import pyarrow as pa

    filtered_table = pa.Table.from_pylist(kept_parquet, schema=table.schema)
    return kept_json, filtered_table, removed


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_parquet(path: Path, table) -> None:
    import pyarrow.parquet as pq

    pq.write_table(table, path, compression="snappy", version="2.6")


def _metadata_entry(path: Path, root: Path, tasks: int, kind: str) -> dict:
    return {
        kind: _portable_path(path, root),
        f"{kind}_sha256": sha256_file(path),
        "tasks": int(tasks),
    }


def audit_dataset(
    *,
    train_jsonl: Path,
    train_parquet: Path,
    validation_jsonl: Path,
    validation_parquet: Path,
    sft_source: Path,
    sft_manifest: Path,
    evaluation: Path,
) -> dict:
    train_json_ids = task_ids_from_jsonl(train_jsonl)
    train_parquet_ids = task_ids_from_parquet(_load_parquet(train_parquet))
    validation_json_ids = task_ids_from_jsonl(validation_jsonl)
    validation_parquet_ids = task_ids_from_parquet(_load_parquet(validation_parquet))
    _require_unique(train_json_ids, "train JSONL")
    _require_unique(validation_json_ids, "validation JSONL")
    if train_json_ids != train_parquet_ids:
        raise ValueError("train JSONL and Parquet task IDs are not order-aligned")
    if validation_json_ids != validation_parquet_ids:
        raise ValueError("validation JSONL and Parquet task IDs are not order-aligned")

    train_ids = set(train_json_ids)
    validation_ids = set(validation_json_ids)
    sft_ids = _load_active_sft_ids(sft_source, sft_manifest)
    evaluation_ids = set(task_ids_from_jsonl(evaluation))
    overlaps = {
        "train_validation": sorted(train_ids & validation_ids),
        "sft_train": sorted(sft_ids & train_ids),
        "sft_validation": sorted(sft_ids & validation_ids),
        "sft_evaluation": sorted(sft_ids & evaluation_ids),
        "evaluation_train": sorted(evaluation_ids & train_ids),
        "evaluation_validation": sorted(evaluation_ids & validation_ids),
    }
    nonempty = {name: ids for name, ids in overlaps.items() if ids}
    if nonempty:
        raise ValueError("GRPO dataset overlap: " + json.dumps(nonempty, sort_keys=True))
    return {
        "jsonl_parquet_order_aligned": True,
        "unique_task_ids": True,
        "train_validation_overlap": 0,
        "active_sft_overlap": 0,
        "sft_evaluation_overlap": 0,
        "final_evaluation_overlap": 0,
        "train_tasks": len(train_ids),
        "validation_tasks": len(validation_ids),
    }


def prepare_dataset(
    *,
    train_jsonl: Path,
    train_parquet: Path,
    validation_jsonl: Path,
    validation_parquet: Path,
    sft_source: Path,
    sft_manifest: Path,
    evaluation: Path,
    metadata_path: Path,
    output_dir: Path,
    root: Path,
) -> dict:
    paths = [
        train_jsonl,
        train_parquet,
        validation_jsonl,
        validation_parquet,
        sft_source,
        sft_manifest,
        evaluation,
    ]
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise ValueError("missing input file(s): " + ", ".join(missing))

    active_sft_ids = _load_active_sft_ids(sft_source, sft_manifest)
    evaluation_ids = set(task_ids_from_jsonl(evaluation))
    excluded = active_sft_ids | evaluation_ids
    train_rows, train_table, removed_train = _filtered_split(
        train_jsonl, train_parquet, "train", excluded
    )
    validation_rows, validation_table, removed_validation = _filtered_split(
        validation_jsonl, validation_parquet, "validation", excluded
    )
    train_ids = {int(row["task_id"]) for row in train_rows}
    validation_ids = {int(row["task_id"]) for row in validation_rows}
    if train_ids & validation_ids:
        raise ValueError("filtered GRPO train and validation task IDs overlap")

    previous = {}
    if Path(metadata_path).is_file():
        previous = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    previous_provenance = previous.get("provenance") or {}
    cumulative_removed_train = sorted(
        {int(value) for value in previous_provenance.get("removed_train_task_ids", [])}
        | set(removed_train)
    )
    cumulative_removed_validation = sorted(
        {int(value) for value in previous_provenance.get("removed_validation_task_ids", [])}
        | set(removed_validation)
    )
    original_train_tasks = int(
        previous_provenance.get(
            "original_train_tasks",
            len(train_rows) + len(cumulative_removed_train),
        )
    )
    original_validation_tasks = int(
        previous_provenance.get(
            "original_validation_tasks",
            len(validation_rows) + len(cumulative_removed_validation),
        )
    )
    input_snapshot = previous_provenance.get("input_snapshot") or {
        "train_jsonl_sha256": sha256_file(train_jsonl),
        "train_parquet_sha256": sha256_file(train_parquet),
        "validation_jsonl_sha256": sha256_file(validation_jsonl),
        "validation_parquet_sha256": sha256_file(validation_parquet),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train_jsonl": output_dir / "train.jsonl",
        "train_parquet": output_dir / "train.parquet",
        "validation_jsonl": output_dir / "validation.jsonl",
        "validation_parquet": output_dir / "validation.parquet",
        "metadata": output_dir / "metadata.json",
    }
    with tempfile.TemporaryDirectory(prefix=".prepare-grpo-", dir=output_dir) as raw_temp:
        temporary = Path(raw_temp)
        temp_paths = {name: temporary / path.name for name, path in output_paths.items()}

        if removed_train:
            _write_jsonl(temp_paths["train_jsonl"], train_rows)
            _write_parquet(temp_paths["train_parquet"], train_table)
        else:
            shutil.copyfile(train_jsonl, temp_paths["train_jsonl"])
            shutil.copyfile(train_parquet, temp_paths["train_parquet"])
        if removed_validation:
            _write_jsonl(temp_paths["validation_jsonl"], validation_rows)
            _write_parquet(temp_paths["validation_parquet"], validation_table)
        else:
            shutil.copyfile(validation_jsonl, temp_paths["validation_jsonl"])
            shutil.copyfile(validation_parquet, temp_paths["validation_parquet"])

        audit = audit_dataset(
            train_jsonl=temp_paths["train_jsonl"],
            train_parquet=temp_paths["train_parquet"],
            validation_jsonl=temp_paths["validation_jsonl"],
            validation_parquet=temp_paths["validation_parquet"],
            sft_source=sft_source,
            sft_manifest=sft_manifest,
            evaluation=evaluation,
        )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "environment": "shopsimulator-environment-v2.1",
            "reward": "shopsimulator-reward-v3",
            "provenance": {
                "rebuilt": True,
                "seed": int(previous_provenance.get("seed", 20260808)),
                "reason": "Keep active Pure V4 SFT, GRPO, and Final-200 task-disjoint",
                "exclusion_policy": EXCLUSION_POLICY,
                "note": "No difficulty probe; probe_steps and length_bucket are unavailable",
                "original_train_tasks": original_train_tasks,
                "original_validation_tasks": original_validation_tasks,
                "removed_train_task_ids": cumulative_removed_train,
                "removed_validation_task_ids": cumulative_removed_validation,
                "input_snapshot": input_snapshot,
                "exclusion_sources": {
                    "sft_source": {
                        "path": _portable_path(sft_source, root),
                        "sha256": sha256_file(sft_source),
                        "tasks": len(active_sft_ids),
                    },
                    "sft_manifest": {
                        "path": _portable_path(sft_manifest, root),
                        "sha256": sha256_file(sft_manifest),
                        "tasks": len(active_sft_ids),
                    },
                    "final_evaluation": {
                        "path": _portable_path(evaluation, root),
                        "sha256": sha256_file(evaluation),
                        "tasks": len(evaluation_ids),
                    },
                },
            },
            "train": {
                **_metadata_entry(
                    temp_paths["train_jsonl"], root, len(train_rows), "jsonl"
                ),
                **_metadata_entry(
                    temp_paths["train_parquet"], root, len(train_rows), "parquet"
                ),
            },
            "validation": {
                **_metadata_entry(
                    temp_paths["validation_jsonl"], root, len(validation_rows), "jsonl"
                ),
                **_metadata_entry(
                    temp_paths["validation_parquet"], root, len(validation_rows), "parquet"
                ),
            },
            "audit": audit,
        }
        # Temporary paths are replaced with the published portable paths after hashing.
        for split in ("train", "validation"):
            metadata[split]["jsonl"] = _portable_path(output_paths[f"{split}_jsonl"], root)
            metadata[split]["parquet"] = _portable_path(
                output_paths[f"{split}_parquet"], root
            )
        temp_paths["metadata"].write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        for name in ("train_jsonl", "train_parquet", "validation_jsonl", "validation_parquet"):
            os.replace(temp_paths[name], output_paths[name])
        os.replace(temp_paths["metadata"], output_paths["metadata"])

    return metadata


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    data = root / "data/grpo"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=data / "train.jsonl")
    parser.add_argument("--train-parquet", type=Path, default=data / "train.parquet")
    parser.add_argument("--validation-jsonl", type=Path, default=data / "validation.jsonl")
    parser.add_argument("--validation-parquet", type=Path, default=data / "validation.parquet")
    parser.add_argument("--sft-source", type=Path, default=root / "data/sft_pure_v4/all.jsonl")
    parser.add_argument(
        "--sft-manifest", type=Path, default=root / "data/sft_curriculum/manifest.json"
    )
    parser.add_argument("--evaluation", type=Path, default=root / "data/evaluation/tasks.jsonl")
    parser.add_argument("--metadata", type=Path, default=data / "metadata.json")
    parser.add_argument("--output-dir", type=Path, default=data)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    metadata = prepare_dataset(
        train_jsonl=args.train_jsonl,
        train_parquet=args.train_parquet,
        validation_jsonl=args.validation_jsonl,
        validation_parquet=args.validation_parquet,
        sft_source=args.sft_source,
        sft_manifest=args.sft_manifest,
        evaluation=args.evaluation,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        root=root,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "train_tasks": metadata["train"]["tasks"],
                "validation_tasks": metadata["validation"]["tasks"],
                "removed_train_task_ids": metadata["provenance"]["removed_train_task_ids"],
                "audit": metadata["audit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
