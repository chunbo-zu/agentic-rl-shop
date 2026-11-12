"""Public CPU and parameterized GRPO entry-point tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.train_grpo import build_command, main as train_grpo_main, parse_args
from shopping_grpo.cli import main as cli_main
from shopping_grpo.smoke import run_cpu_smoke


class PublicEntrypointTest(unittest.TestCase):
    def test_cpu_smoke_covers_public_contracts(self):
        result = run_cpu_smoke()

        self.assertEqual(
            result["checks"],
            [
                "action_schema",
                "trajectory_normalization",
                "reward_sample",
                "sft_label_mask",
                "dynamic_sampling_grouping",
            ],
        )

    def test_offline_example_cli_runs_without_models_or_environment(self):
        root = Path(__file__).resolve().parents[1]
        with patch.object(
            sys,
            "argv",
            [
                "shopping-grpo",
                "evaluate",
                str(root / "examples/trajectories.jsonl"),
            ],
        ), patch("builtins.print") as output:
            cli_main()

        summary = json.loads(output.call_args.args[0])
        self.assertEqual(summary["trajectory_count"], 3)
        self.assertEqual(summary["strict_gold_success_count"], 1)

    def test_public_grpo_launcher_accepts_sharded_weights_and_console(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors.index.json").write_text(
                "{}",
                encoding="utf-8",
            )
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            output = temporary / "output"
            with patch.object(
                sys,
                "argv",
                [
                    "train_grpo.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--output",
                    str(output),
                    "--config",
                    str(root / "configs/grpo.yaml"),
                    "--logger",
                    "console",
                    "--dry-run",
                ],
            ):
                args = parse_args()
            with patch.dict("os.environ", {"VLLM_RPC_BASE_PATH": ""}):
                command, environment = build_command(args)

        self.assertIn("verl.trainer.main_ppo", command)
        self.assertEqual(environment["GRPO_MODEL_PATH"], str(model.resolve()))
        self.assertEqual(environment["GRPO_TRAIN_FILE"], str(train.resolve()))
        self.assertEqual(environment["GRPO_VAL_FILE"], str(validation.resolve()))
        self.assertEqual(
            environment["SHOPPING_GRPO_DIAGNOSTICS_PATH"],
            str(output.resolve() / "training_diagnostics.jsonl"),
        )
        rpc_socket_probe = (
            Path(environment["VLLM_RPC_BASE_PATH"]) / ("0" * 36)
        )
        self.assertLessEqual(len(str(rpc_socket_probe).encode()), 107)
        self.assertEqual(rpc_socket_probe.parent.parent, Path("/tmp"))
        python_mp_probe = (
            Path(environment["TMPDIR"]) / "pymp-12345678" / "listener-12345678"
        )
        self.assertLessEqual(len(str(python_mp_probe).encode()), 107)
        self.assertEqual(python_mp_probe.parent.parent.parent, Path("/tmp"))
        self.assertEqual(environment["SHOPPING_PYTHON_TMPDIR"], environment["TMPDIR"])
        self.assertIn("trainer.logger=[console]", command)

    def test_public_grpo_launcher_selects_fully_async_entrypoint(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"example")
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            with patch.object(
                sys,
                "argv",
                [
                    "train_grpo.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--output",
                    str(temporary / "output"),
                    "--config",
                    str(root / "configs/sao_fully_async.yaml"),
                    "--dry-run",
                ],
            ):
                args = parse_args()
            command, environment = build_command(args)

        self.assertIn(
            "shopping_grpo.training.grpo.fully_async_main",
            command,
        )
        self.assertEqual(environment["GRPO_FULLY_ASYNC"], "1")

    def test_public_grpo_launcher_rejects_long_vllm_rpc_path(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"example")
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            with patch.object(
                sys,
                "argv",
                [
                    "train_grpo.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--output",
                    str(temporary / "output"),
                    "--config",
                    str(root / "configs/grpo.yaml"),
                    "--dry-run",
                ],
            ), patch.dict(
                "os.environ",
                {"VLLM_RPC_BASE_PATH": "/tmp/" + "x" * 100},
            ):
                args = parse_args()
                with self.assertRaisesRegex(SystemExit, "too long"):
                    build_command(args)

    def test_public_grpo_launcher_rejects_long_python_mp_tmpdir(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"example")
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            with patch.object(
                sys,
                "argv",
                [
                    "train_grpo.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--output",
                    str(temporary / "output"),
                    "--config",
                    str(root / "configs/grpo.yaml"),
                    "--dry-run",
                ],
            ), patch.dict(
                "os.environ",
                {"SHOPPING_PYTHON_TMPDIR": "/tmp/" + "x" * 100},
            ):
                args = parse_args()
                with self.assertRaisesRegex(SystemExit, "too long"):
                    build_command(args)

    def test_public_grpo_launcher_runs_preflight_before_training(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"example")
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            output = temporary / "output"
            argv = [
                "train_grpo.py",
                "--model",
                str(model),
                "--train-data",
                str(train),
                "--val-data",
                str(validation),
                "--output",
                str(output),
                "--config",
                str(root / "configs/grpo.yaml"),
                "--experiment-name",
                "scope-regression",
                "--",
                "trainer.total_training_steps=20",
            ]
            with patch.object(sys, "argv", argv), patch.dict(
                "os.environ", {"VLLM_RPC_BASE_PATH": ""}
            ), patch(
                "scripts.train_grpo.subprocess.call", side_effect=(0, 0)
            ) as subprocess_call, self.assertRaises(SystemExit) as exit_status:
                train_grpo_main()

        self.assertEqual(exit_status.exception.code, 0)
        self.assertEqual(subprocess_call.call_count, 2)
        preflight = subprocess_call.call_args_list[0].args[0]
        training = subprocess_call.call_args_list[1].args[0]
        expected_overrides = [
            "trainer.logger=[console]",
            "trainer.experiment_name=scope-regression",
            "trainer.total_training_steps=20",
        ]
        self.assertTrue(str(preflight[1]).endswith("scripts/check_grpo_runtime.py"))
        self.assertEqual(preflight[2:], expected_overrides)
        self.assertEqual(training[-3:], expected_overrides)


if __name__ == "__main__":
    unittest.main()
