#!/usr/bin/env python3
"""依次评测 SFT 各阶段和一个 GRPO 模型。

这个脚本只负责评测编排：ShopSimulator 启动一次，vLLM 在不同模型之间串行
重启；每个模型仍调用 ``evaluate_shop_benchmark.py``，因此使用的是同一套
Reward v3、工具守卫和上下文预算。评测器本身支持断点续跑，本脚本也会在每个
run 目录写入 ``run.json``，防止不小心把不同模型的轨迹混在一起。

示例（模型已导出为 Hugging Face 目录）：

    .venv/bin/python scripts/evaluate_sft_grpo.py \
      --sft-root /data/shopping-rl/models/sft-curriculum \
      --grpo-model /home/ml-user/workdir/zcb26/shopping-rl/outputs/models \
      --benchmark data/evaluation/tasks.jsonl

如果只有 veRL actor checkpoint，可显式指定要评测的 checkpoint，并让脚本先导出：

    .venv/bin/python scripts/evaluate_sft_grpo.py \
      --sft-root /home/ml-user/workdir/zcb26/shopping-rl/outputs/models/sft-curriculum \
      --grpo-actor /home/ml-user/workdir/zcb26/shopping-rl/outputs/models/grpo-qwen35-2b-stage-c-main/global_step_50/actor \
      --grpo-export /home/ml-user/workdir/zcb26/shopping-rl/outputs/models/grpo-50step-merged

默认不会自动选择 GRPO 的 latest checkpoint；checkpoint 应先依据 GRPO validation
指标选择。正式 Final-200 也应在 checkpoint 冻结后再运行。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
STAGES = ("a", "b", "c")


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: Path


def _parse_model(value: str) -> ModelSpec:
    """Parse one ``LABEL=PATH`` argument without interpreting shell syntax."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model 格式必须是 LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not LABEL_RE.fullmatch(label):
        raise argparse.ArgumentTypeError(
            f"模型标签 {label!r} 非法；只允许字母、数字、点、下划线和短横线"
        )
    if not raw_path:
        raise argparse.ArgumentTypeError("--model 的 PATH 不能为空")
    return ModelSpec(label=label, path=Path(raw_path).expanduser())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="串行评测 SFT stage-a/b/c 和 GRPO 模型，并生成对比表"
    )
    parser.add_argument(
        "--model",
        action="append",
        type=_parse_model,
        metavar="LABEL=PATH",
        help="自定义模型列表；传入后不再使用 --sft-root/--grpo-model",
    )
    parser.add_argument(
        "--sft-root",
        type=Path,
        default=ROOT / "outputs/models/sft-curriculum",
        help="SFT 输出根目录，目录下应有 stage-{a,b,c}/merged",
    )
    parser.add_argument(
        "--stages",
        default="a,b,c",
        help="要评测的 SFT 阶段，例如 a,b 或 c（默认 a,b,c）",
    )
    grpo_group = parser.add_mutually_exclusive_group()
    grpo_group.add_argument("--grpo-model", type=Path, help="已导出的 GRPO Hugging Face 模型目录")
    grpo_group.add_argument(
        "--grpo-actor",
        type=Path,
        help="veRL global_step_* /actor 目录；需同时传 --grpo-export",
    )
    parser.add_argument(
        "--grpo-export",
        type=Path,
        help="GRPO actor 导出目录（仅 --grpo-actor 使用；目录不存在时执行 export_grpo.sh）",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data/evaluation/tasks.jsonl",
        help="评测任务 JSONL；默认是冻结的 Final-200 Clean",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/evaluation/sft-grpo",
        help="每个模型的轨迹、summary 和 report 输出目录",
    )
    parser.add_argument(
        "--base-url", default=os.environ.get("SHOPSIM_BASE_URL", "http://127.0.0.1:5700")
    )
    parser.add_argument(
        "--llm-base-url", default=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    )
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", "EMPTY"))
    parser.add_argument(
        "--served-model-name",
        default=os.environ.get("EVAL_SERVED_MODEL_NAME", "shopping-agent-eval"),
        help="每轮 vLLM 使用的 served model 名称",
    )
    parser.add_argument(
        "--python", type=Path, default=ROOT / ".venv/bin/python", help="运行评测器的 Python"
    )
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=180, help="单次模型请求超时（秒）")
    parser.add_argument("--context-window", type=int, default=24576)
    parser.add_argument("--context-safety-margin", type=int, default=512)
    parser.add_argument("--observation-token-budget", type=int, default=1536)
    parser.add_argument("--observation-detail-token-budget", type=int, default=4096)
    parser.add_argument("--observation-generic-token-budget", type=int, default=768)
    parser.add_argument("--observation-search-top-k", type=int, default=20)
    parser.add_argument("--context-compaction", action="store_true")
    parser.add_argument(
        "--ready-timeout",
        type=int,
        default=900,
        help="等待 ShopSimulator/vLLM 就绪的最长时间（秒）",
    )
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument(
        "--no-start-services", action="store_true", help="使用已有 ShopSimulator 和 vLLM"
    )
    parser.add_argument("--no-start-shopsim", action="store_true", help="不自动启动 ShopSimulator")
    parser.add_argument("--no-start-llm", action="store_true", help="不自动启动 vLLM（多模型时通常不可用）")
    parser.add_argument("--dry-run", action="store_true", help="只检查并打印计划，不启动服务或调用模型")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="单个模型失败后继续评测其余模型；默认立即退出",
    )
    return parser.parse_args(argv)


def _get_json(url: str, timeout: float = 3.0):
    request = Request(url, headers={"User-Agent": "shopping-grpo-eval/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_ready(url: str, timeout: float = 3.0) -> bool:
    """Return true once an HTTP server is listening, even if this path is absent.

    ShopSimulator intentionally exposes only ``POST /api/shop_agent`` and returns
    404 for ``GET /``.  An ``HTTPError`` therefore proves that Flask is ready to
    accept requests; connection and timeout errors still mean it is unavailable.
    """
    request = Request(url, headers={"User-Agent": "shopping-grpo-eval/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return 100 <= int(response.status) < 600
    except HTTPError:
        return True
    except (URLError, TimeoutError, OSError):
        return False


def _is_ready(url: str) -> bool:
    try:
        _get_json(url)
        return True
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        return False


def _wait_ready(
    url: str,
    label: str,
    timeout: int,
    poll_interval: float,
    process=None,
    json_endpoint: bool = False,
) -> None:
    ready_check = _is_ready if json_endpoint else _http_ready
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready_check(url):
            print(f"[{label}] ready: {url}", flush=True)
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"{label} 进程提前退出，exit_code={process.returncode}；请查看日志")
        time.sleep(poll_interval)
    raise TimeoutError(f"{label} 在 {timeout} 秒内未就绪：{url}")


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _assert_model_available(base_url: str, model_name: str) -> None:
    try:
        data = _get_json(_models_url(base_url), timeout=5)
    except Exception as exc:  # noqa: BLE001 - converted to an actionable message below
        raise RuntimeError(f"无法读取 vLLM 模型列表 {_models_url(base_url)}: {exc}") from exc
    ids = {str(item.get("id")) for item in data.get("data", []) if isinstance(item, dict)}
    if model_name not in ids:
        raise RuntimeError(f"vLLM 已响应，但没有 served model {model_name!r}；当前模型：{sorted(ids)}")


def _stop_process(process: subprocess.Popen | None, label: str) -> None:
    if process is None or process.poll() is not None:
        return
    print(f"[{label}] stopping process {process.pid}", flush=True)
    try:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        print(f"[{label}] did not stop after 45s; killing", flush=True)
        process.kill()
        process.wait(timeout=15)


def _validate_stage_list(raw: str) -> tuple[str, ...]:
    stages = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    if not stages or any(item not in STAGES for item in stages) or len(set(stages)) != len(stages):
        raise ValueError("--stages 必须是 a、b、c 的逗号分隔列表，且不能重复")
    return stages


def _build_specs(args: argparse.Namespace, stages: tuple[str, ...]) -> list[ModelSpec]:
    if args.model:
        if args.grpo_model or args.grpo_actor or args.grpo_export:
            raise ValueError("传入 --model 后不能再传 --grpo-model/--grpo-actor/--grpo-export")
        labels = [item.label for item in args.model]
        if len(labels) != len(set(labels)):
            raise ValueError("--model 的 LABEL 不能重复")
        return args.model

    if args.grpo_export and not args.grpo_actor:
        raise ValueError("--grpo-export 只能与 --grpo-actor 一起使用")

    specs = [
        ModelSpec(f"sft-stage-{stage}", args.sft_root / f"stage-{stage}" / "merged")
        for stage in stages
    ]
    if args.grpo_actor:
        if not args.grpo_export:
            raise ValueError("--grpo-actor 必须同时传 --grpo-export，明确指定导出位置")
        specs.append(ModelSpec("grpo", args.grpo_export))
    elif args.grpo_model:
        specs.append(ModelSpec("grpo", args.grpo_model))
    else:
        raise ValueError("默认模式需要 --grpo-model 或 --grpo-actor；若只评测自定义模型请使用 --model LABEL=PATH")
    if len({item.label for item in specs}) != len(specs):
        raise ValueError("模型标签重复")
    return specs


def _check_model_dir(path: Path, label: str, dry_run: bool) -> None:
    config = path / "config.json"
    if not config.is_file():
        if dry_run:
            print(f"[dry-run] 警告：{label} 缺少 {config}（实际运行会失败）")
            return
        raise FileNotFoundError(f"{label} 不是可直接 serve 的 Hugging Face 模型目录：缺少 {config}")


def _metadata_for(args: argparse.Namespace, spec: ModelSpec) -> dict:
    return {
        "label": spec.label,
        "model_path": str(spec.path.resolve()),
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "served_model_name": args.served_model_name,
        "max_steps": args.max_steps,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "context_window": args.context_window,
        "context_safety_margin": args.context_safety_margin,
        "observation_token_budget": args.observation_token_budget,
        "context_compaction": args.context_compaction,
    }


def _prepare_run_dir(args: argparse.Namespace, spec: ModelSpec) -> Path:
    run_dir = args.output_root / spec.label
    metadata_path = run_dir / "run.json"
    expected = _metadata_for(args, spec)
    if metadata_path.exists():
        try:
            old = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"无法读取已有元数据 {metadata_path}：{exc}") from exc
        compared = {key: old.get(key) for key in expected}
        if compared != expected:
            raise RuntimeError(
                f"{run_dir} 已属于另一个模型或 benchmark；请换 --output-root，避免混用断点轨迹"
            )
    elif any((run_dir / name).exists() for name in ("trajectories.jsonl", "summary.json")):
        raise RuntimeError(
            f"{run_dir} 已有评测文件但缺少 run.json；请换 --output-root，避免混用未知来源轨迹"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def _evaluator_command(args: argparse.Namespace, spec: ModelSpec, run_dir: Path) -> list[str]:
    return [
        str(args.python),
        str(ROOT / "scripts/evaluate_shop_benchmark.py"),
        "--benchmark",
        str(args.benchmark),
        "--output",
        str(run_dir / "trajectories.jsonl"),
        "--summary",
        str(run_dir / "summary.json"),
        "--base-url",
        args.base_url,
        "--model",
        args.served_model_name,
        "--llm-base-url",
        args.llm_base_url,
        "--api-key",
        args.api_key,
        "--max-steps",
        str(args.max_steps),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--timeout",
        str(args.timeout),
        "--context-window",
        str(args.context_window),
        "--context-safety-margin",
        str(args.context_safety_margin),
        "--observation-token-budget",
        str(args.observation_token_budget),
        "--observation-detail-token-budget",
        str(args.observation_detail_token_budget),
        "--observation-generic-token-budget",
        str(args.observation_generic_token_budget),
        "--observation-search-top-k",
        str(args.observation_search_top_k),
    ] + (["--context-compaction"] if args.context_compaction else [])


def _write_comparison(args: argparse.Namespace, specs: list[ModelSpec]) -> None:
    runs = []
    for spec in specs:
        summary_path = args.output_root / spec.label / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _annotate_summary(summary, args, spec)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        runs.append(
            {
                "label": spec.label,
                "model_path": str(spec.path.resolve()),
                "summary": summary,
                "run_dir": str((args.output_root / spec.label).resolve()),
            }
        )
    if not runs:
        raise RuntimeError("没有可汇总的 summary.json")
    comparison = {
        "benchmark": str(args.benchmark.resolve()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": runs,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    columns = (
        ("模型", "label"),
        ("严格成功率", "strict_success_rate"),
        ("严格成功", "strict_successes"),
        ("完成率", "done_rate"),
        ("购买成功率", "purchase_success_rate"),
        ("Reward 有效率", "reward_valid_rate"),
        ("平均 Reward", "mean_final_reward"),
        ("平均加权分", "mean_weighted_score"),
        ("平均步数", "average_steps"),
    )
    lines = [
        "# SFT / GRPO 评测对比",
        "",
        f"Benchmark: `{args.benchmark}`",
        "",
        "| " + " | ".join(name for name, _ in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for item in runs:
        summary = item["summary"]
        values = []
        for name, key in columns:
            if key == "label":
                values.append(item["label"])
            elif key.endswith("_rate"):
                values.append(f"{float(summary.get(key, 0.0)) * 100:.2f}%")
            elif key in {"mean_final_reward", "mean_weighted_score", "average_steps"}:
                values.append(f"{float(summary.get(key, 0.0)):.4f}")
            else:
                values.append(str(summary.get(key, 0)))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "每个模型的详细轨迹报告位于对应子目录的 `report.html`；原始轨迹为 `trajectories.jsonl`。",
            "脚本按 `task_id` 断点续跑，严格成功只认 Reward v3 合法 `gold_purchase`。",
        ]
    )
    (args.output_root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[comparison] {args.output_root / 'comparison.md'}", flush=True)


def _annotate_summary(summary: dict, args: argparse.Namespace, spec: ModelSpec) -> None:
    """Keep reports human-readable while preserving the evaluator's served-model field."""
    protocol = summary.setdefault("protocol", {})
    protocol["model"] = spec.label
    protocol["model_path"] = str(spec.path.resolve())
    protocol["served_model"] = args.served_model_name


def _export_actor(args: argparse.Namespace) -> None:
    if not args.grpo_actor:
        return
    assert args.grpo_export is not None
    config = args.grpo_export / "config.json"
    if config.is_file():
        print(f"[grpo] using existing export: {args.grpo_export}", flush=True)
        return
    args.grpo_export.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "bash",
        str(ROOT / "scripts/export_grpo.sh"),
        str(args.grpo_actor),
        str(args.grpo_export),
    ]
    print("[grpo] exporting actor:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _start_process(command: list[str], log_path: Path, env: dict[str, str]) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    try:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
        handle.close()
        return process
    except Exception:
        handle.close()
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_start_services:
        args.no_start_shopsim = True
        args.no_start_llm = True
    if args.max_steps < 1 or args.max_tokens < 1 or args.ready_timeout < 1:
        raise SystemExit("--max-steps、--max-tokens、--ready-timeout 必须为正数")
    if args.no_start_llm and not args.dry_run:
        # A shared server cannot change weights between labels. Requiring one model
        # avoids silently reporting the same server under four different labels.
        stages = _validate_stage_list(args.stages)
        requested_count = len(args.model) if args.model else len(stages) + 1
        if requested_count > 1:
            raise SystemExit("多模型评测需要脚本管理 vLLM；请去掉 --no-start-llm，或一次只传一个 --model")
    try:
        stages = _validate_stage_list(args.stages)
        specs = _build_specs(args, stages)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.benchmark.is_file():
        raise SystemExit(f"找不到 benchmark：{args.benchmark}")
    if not args.python.is_file() and not args.dry_run:
        raise SystemExit(f"找不到 Python：{args.python}；请先运行 bash scripts/setup.sh")
    if args.grpo_actor and not args.dry_run:
        if not args.grpo_actor.is_dir():
            raise SystemExit(f"找不到 GRPO actor：{args.grpo_actor}")
        _export_actor(args)
    for spec in specs:
        _check_model_dir(spec.path, spec.label, args.dry_run)
    print("评测计划：", flush=True)
    for spec in specs:
        print(f"  {spec.label}: {spec.path}", flush=True)
    print(f"  benchmark: {args.benchmark}", flush=True)
    print(f"  output:    {args.output_root}", flush=True)
    if args.dry_run:
        if args.grpo_actor:
            assert args.grpo_export is not None
            print(
                "[dry-run] bash "
                f"{ROOT / 'scripts/export_grpo.sh'} {args.grpo_actor} {args.grpo_export}",
                flush=True,
            )
        for spec in specs:
            run_dir = args.output_root / spec.label
            print("[dry-run]", " ".join(_evaluator_command(args, spec, run_dir)), flush=True)
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    shop_process = None
    model_process = None
    started_shop = False
    started_llm = False
    failed_labels: list[str] = []
    try:
        if args.no_start_shopsim:
            _wait_ready(args.base_url, "shopsim", args.ready_timeout, args.poll_interval)
        elif _http_ready(args.base_url):
            print(f"[shopsim] using existing service: {args.base_url}", flush=True)
        else:
            shop_log = args.output_root / "shopsim.log"
            shop_process = _start_process(
                ["bash", str(ROOT / "scripts/start_environment.sh")], shop_log, os.environ.copy()
            )
            started_shop = True
            _wait_ready(
                args.base_url, "shopsim", args.ready_timeout, args.poll_interval, shop_process
            )

        if args.no_start_llm:
            _wait_ready(
                _models_url(args.llm_base_url),
                "vllm",
                args.ready_timeout,
                args.poll_interval,
                json_endpoint=True,
            )
            _assert_model_available(args.llm_base_url, args.served_model_name)

        for spec in specs:
            try:
                run_dir = _prepare_run_dir(args, spec)
                if not args.no_start_llm:
                    # Refuse to hijack a server owned by another process. Set LLM_PORT or
                    # stop that server explicitly before launching this orchestrator.
                    if not started_llm and _http_ready(_models_url(args.llm_base_url)):
                        raise RuntimeError(
                            f"vLLM 地址已被占用：{args.llm_base_url}；为避免错评，脚本不会覆盖外部服务，请停止它或更换端口"
                        )
                    env = os.environ.copy()
                    env["SERVED_MODEL_NAME"] = args.served_model_name
                    # serve_model.sh reads LLM_PORT; preserve any other vLLM/CUDA settings.
                    match = re.search(r":(\d+)(?:/v1)?/?$", args.llm_base_url.rstrip("/"))
                    if match:
                        env["LLM_PORT"] = match.group(1)
                    model_log = run_dir / "vllm.log"
                    model_process = _start_process(
                        ["bash", str(ROOT / "scripts/serve_model.sh"), str(spec.path)],
                        model_log,
                        env,
                    )
                    started_llm = True
                    try:
                        _wait_ready(
                            _models_url(args.llm_base_url),
                            f"vllm/{spec.label}",
                            args.ready_timeout,
                            args.poll_interval,
                            model_process,
                            json_endpoint=True,
                        )
                        _assert_model_available(args.llm_base_url, args.served_model_name)
                    except Exception:
                        _stop_process(model_process, f"vllm/{spec.label}")
                        model_process = None
                        raise

                command = _evaluator_command(args, spec, run_dir)
                print(f"[{spec.label}] evaluating", flush=True)
                print(" ".join(command), flush=True)
                try:
                    subprocess.run(command, cwd=ROOT, check=True)
                    summary_path = run_dir / "summary.json"
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    _annotate_summary(summary, args, spec)
                    summary_path.write_text(
                        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    report_command = [
                        str(args.python),
                        str(ROOT / "scripts/build_eval_report.py"),
                        "--run-dir",
                        str(run_dir),
                    ]
                    subprocess.run(report_command, cwd=ROOT, check=True)
                finally:
                    if started_llm:
                        _stop_process(model_process, f"vllm/{spec.label}")
                        model_process = None
                        # A short grace period lets vLLM release its listening socket/GPU
                        # allocations before the next large checkpoint is loaded.
                        time.sleep(min(max(args.poll_interval, 0.2), 5.0))
                print(f"[{spec.label}] done: {run_dir}", flush=True)
            except Exception as exc:
                print(f"[{spec.label}] 评测失败：{exc}", file=sys.stderr, flush=True)
                failed_labels.append(spec.label)
                if not args.continue_on_error:
                    raise
    except Exception as exc:  # noqa: BLE001 - main converts all failures to exit 1
        print(f"评测失败：{exc}", file=sys.stderr, flush=True)
        if not args.continue_on_error:
            return 1
    finally:
        _stop_process(model_process, "vllm")
        if started_shop:
            _stop_process(shop_process, "shopsim")

    try:
        _write_comparison(args, specs)
    except Exception as exc:  # noqa: BLE001
        print(f"生成对比表失败：{exc}", file=sys.stderr, flush=True)
        return 1
    if failed_labels:
        print(f"以下模型未完成：{', '.join(failed_labels)}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("已中断；已有 trajectories.jsonl 可直接重新运行以断点续跑。", file=sys.stderr)
        raise SystemExit(130)
