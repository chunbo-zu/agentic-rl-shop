#!/usr/bin/env python3
"""Audit one known task without an LLM; a legal successful replay is the witness.

This is a private, target-aware diagnostic, not a blind policy evaluation. Static
scoring and bounded search failures never establish global unreachability.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments/ShopSimulator/shop_env"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SHOP_ENV))

from shopping_grpo.environment.actions import action_reject_reason, product_ids  # noqa: E402
from shopping_grpo.environment.client import ShopAgentEnv  # noqa: E402
from shopping_grpo.environment.manifest import validate_manifest  # noqa: E402
from shopping_grpo.environment.observation import render_structured_observation  # noqa: E402
from shopping_grpo.environment.tools import tool_call_to_action  # noqa: E402
from shopping_grpo.evaluation.artifacts import write_json_atomic  # noqa: E402
from shopping_grpo.training.grpo.adapter.runtime import validate_reward  # noqa: E402
from web_agent_site.engine.constraints import explicit_budget_from_instruction  # noqa: E402
from web_agent_site.engine.reward import evaluate_purchase  # noqa: E402
from web_agent_site.engine.reward_features import (  # noqa: E402
    canonicalize_option_axis,
    compile_reward_features,
    normalize_option_text,
)
from web_agent_site.engine.search import MultiFieldBM25Searcher  # noqa: E402
from web_agent_site.engine.variant_price import (  # noqa: E402
    candidate_options_for_evaluation,
    resolve_variant_price,
)


def task_from_products(products, task_id):
    """Follow load_products/get_existed_goals ordering, not raw array indexing."""
    if task_id < 0:
        raise ValueError("task_id must be nonnegative")
    seen = set()
    current = 0
    for raw in products:
        asin = raw["asin"]
        if asin == "nan" or len(asin) > 20 or asin in seen:
            continue
        seen.add(asin)
        for instruction in raw.get("instructions", []):
            if not instruction.get("attributes"):
                continue
            if current == task_id:
                # Only fields consumed by Reward v3 need runtime enrichment.
                # load_products removes brand; Reward then falls back to shop_name.
                product = deepcopy(raw)
                product.pop("brand", None)
                bullets = product.get("small_description", "")
                product.update(
                    Title=product["title"],
                    Description=product.get("full_description", ""),
                    BulletPoints=bullets if isinstance(bullets, list) else [bullets],
                    Attributes=product.get("attribute", []),
                    pricing=(product.get("pricing") or [100.0])[:2],
                )
                goal = {
                    "asin": asin,
                    "category": product["category"],
                    "instruction_text": instruction["instruction"],
                    "price_upper": explicit_budget_from_instruction(instruction["instruction"]),
                    "goal_options": instruction["instruction_options"],
                }
                goal.update(compile_reward_features(instruction, product))
                return product, goal
            current += 1
    raise ValueError(f"task_id {task_id} is outside the {current} available tasks")


def check_static(product, goal):
    selected, options = candidate_options_for_evaluation(product, goal["required_options_by_key"])
    price = resolve_variant_price(product, selected)
    detail = evaluate_purchase(product, goal, selected_options=selected, price_resolution=price)
    passed = (
        not goal["unresolved_option_requirements"]
        and options["status"] == "pass"
        and price["status"] == "pass"
        and detail.reward_valid
        and detail.reward_type == "gold_purchase"
    )
    return {
        "status": "pass" if passed else "not_proven",
        "selected_options": selected,
        "option_resolution": options,
        "price_resolution": price,
        "reward_detail": detail.to_dict(),
        "unresolved_option_requirements": goal["unresolved_option_requirements"],
        "variant_search_exhaustive": False,
    }


def normal_query(value):
    value = value.strip()
    # The simulator has internal <a>/<q>/<c>/<r> search shortcuts. Do not let an
    # oracle ID lookup masquerade as retrieval through ordinary search tools.
    if not value or any(char in value for char in "<>[]\\\n\r"):
        raise ValueError("query must be ordinary search text, without shortcuts or action syntax")
    return value


def check_search(index_path, target_asin, queries, manifest):
    searcher = MultiFieldBM25Searcher(
        index_path, expected_product_sha256=manifest["product_data_sha256"]
    )
    try:
        if searcher.manifest["field_weights"] != manifest["search"]["field_weights"]:
            raise ValueError("search index weights differ from data/environment.json")
        rows = []
        for query in queries:
            hits = searcher.search(normal_query(query), k=manifest["search"]["top_k"])
            rank = next((hit.rank for hit in hits if hit.asin == target_asin), None)
            rows.append({
                "query": query,
                "returned_count": len(hits),
                "target_rank": rank,
                "target_page": (rank - 1) // manifest["search"]["page_size"] + 1 if rank else None,
            })
        return {
            "status": "pass" if any(row["target_rank"] for row in rows) else "not_proven",
            "target_in_index": searcher.contains_asin(target_asin),
            "queries": rows,
        }
    finally:
        searcher.close()


class PathNotProven(Exception):
    """A bounded legal path failed; this is not a proof of impossibility."""


def normalized_options(options):
    return {
        canonicalize_option_axis(axis): normalize_option_text(value)
        for axis, value in options.items()
    }


def strict_gold(terminal):
    detail = validate_reward(terminal.get("reward_detail"))
    return (
        terminal.get("done") is True
        and terminal.get("over") is True
        and terminal.get("reward_valid") is True
        and terminal.get("termination_reason") == "gold_purchase"
        and detail["reward_type"] == "gold_purchase"
        and detail["reward_valid"] is True
        and detail["purchase_success"] is True
        and terminal["reward_detail"].get("target_asin_match") is True
    )


def replay_query(task_id, goal, selected, query, *, base_url, max_steps, manifest,
                 env_factory=ShopAgentEnv):
    """Search, page, open, select, buy; never click an unseen ASIN or value."""
    query = normal_query(query)
    if not 1 <= max_steps <= manifest["max_steps"]:
        raise ValueError("max_steps must be within the frozen environment limit")
    attempt = {"query": query, "status": "not_proven", "steps": []}
    result = {}
    try:
        with env_factory(base_url=base_url) as env:
            env.include_trace_target = True
            result = env.reset(task_id)
            attempt["initial_observation_state"] = result.get("observation_state")
            if result.get("environment_version") != manifest["environment_version"]:
                raise ValueError("server environment version mismatch")
            trace = result.get("_trace_target") or {}
            if (trace.get("asin") != goal["asin"]
                    or trace.get("options") != goal["goal_options"]
                    or result.get("instruction") != goal["instruction_text"]):
                raise ValueError("server task differs from the frozen local task")

            def execute(name, arguments):
                nonlocal result
                if result.get("done") or result.get("over"):
                    raise PathNotProven(result.get("termination_reason") or "environment_over")
                if len(attempt["steps"]) >= max_steps:
                    raise PathNotProven("audit_step_limit")
                observation = render_structured_observation(result.get("observation_state"))
                rejected = action_reject_reason(name, arguments, observation)
                if rejected:
                    raise PathNotProven(f"action_guard:{rejected}")
                action = tool_call_to_action(name, arguments)
                result = env.step(action)
                attempt["steps"].append({
                    "tool": name, "arguments": arguments, "action": action,
                    "observation_state": result.get("observation_state"),
                    "done": result.get("done", False), "over": result.get("over", False),
                })
                # Also validate the final observation's public-state contract.
                render_structured_observation(result.get("observation_state"))

            execute("search_products", {"query": query})
            while not result.get("done") and not result.get("over"):
                observation = render_structured_observation(result["observation_state"])
                if goal["asin"] in product_ids(observation):
                    break
                if action_reject_reason("next_page", {}, observation):
                    raise PathNotProven("target_not_found_by_this_query")
                execute("next_page", {})
            execute("open_product", {"asin": goal["asin"]})
            state = result["observation_state"]
            if (state.get("page_type") != "product_detail"
                    or state.get("product", {}).get("asin") != goal["asin"]):
                raise ValueError("open_product did not open the expected target")
            for axis, wanted in selected.items():
                state = result["observation_state"]
                axis_key = canonicalize_option_axis(axis)
                if normalized_options(state.get("selected_options", {})).get(axis_key) == (
                    normalize_option_text(wanted)
                ):
                    continue
                choices = [
                    value
                    for visible_axis, values in state.get("available_options", {}).items()
                    if canonicalize_option_axis(visible_axis) == axis_key
                    for value in values
                    if normalize_option_text(value) == normalize_option_text(wanted)
                ]
                if len(choices) != 1:
                    raise PathNotProven(f"required_option_not_uniquely_visible:{axis}")
                execute("select_option", {"value": choices[0]})
            if normalized_options(result["observation_state"].get("selected_options", {})) != (
                normalized_options(selected)
            ):
                raise PathNotProven("selected_options_differ_from_planned_variant")
            execute("buy_now", {})
            if strict_gold(result):
                attempt.update(status="reachable", reason="legal_gold_purchase_witness")
            else:
                attempt["reason"] = result.get("termination_reason") or "not_strict_gold"
    except PathNotProven as exc:
        attempt["reason"] = str(exc)
    except Exception as exc:
        attempt.update(status="error", reason=f"{type(exc).__name__}: {exc}")
    if result.get("done") or result.get("over"):
        attempt["terminal_result"] = result
    attempt["environment_steps"] = len(attempt["steps"])
    return attempt


def audit(args):
    manifest = validate_manifest(json.loads((ROOT / "data/environment.json").read_text("utf-8")))
    if not 1 <= args.max_steps <= manifest["max_steps"]:
        raise ValueError(f"--max-steps must be between 1 and {manifest['max_steps']}")
    with gzip.open(SHOP_ENV / "data/fine_items_eval_train_all.json.gz", "rb") as stream:
        data = stream.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest["product_data_sha256"]:
        raise ValueError("frozen product data SHA-256 mismatch")
    product, goal = task_from_products(json.loads(data), args.task_id)
    if args.expected_asin and args.expected_asin != goal["asin"]:
        raise ValueError("--expected-asin differs from this task's target")
    static = check_static(product, goal)
    report = {
        "audit_version": "shopping-target-reachability-v1",
        "scope": "private_target_aware_environment_audit",
        "task_id": args.task_id,
        "target_asin": goal["asin"],
        "instruction": goal["instruction_text"],
        "product_data_sha256": digest,
        "environment_version": manifest["environment_version"],
        "max_steps_per_attempt": args.max_steps,
        "reachable": None,
        "status": "static_pass" if static["status"] == "pass" else "not_proven",
        "static": static,
        "retrieval": {"status": "not_checked"},
        "replay": {"status": "not_checked", "attempts": []},
        "model_context_budget_checked": False,
    }
    if args.index:
        report["retrieval"] = check_search(args.index, goal["asin"], args.query, manifest)
        report["status"] = (
            "retrieval_pass"
            if static["status"] == report["retrieval"]["status"] == "pass" else "not_proven"
        )
    if args.replay:
        replay = report["replay"]
        replay["status"] = "not_proven"
        if static["status"] != "pass":
            replay["reason"] = "no_statically_valid_variant_witness"
        else:
            for query in args.query:
                attempt = replay_query(
                    args.task_id, goal, static["selected_options"], query,
                    base_url=args.base_url, max_steps=args.max_steps, manifest=manifest,
                )
                replay["attempts"].append(attempt)
                if attempt["status"] in {"reachable", "error"}:
                    replay["status"] = attempt["status"]
                    break
        report["status"] = replay["status"]
        if replay["status"] == "reachable":
            report["reachable"] = True
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--expected-asin", help="Optional assertion about this task's target")
    parser.add_argument("--query", action="append", type=normal_query, default=[])
    parser.add_argument("--index", type=Path, help="Existing frozen BM25 index to inspect")
    parser.add_argument("--replay", action="store_true", help="Execute a witness in ShopSimulator")
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--output", type=Path, help="Private JSON report; default is stdout")
    parser.add_argument("--force", action="store_true", help="Replace an existing output report")
    args = parser.parse_args(argv)
    if (args.index or args.replay) and not args.query:
        parser.error("--index and --replay require at least one --query")
    if args.output and args.output.exists() and not args.force:
        parser.error("output exists; use a different path or --force")
    try:
        report = audit(args)
    except Exception as exc:
        report = {"task_id": args.task_id, "status": "error", "reachable": None,
                  "reason": f"{type(exc).__name__}: {exc}"}
    if args.output:
        write_json_atomic(args.output, report, force=args.force)
        print(json.dumps({"status": report["status"], "reachable": report["reachable"],
                          "output": str(args.output)}, ensure_ascii=False))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if report["status"] == "error" else 1 if report["status"] == "not_proven" else 0


if __name__ == "__main__":
    raise SystemExit(main())
