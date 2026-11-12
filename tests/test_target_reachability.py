"""Reachability proofs must not confuse static scores with executable witnesses."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import check_target_reachability as audit  # noqa: E402
from web_agent_site.engine.search import build_index  # noqa: E402


MANIFEST = json.loads((ROOT / "data/environment.json").read_text("utf-8"))
ASIN = "123456789012"


def source_product():
    return {
        "asin": ASIN, "title": "天然乳胶枕头", "shop_name": "枕头店",
        "category": "床上用品›枕头›乳胶枕", "pricing": [99.0], "attribute": ["乳胶"],
        "customization_options": {"颜色": [{"value": "红/白", "price": 99.0}]},
        "instructions": [{"instruction": "要乳胶枕头，红白配色，预算100元以下",
                          "attributes": ["乳胶"], "instruction_options": ["红/白"]}],
    }


def state(page_type, **fields):
    return {"observation_version": "shopping-observation-v2", "page_type": page_type,
            "actions": [], "search_available": False, **fields}


class TargetReachabilityTests(unittest.TestCase):
    def setUp(self):
        self.product, self.goal = audit.task_from_products([source_product()], 0)
        self.static = audit.check_static(self.product, self.goal)
        self.assertEqual(self.static["status"], "pass")
        self.detail = state(
            "product_detail", product={"asin": ASIN, "title": "天然乳胶枕头", "price": 99},
            actions=["红 | 白", "Buy Now"], available_options={"颜色": ["红 | 白"]},
            selected_options={},
        )

    def terminal(self):
        return {"done": True, "over": True, "reward_valid": True,
                "termination_reason": "gold_purchase", "reward": 1.0,
                "reward_detail": deepcopy(self.static["reward_detail"]),
                "observation_state": state("terminal")}

    def search_result(self, *, target=True, next_page=False):
        products = [{"asin": ASIN, "rank": 1, "title": "天然乳胶枕头"}] if target else []
        actions = ([ASIN] if target else []) + (["Next >"] if next_page else [])
        return {"done": False, "over": False,
                "observation_state": state("search_results", products=products, actions=actions)}

    def replay(self, responses, *, initial_override=None, max_steps=35):
        initial = {"env_idx": 7, "instruction": self.goal["instruction_text"],
                   "environment_version": MANIFEST["environment_version"],
                   "_trace_target": {"asin": ASIN, "options": self.goal["goal_options"]},
                   "observation_state": state("search_home", search_available=True)}
        initial.update(initial_override or {})
        calls = []
        pending = iter(responses)

        def transport(endpoint, payload, timeout):
            calls.append(payload)
            if payload["action"] == "reset":
                self.assertTrue(payload["include_trace_target"])
                return {"result": initial}
            if payload["action"] == "release_one":
                return {"result": {"released": True}}
            self.assertEqual(payload["env_idx"], 7)
            response = next(pending)
            if isinstance(response, Exception):
                raise response
            return {"result": response}

        report = audit.replay_query(
            0, self.goal, self.static["selected_options"], "乳胶枕头",
            base_url="http://fixture.invalid", max_steps=max_steps, manifest=MANIFEST,
            env_factory=lambda **kw: audit.ShopAgentEnv(transport=transport, **kw),
        )
        self.assertEqual(calls[-1]["action"], "release_one")
        return report, calls

    def test_legal_witness_selects_visible_normalized_option_and_releases(self):
        selected = deepcopy(self.detail)
        selected["selected_options"] = {"颜色": "红 | 白"}
        report, calls = self.replay([
            self.search_result(), {"observation_state": self.detail},
            {"observation_state": selected}, self.terminal(),
        ])
        self.assertEqual(report["status"], "reachable")
        self.assertEqual(report["environment_steps"], 4)
        self.assertEqual([row["response"] for row in calls if row["action"] == "interact"],
                         ["search[乳胶枕头]", f"click[{ASIN}]", "click[红 | 白]", "click[Buy Now]"])

    def test_missing_target_never_clicks_hidden_asin(self):
        report, calls = self.replay([self.search_result(target=False)])
        self.assertEqual(report["status"], "not_proven")
        self.assertEqual(report["reason"], "target_not_found_by_this_query")
        self.assertEqual(len(calls), 3)  # reset, search, release

    def test_loop_terminal_and_step_cap_are_not_impossibility_proofs(self):
        for responses, limit, expected in [
            ([{"done": True, "over": True, "termination_reason": "repeat_loop",
               "observation_state": state("terminal")}], 35, "repeat_loop"),
            ([self.search_result()], 1, "audit_step_limit"),
            ([{"done": False, "over": True, "observation_state": state("terminal")}],
             35, "environment_over"),
        ]:
            with self.subTest(expected=expected):
                report, _ = self.replay(responses, max_steps=limit)
                self.assertEqual(report["status"], "not_proven")
                self.assertEqual(report["reason"], expected)

    def test_guard_refuses_an_option_missing_from_clickable_buttons(self):
        detail = deepcopy(self.detail)
        detail["actions"] = ["Buy Now"]
        report, _ = self.replay([self.search_result(), {"observation_state": detail}])
        self.assertEqual(report["status"], "not_proven")
        self.assertTrue(report["reason"].startswith("action_guard:"))

    def test_server_mismatch_and_transport_failure_are_errors_and_release(self):
        for override, responses in [
            ({"environment_version": "old"}, []),
            ({"_trace_target": {"asin": "999999999999", "options": []}}, []),
            ({}, [OSError("connection lost")]),
        ]:
            with self.subTest(override=override):
                report, _ = self.replay(responses, initial_override=override)
                self.assertEqual(report["status"], "error")

    def test_strict_gold_rejects_incomplete_terminal(self):
        self.assertTrue(audit.strict_gold(self.terminal()))
        for field in ["done", "over", "reward_valid"]:
            terminal = self.terminal()
            terminal[field] = False
            self.assertFalse(audit.strict_gold(terminal), field)
        terminal = self.terminal()
        del terminal["reward_detail"]["purchase_success"]
        with self.assertRaises(ValueError):
            audit.strict_gold(terminal)

    def test_task_id_uses_filtered_goal_order(self):
        invalid = source_product()
        invalid["asin"] = "nan"
        empty = source_product()
        empty["asin"] = "999999999999"
        empty["instructions"][0]["attributes"] = []
        second = source_product()
        second["asin"] = "888888888888"
        _, goal = audit.task_from_products(
            [invalid, empty, source_product(), source_product(), second], 1
        )
        self.assertEqual(goal["asin"], second["asin"])

    def test_static_budget_failure_is_not_a_global_unreachable_verdict(self):
        goal = deepcopy(self.goal)
        goal["price_upper"] = 50
        result = audit.check_static(self.product, goal)
        self.assertEqual(result["status"], "not_proven")
        self.assertEqual(result["reward_detail"]["reward_type"], "wrong_purchase")

    def test_search_checks_real_bm25_rank_and_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products.sqlite3"
            build_index([self.product], path,
                        product_data_sha256=MANIFEST["product_data_sha256"])
            result = audit.check_search(path, ASIN, ["乳胶枕头", "nomatchingword"], MANIFEST)
            self.assertTrue(result["target_in_index"])
            self.assertEqual(result["queries"][0]["target_rank"], 1)
            self.assertEqual(result["queries"][0]["target_page"], 1)
            self.assertIsNone(result["queries"][1]["target_rank"])
            changed = deepcopy(MANIFEST)
            changed["product_data_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                audit.check_search(path, ASIN, ["乳胶枕头"], changed)

    def test_reject_internal_search_shortcuts(self):
        for query in ["<a>123456789012", "<q>乳胶枕头", "x]\nclick[Buy Now]", ""]:
            with self.subTest(query=query), self.assertRaises(ValueError):
                audit.normal_query(query)

    def test_static_cli_cannot_claim_reachable(self):
        import argparse
        raw = json.dumps([source_product()], ensure_ascii=False).encode()
        manifest = deepcopy(MANIFEST)
        manifest["product_data_sha256"] = audit.hashlib.sha256(raw).hexdigest()
        args = argparse.Namespace(task_id=0, max_steps=35, expected_asin=None,
                                  index=None, replay=False)
        with patch.object(audit, "validate_manifest", return_value=manifest), \
                patch.object(audit.gzip, "open") as stream:
            stream.return_value.__enter__.return_value.read.return_value = raw
            report = audit.audit(args)
        self.assertEqual(report["status"], "static_pass")
        self.assertIsNone(report["reachable"])
        self.assertFalse(report["model_context_budget_checked"])


if __name__ == "__main__":
    unittest.main()
