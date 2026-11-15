from copy import deepcopy
import unittest

from scripts.prepare_framework_pair import paired_budget_projection, select_fixed_case, validate_raw_case


def request(framework, input_tokens, output_tokens):
    return {"framework": framework, "estimated_input_tokens": input_tokens, "reserved_output_tokens": output_tokens,
            "native_wire_sha256": "a" * 64, "estimate_basis": "official_tokenizer_native_request_estimate"}


class FrameworkPairPreparationTests(unittest.TestCase):
    def test_later_arm_cannot_reset_shared_prior_or_first_arm_usage(self):
        plan = [request("langgraph", 2000, 1000), request("langgraph", 3000, 2000),
                request("dsh", 2000, 1000), request("dsh", 3000, 2000)]
        result = paired_budget_projection(plan, prior_known_tokens=4000)
        self.assertEqual(result["reserved_pair_upper_bound"], 20512)
        self.assertFalse(result["all_native_fixtures_fit"])
        self.assertEqual(result["requests"][-1]["shared_reserved_total_after"], 20000)
        self.assertFalse(result["paid_execution_authorized"])

    def test_incomplete_native_estimates_cannot_claim_pair_fit(self):
        result = paired_budget_projection([request("langgraph", 100, 100), request("langgraph", 100, 100)])
        self.assertFalse(result["complete_two_step_pair"])
        self.assertFalse(result["all_native_fixtures_fit"])

    def test_heuristic_or_unbound_estimate_rejected(self):
        bad = request("langgraph", 100, 100)
        bad["estimate_basis"] = "characters_divided_by_four"
        with self.assertRaisesRegex(ValueError, "native_serialized_estimate_required"):
            paired_budget_projection([bad])
        bad = request("langgraph", 100, 100)
        bad["native_wire_sha256"] = ""
        with self.assertRaisesRegex(ValueError, "native_wire_binding_required"):
            paired_budget_projection([bad])

    def test_reorder_cannot_be_used_after_results_seen(self):
        with self.assertRaisesRegex(ValueError, "fixed_framework_order_required"):
            paired_budget_projection([request("dsh", 100, 100), request("langgraph", 100, 100)])

    def test_diagnostic_variant_or_easier_question_cannot_be_substituted(self):
        index = {"format": "frozen-local-rerank-native-fit-inputs-v1", "cases": [{"topic_id": "04", "question_id": "q_003",
                  "variants": [{"label": "title_bge_hybrid_rrf40", "file": "not-read", "sha256": "a" * 64}]}]}
        with self.assertRaisesRegex(ValueError, "fixed_case_unique_required"):
            select_fixed_case(index)
        index["cases"][0]["question_id"] = "q_002"
        index["cases"][0]["variants"][0]["label"] = "title_bge40"
        with self.assertRaisesRegex(ValueError, "preregistered_primary_variant_required"):
            select_fixed_case(index)

    def test_source_identity_and_complete_parent_are_enforced(self):
        case = {"question": {"question_id": "q_002", "question": "原题"}, "catalog": [
            {"paragraph_ref": f"P{i}", "doc_id": f"doc_{i}", "para_id": "p1", "title": "标题", "text": "原段🦊",
             "parent_complete": True, "paragraph_complete": True} for i in range(1, 41)]}
        validate_raw_case(case)
        changed = deepcopy(case)
        changed["catalog"][0]["paragraph_complete"] = False
        with self.assertRaisesRegex(ValueError, "complete_raw_parent_required"):
            validate_raw_case(changed)
        changed = deepcopy(case)
        changed["catalog"][1]["doc_id"] = "doc_1"
        with self.assertRaisesRegex(ValueError, "duplicate_parent_identity"):
            validate_raw_case(changed)


if __name__ == "__main__":
    unittest.main()
