import copy
import itertools
import unittest

from build_dataset import (TRAIN_TREES, TRANSFER_TREES, eval_tree, policy_label,
                           policy_records, public_record, validate_partitions, messages)


class PolicyTests(unittest.TestCase):
    def test_missing_irrelevant_fact_does_not_force_unknown(self):
        self.assertEqual(policy_label(("and", "a", "b"), {"a": False}), "ineligible")
        self.assertEqual(policy_label(("or", "a", "b"), {"a": True}), "eligible")
        self.assertEqual(policy_label(("and", "a", "b"), {"a": True}), "unknown")
        self.assertEqual(policy_label(("not", "a"), {}), "unknown")

    def test_exhaustive_rule_semantics(self):
        for a, b, c in itertools.product([False, True], repeat=3):
            facts = dict(a=a, b=b, c=c)
            expected = [a and b, a or b, a and not b, (a and b) or c,
                        (a or b) and not c, (a and not b) or (not a and c)]
            for tree, value in zip(TRAIN_TREES + TRANSFER_TREES, expected):
                self.assertEqual(eval_tree(tree, facts), value)

    def test_concrete_policy_records_match_oracle(self):
        rows = policy_records("train", 300)
        for row in rows:
            facts = row["state"]["record"]
            p = row["provenance"]
            reconstructed = {}
            if "days_since_purchase" in facts:
                reconstructed["a"] = facts["days_since_purchase"] <= p["atom_thresholds"]["a"]
            if "item_unopened" in facts:
                reconstructed["b"] = facts["item_unopened"]
            if "order_value_usd" in facts:
                reconstructed["c"] = facts["order_value_usd"] >= p["atom_thresholds"]["c"]
            self.assertEqual(reconstructed, p["atom_facts"])
            self.assertEqual(policy_label(p["tree"], reconstructed), row["answer_key"])
        for start in range(0, len(rows), 3):
            a, b, c = rows[start:start+3]
            self.assertEqual([r["answer_key"] for r in (a,b,c)], ["eligible", "ineligible", "unknown"])
            self.assertEqual(len({r["group_id"] for r in (a,b,c)}), 1)
            af, bf, cf = [r["state"]["record"] for r in (a,b,c)]
            changed = [k for k in af if af[k] != bf[k]]
            self.assertEqual(len(changed), 1)
            self.assertEqual(cf, {k:v for k,v in af.items() if k != changed[0]})

    def test_repeatability_and_transfer_families(self):
        self.assertEqual(policy_records("train", 30), policy_records("train", 30))
        train = {str(t) for t in TRAIN_TREES}
        for row in policy_records("policy_transfer", 30):
            self.assertNotIn(str(row["provenance"]["tree"]), train)


class DataTests(unittest.TestCase):
    def test_leakage_rejected(self):
        train = policy_records("train", 3)
        leaked = copy.deepcopy(train[0])
        leaked.update(id="different-id", split="test")
        with self.assertRaisesRegex(AssertionError, "leakage"):
            validate_partitions({"train": train, "test": [leaked]})

    def test_label_mapping(self):
        source = {"repo": "fixture", "revision": "fixed"}
        for label, key in enumerate(["entailed", "neutral", "contradicted"]):
            item = {"raw": {"premise":"The cat is asleep.", "hypothesis":"The cat is awake.", "label":label},
                    "state":"The cat is asleep.", "gold":key, "index":label, "original_split":"train", "group":str(label)}
            row = public_record("mnli", item, "train", source)
            selected = next(o for o in row["options"] if o["label"] == row["answer"])
            self.assertEqual(selected["key"], key)
            self.assertEqual(messages(row)[-1]["content"], row["answer"])

    def test_score_levels_stay_ordered(self):
        row = public_record("sst5", {"raw":{"text":"good", "label":3}, "state":"good", "gold":"3", "index":1,
                                     "original_split":"train", "group":"g"}, "train", {"repo":"fixture", "revision":"fixed"})
        self.assertEqual([o["key"] for o in row["options"]], ["0", "1", "2", "3", "4"])
        self.assertEqual(row["answer"], "D")


if __name__ == "__main__":
    unittest.main()
