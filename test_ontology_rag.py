import unittest
from pathlib import Path

from ontology_rag import load_json, query_semantics, retrieve, validate_ontology


ROOT = Path(__file__).parent


class OntologyRagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ontology = load_json(ROOT / "banking_ontology.json", "Ontology")
        cls.index = load_json(ROOT / "ontology_index/ontology_rag_index.json", "Index")

    def test_ontology_schema_is_valid(self):
        validate_ontology(self.ontology)

    def test_query_maps_to_controlled_semantics(self):
        concepts, predicates, refs = query_semantics(
            self.index,
            "Chủ thể dữ liệu có những quyền gì?",
        )
        self.assertIn("data_subject", concepts)
        self.assertIn("has_right", predicates)
        self.assertEqual([], refs)

    def test_specific_document_number_routes_to_that_document(self):
        results, semantics = retrieve(
            self.index,
            "Thông tư 09/2020/TT-NHNN quy định gì về an toàn hệ thống thông tin?",
            5,
        )
        self.assertEqual(["09/2020/TT-NHNN"], semantics["legal_references"])
        self.assertTrue(results)
        self.assertTrue(all(item[1]["document"] == "VanBanGoc_09.2020.TT.NHNN.pdf" for item in results))

    def test_specific_actor_and_control_beat_ocr_noise(self):
        results, _ = retrieve(
            self.index,
            "Tổ chức tín dụng phi ngân hàng phải làm gì về hệ thống kiểm soát nội bộ?",
            3,
        )
        self.assertTrue(results)
        self.assertIn("non_bank_credit_institution", results[0][1]["concepts"])
        self.assertIn("internal_control", results[0][1]["concepts"])


if __name__ == "__main__":
    unittest.main()
