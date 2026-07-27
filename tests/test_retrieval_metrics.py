import unittest

import torch

from utils.metric import retrieval_metrics


class RetrievalMetricsTest(unittest.TestCase):
    def test_perfect_similarity_has_perfect_retrieval_metrics(self):
        similarity = torch.eye(10)
        identities = torch.arange(10)

        metrics = retrieval_metrics(
            similarity,
            query_ids=identities,
            gallery_ids=identities,
            prefix="t2i",
        )

        self.assertEqual(
            metrics,
            {
                "t2i_R1": 100.0,
                "t2i_R5": 100.0,
                "t2i_R10": 100.0,
                "t2i_mAP": 100.0,
                "t2i_mINP": 100.0,
            },
        )

    def test_retrieval_metrics_use_all_positives_for_map(self):
        similarity = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
        query_ids = torch.tensor([1])
        gallery_ids = torch.tensor([1, 2, 1, 3])

        metrics = retrieval_metrics(
            similarity,
            query_ids=query_ids,
            gallery_ids=gallery_ids,
            prefix="t2i",
        )

        self.assertAlmostEqual(
            metrics["t2i_mAP"],
            (1.0 + 2.0 / 3.0) / 2.0 * 100,
            places=4,
        )
        self.assertAlmostEqual(
            metrics["t2i_mINP"],
            2.0 / 3.0 * 100,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
