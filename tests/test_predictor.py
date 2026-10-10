import unittest
from unittest.mock import Mock, patch

from scipy.sparse import csr_matrix

from src.pipeline import predictor
from src.pipeline.predictor import predict_ticket


class PredictTicketTests(unittest.TestCase):
    def test_predict_ticket_returns_classification_payload(self):
        queue_tfidf = Mock()
        queue_tfidf.transform.return_value = csr_matrix((1, 1))
        queue_encoder = Mock()
        queue_encoder.transform.return_value = csr_matrix((1, 1))
        queue_classifier = Mock()
        queue_classifier.predict.return_value = ["IT"]

        priority_tfidf = Mock()
        priority_tfidf.transform.return_value = csr_matrix((1, 1))
        priority_encoder = Mock()
        priority_encoder.transform.return_value = csr_matrix((1, 1))
        priority_classifier = Mock()
        priority_classifier.predict.return_value = ["high"]

        models = {
            "type_model": Mock(**{"predict.return_value": ["network"]}),
            "queue_data": {
                "tfidf": queue_tfidf,
                "encoder": queue_encoder,
                "classifier": queue_classifier,
            },
            "priority_data": {
                "tfidf": priority_tfidf,
                "encoder": priority_encoder,
                "classifier": priority_classifier,
            },
        }
        with patch.object(predictor, "_load_models", return_value=models):
            result = predict_ticket(
                "VPN is down and employees cannot connect to the network"
            )

        self.assertIsInstance(result, dict)
        self.assertEqual(
            result,
            {"ticket_type": "network", "priority": "HIGH", "queue": "IT"},
        )


if __name__ == "__main__":
    unittest.main()
