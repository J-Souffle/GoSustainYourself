import unittest
from carbon_perdiction_service import CarbonPredictionService

class TestCarbonChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = CarbonPredictionService()
        cls.test_input = {
            "Body Type": "Average",
            "Sex": "Male",
            "Diet": "Non-Vegetarian",
            "How Often Shower": "Daily",
            "Heating Energy Source": "Electricity",
            "Transport": "Car",
            "Vehicle Type": "SUV",
            "Social Activity": "High",
            "Monthly Grocery Bill": "200-300",
            "Frequency of Traveling by Air": "Rarely",
            "Vehicle Monthly Distance Km": "500-1000",
            "Waste Bag Size": "Medium",
            "Waste Bag Weekly Count": "2",
            "How Long TV PC Daily Hour": "4-6",
            "How Many New Clothes Monthly": "2-3",
            "How Long Internet Daily Hour": "4-6",
            "Energy efficiency": "Medium",
            "Recycling": "Sometimes",
            "Cooking_With": "Electricity"
        }

    def test_model_initialization(self):
        self.assertIsNotNone(self.service.nn_model)
        self.assertIsNotNone(self.service.xgb_model)
        self.assertIsNotNone(self.service.lgb_model)
        self.assertIsNotNone(self.service.rf_model)
        self.assertIsNotNone(self.service.hgb_model)

    def test_prediction(self):
        result = self.service.predict_carbon_emission(self.test_input)
        self.assertIn('prediction', result)
        self.assertIn('model_predictions', result)
        self.assertIn('suggestions', result)
        self.assertIsInstance(result['prediction'], float)
        self.assertGreater(result['prediction'], 0)

    def test_suggestions(self):
        result = self.service.predict_carbon_emission(self.test_input)
        self.assertIsInstance(result['suggestions'], str)
        self.assertGreater(len(result['suggestions']), 0)

    def test_model_predictions(self):
        result = self.service.predict_carbon_emission(self.test_input)
        expected_models = {
            'neural_network', 'xgboost', 'lightgbm', 
            'random_forest', 'hist_gradient_boosting'
        }
        self.assertEqual(set(result['model_predictions'].keys()), expected_models)
        for pred in result['model_predictions'].values():
            self.assertIsInstance(pred, float)
            self.assertGreater(pred, 0)

if __name__ == '__main__':
    unittest.main()