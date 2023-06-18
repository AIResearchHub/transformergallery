import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from stockai.transformer import Transformer, TransformerLM, BlockRecurrentTransformer
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

class Metrics:
    @staticmethod
    def mae(y_true, y_pred):
        """Calculate Mean Absolute Error.
        Parameters:
        y_true (numpy.array): Array of actual values.
        y_pred (numpy.array): Array of predicted values.
        """
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true, y_pred):
        """Calculate Root Mean Squared Error.
        Parameters:
        y_true (numpy.array): Array of actual values.
        y_pred (numpy.array): Array of predicted values.
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mape(y_true, y_pred):
        """Calculate Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def smape(y_true, y_pred):
        """Calculate Symmetric Mean Absolute Percentage Error."""
        return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

    @staticmethod
    def mase(y_true, y_pred):
        """Calculate Mean Absolute Scaled Error."""
        n = len(y_true)
        d = np.abs(  np.diff( y_true) ).sum()/(n-1)
        errors = np.abs(y_true - y_pred )
        return errors.mean()/d

    @staticmethod
    def msle(y_true, y_pred):
        """Calculate Mean Squared Logarithmic Error."""
        return mean_squared_log_error(y_true, y_pred)

    @staticmethod
    def r2(y_true, y_pred):
        """Calculate R-squared."""
        return r2_score(y_true, y_pred)

    @staticmethod
    def theil_u(y_true, y_pred):
        """Calculate Theil's U statistic."""
        return np.sqrt(np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
class Evaluator:
    """Class for loading model, generating predictions and calculating evaluation metrics."""

    def __init__(self, model_path, data_path, config_path):
        """Initialize Evaluator with paths to model, data, and configuration.
        Parameters:
        model_path (str): Path to the file containing the pre-trained model.
        data_path (str): Path to the CSV file containing the data.
        config_path (str): Path to the JSON file containing the model configuration.
        """
        self.model_path = model_path
        self.data_path = data_path
        self.config_path = config_path
        self.model = None
        self.data = None
        self.y_true = None
        self.y_pred = None
        self.config = None

    def load_model_config(self):
        """Load model configuration from a JSON file."""
        with open(self.config_path) as f:
            self.config = json.load(f)

    def load_data(self, standardize=False):
        """Load and optionally standardize the data."""
        self.data = pd.read_csv(self.data_path)
        if standardize:
            scaler = StandardScaler()
            self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)
        self.y_true = self.data.iloc[:, -1].values

    def load_model(self):
        """Load a pretrained model from a file."""
        model_class = {
            "Transformer": Transformer,
            "TransformerLM": TransformerLM,
            "BlockRecurrentTransformer": BlockRecurrentTransformer,
        }.get(self.config["model_type"])

        if model_class is None:
            raise ValueError(f"Unknown model type: {self.config['model_type']}")

        self.model = model_class(**self.config["model_args"])
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def generate_predictions(self):
        """Generate predictions from the model."""
        X = torch.tensor(self.data.iloc[:, :-1].values, dtype=torch.float32)
        with torch.no_grad():
            self.y_pred, _ = self.model(X)
        self.y_pred = self.y_pred.numpy()


def evaluate(self):
    """Calculate and print all evaluation metrics in an organized way."""
    metrics = {
        "MAE": Metrics.mae(self.y_true, self.y_pred),
        "RMSE": Metrics.rmse(self.y_true, self.y_pred),
        # ... TODO: include other metrics here ...
    }

    print("Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    def plot_predictions(self):
        """Plot the true and predicted values."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.y_true, label='True')
        plt.plot(self.y_pred, label='Predicted')
        plt.legend()
        plt.show()

def main():
    # model_path = "path/to/model.pth"
    # data_path = "path/to/data.csv"
    # config_path = "path/to/config.json"

    evaluator = Evaluator(model_path, data_path, config_path)

    evaluator.load_model_config()
    evaluator.load_data(standardize=True)
    evaluator.load_model()
    evaluator.generate_predictions()
    evaluator.evaluate()
    evaluator.plot_predictions()

if __name__ == "__main__":
    main()
