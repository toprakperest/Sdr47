import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras


class AIModels:
    """
    Manages AI models for object detection and classification in GPR data.
    This includes loading pre-trained models, training new models (if applicable),
    and performing inference.
    """

    def __init__(self, model_path_cnn: str = None, model_path_lstm: str = None):
        """
        Initializes the AIModels class.

        Args:
            model_path_cnn (str, optional): Path to the pre-trained CNN model file. Defaults to None.
            model_path_lstm (str, optional): Path to the pre-trained LSTM model file. Defaults to None.
        """
        self.cnn_model = None
        self.lstm_model = None

        if model_path_cnn:
            try:
                self.cnn_model = keras.models.load_model(model_path_cnn)
                print(f"CNN model loaded successfully from {model_path_cnn}")
            except Exception as e:
                print(f"Error loading CNN model from {model_path_cnn}: {e}")

        if model_path_lstm:
            try:
                self.lstm_model = keras.models.load_model(model_path_lstm)
                print(f"LSTM model loaded successfully from {model_path_lstm}")
            except Exception as e:
                print(f"Error loading LSTM model from {model_path_lstm}: {e}")

        if not self.cnn_model:
            print("CNN model not loaded. Creating a default CNN model.")
            self.cnn_model = self._create_default_cnn_model()

        if not self.lstm_model:
            print("LSTM model not loaded. Creating a default LSTM model.")
            self.lstm_model = self._create_default_lstm_model()

    def _create_default_cnn_model(self, input_shape=(128, 1), num_classes=3):
        """Creates a default 1D CNN model for GPR data."""
        model = keras.Sequential(
            [
                layers.InputLayer(input_shape=input_shape),
                layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
                layers.MaxPooling1D(pool_size=2),
                layers.Flatten(),
                layers.Dense(units=100, activation='relu'),
                layers.Dense(num_classes, activation='softmax')  # num_classes: e.g., soil, rock, void
            ]
        )
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Default 1D CNN model created.")
        return model

    def _create_default_lstm_model(self, input_shape=(128, 1), num_classes=3):
        """Creates a default LSTM model for GPR data."""
        model = keras.Sequential(
            [
                layers.InputLayer(input_shape=input_shape),
                layers.LSTM(units=64, return_sequences=True),
                layers.LSTM(units=32),
                layers.Dense(units=100, activation='relu'),
                layers.Dense(num_classes, activation='softmax')  # num_classes: e.g., soil, rock, void
            ]
        )
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Default LSTM model created.")
        return model

    def predict_cnn(self, data):
        """
        Performs prediction using the loaded CNN model.
        Args:
            data (np.ndarray): Input data for prediction (e.g., processed GPR traces).
                               Shape should be (num_samples, sequence_length, num_features).
                               For example, if each GPR trace has 128 points, and you have 100 traces,
                               data shape could be (100, 128, 1) if using raw amplitude as a feature.
        Returns:
            np.ndarray: Prediction results, or None if an error occurs.
        """
        if self.cnn_model is None:
            print("CNN model is not loaded. Cannot predict.")
            return None
        try:
            # Ensure data is in the correct shape (num_samples, sequence_length, num_features)
            # Example: data might be (num_traces, num_points_per_trace, 1) if using amplitude as feature
            if data.ndim == 2:  # If we have (num_traces, num_points_per_trace)
                data = np.expand_dims(data, axis=-1)  # Add feature dimension

            predictions = self.cnn_model.predict(data)
            return predictions
        except Exception as e:
            print(f"Error during CNN prediction: {e}")
            return None

    def predict_lstm(self, data):
        """
        Performs prediction using the loaded LSTM model.
        Args:
            data (np.ndarray): Input data for prediction (e.g., time-series GPR data).
                               Shape should be (num_samples, timesteps, num_features).
                               For example, (num_traces, num_time_steps_per_trace, 1).
        Returns:
            np.ndarray: Prediction results, or None if an error occurs.
        """
        if self.lstm_model is None:
            print("LSTM model is not loaded. Cannot predict.")
            return None
        try:
            # Ensure data is in the correct shape (num_samples, timesteps, num_features)
            if data.ndim == 2:  # If we have (num_traces, num_time_steps)
                data = np.expand_dims(data, axis=-1)  # Add feature dimension

            predictions = self.lstm_model.predict(data)
            return predictions
        except Exception as e:
            print(f"Error during LSTM prediction: {e}")
            return None


# Example Usage (for testing, comment out when integrating):
if __name__ == '__main__':
    # Create dummy data for testing
    # Assume GPR traces with 128 points each
    num_traces = 10
    trace_length = 128
    dummy_gpr_data = np.random.rand(num_traces, trace_length)

    # Initialize AIModels (this will create default models if paths are not provided)
    ai_models = AIModels()

    # Test CNN prediction
    print("\n--- Testing CNN Prediction ---")
    cnn_predictions = ai_models.predict_cnn(dummy_gpr_data)
    if cnn_predictions is not None:
        print(f"CNN Predictions shape: {cnn_predictions.shape}")
        # print(f"CNN Predictions (first trace): {cnn_predictions[0]}")

    # Test LSTM prediction
    print("\n--- Testing LSTM Prediction ---")
    lstm_predictions = ai_models.predict_lstm(dummy_gpr_data)
    if lstm_predictions is not None:
        print(f"LSTM Predictions shape: {lstm_predictions.shape}")
        # print(f"LSTM Predictions (first trace): {lstm_predictions[0]}")

    print("\n--- AIModels tests complete ---")