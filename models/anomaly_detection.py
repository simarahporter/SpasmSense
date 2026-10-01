import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class EMGAnomalyDetector:
    """
    Personalized anomaly detector for EMG feature vectors.

    The idea:
    train mainly on examples of a person's normal muscle activity,
    then flag unusual windows that differ strongly from that baseline.
    """

    def __init__(
        self,
        contamination=0.05,
        random_state=42,
    ):
        self.scaler = StandardScaler()

        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
        )

        self.feature_names = [
            "rms",
            "mav",
            "zero_crossings",
            "waveform_length",
            "median_frequency",
        ]

        self.is_fitted = False

    def _to_matrix(self, feature_dicts):
        """
        Convert feature dictionaries to a numeric matrix.
        """

        return np.array(
            [
                [features[name] for name in self.feature_names]
                for features in feature_dicts
            ],
            dtype=float,
        )

    def fit(self, normal_feature_dicts):
        """
        Learn the user's normal EMG behavior.

        Parameters
        ----------
        normal_feature_dicts : list of dict
            Feature dictionaries from EMG windows
            considered normal for that user.
        """

        X = self._to_matrix(normal_feature_dicts)

        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled)

        self.is_fitted = True

    def predict(self, feature_dict):
        """
        Evaluate one EMG window.

        Returns
        -------
        dict
            is_anomaly:
                True if the window looks unusual.

            anomaly_score:
                Lower values generally indicate
                more unusual activity.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Detector must be fitted before prediction."
            )

        X = self._to_matrix([feature_dict])
        X_scaled = self.scaler.transform(X)

        prediction = self.model.predict(X_scaled)[0]
        score = self.model.decision_function(X_scaled)[0]

        return {
            "is_anomaly": prediction == -1,
            "anomaly_score": float(score),
        }

    def predict_batch(self, feature_dicts):
        """
        Evaluate multiple EMG windows at once.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Detector must be fitted before prediction."
            )

        X = self._to_matrix(feature_dicts)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)

        results = []

        for prediction, score in zip(predictions, scores):
            results.append(
                {
                    "is_anomaly": prediction == -1,
                    "anomaly_score": float(score),
                }
            )

        return results