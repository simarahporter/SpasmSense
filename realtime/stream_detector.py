from collections import deque


class SpasmEventDetector:
    """
    Converts individual anomaly predictions into sustained events.

    A single unusual EMG window should not automatically be treated
    as a possible spasm-like event. This detector requires several
    abnormal windows within a recent history.
    """

    def __init__(
        self,
        history_size=5,
        minimum_anomalies=3,
    ):
        self.history = deque(
            maxlen=history_size
        )

        self.minimum_anomalies = minimum_anomalies

    def update(self, is_anomaly):
        """
        Add the newest anomaly result and classify the current state.
        """

        self.history.append(
            bool(is_anomaly)
        )

        anomaly_count = sum(
            self.history
        )

        if anomaly_count >= self.minimum_anomalies:
            return "possible_sustained_spasm_like_event"

        if is_anomaly:
            return "elevated_activity"

        return "normal"

    def reset(self):
        """
        Clear recent history.
        """

        self.history.clear()