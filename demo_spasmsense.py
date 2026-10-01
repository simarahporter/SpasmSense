import numpy as np
import matplotlib.pyplot as plt

from signal_processing.filtering import preprocess_emg
from signal_processing.segmentation import segment_with_timestamps
from signal_processing.features import extract_features
from models.anomaly_detection import EMGAnomalyDetector
from realtime.stream_detector import SpasmEventDetector


SAMPLING_RATE = 1000
DURATION_SECONDS = 12

rng = np.random.default_rng(42)

time = np.arange(
    0,
    DURATION_SECONDS,
    1 / SAMPLING_RATE,
)

# Simulated normal EMG
signal = rng.normal(
    0,
    0.08,
    size=len(time),
)

signal += (
    0.03 * np.sin(2 * np.pi * 80 * time)
    + 0.02 * np.sin(2 * np.pi * 120 * time)
)

# Inject an abnormal burst from 8.0 to 9.5 seconds
burst_start = 8.0
burst_end = 9.5

burst_mask = (
    (time >= burst_start)
    & (time <= burst_end)
)

signal[burst_mask] += rng.normal(
    0,
    0.45,
    np.sum(burst_mask),
)

signal[burst_mask] += (
    0.20
    * np.sin(
        2 * np.pi * 65 * time[burst_mask]
    )
)

# Preprocess
clean_signal = preprocess_emg(
    signal,
    sampling_rate=SAMPLING_RATE,
)

# Split into short windows
segments = segment_with_timestamps(
    clean_signal,
    sampling_rate=SAMPLING_RATE,
    window_seconds=0.5,
    overlap=0.5,
)

# Extract features from each window
feature_rows = []

for segment in segments:
    features = extract_features(
        segment["signal"],
        SAMPLING_RATE,
    )

    feature_rows.append(
        {
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "features": features,
        }
    )

# Train anomaly detector only on early normal data
normal_training_features = [
    row["features"]
    for row in feature_rows
    if row["end_time"] < 6.0
]

detector = EMGAnomalyDetector(
    contamination=0.05,
)

detector.fit(
    normal_training_features
)

# Detect unusual windows
detected_times = []
event_detector = SpasmEventDetector(
    history_size=5,
    minimum_anomalies=3,
)

print("\nSpasmSense synthetic test\n")

for row in feature_rows:
    result = detector.predict(
        row["features"]
    )

    state = event_detector.update(
        result["is_anomaly"]
    )

    if result["is_anomaly"]:
        detected_times.append(
            (
                row["start_time"],
                row["end_time"],
            )
        )
            
        

        print(
    f"{row['start_time']:.2f}s - "
    f"{row['end_time']:.2f}s | "
    f"{state}"
)

# Plot
plt.figure(figsize=(12, 5))

plt.plot(
    time,
    clean_signal,
    label="Filtered synthetic EMG",
)

plt.axvspan(
    burst_start,
    burst_end,
    alpha=0.25,
    label="Injected abnormal activity",
)

for start, end in detected_times:
    plt.axvspan(
        start,
        end,
        alpha=0.08,
    )

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title(
    "SpasmSense Synthetic EMG Anomaly Detection"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "spasmsense_demo.png",
    dpi=150,
)

plt.show()
