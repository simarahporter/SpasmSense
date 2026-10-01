import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_feature_matrix(feature_dicts):
    """
    Convert a list of feature dictionaries into a numeric matrix.

    Expected keys:
        rms
        mav
        zero_crossings
        waveform_length
        median_frequency
    """

    feature_names = [
        "rms",
        "mav",
        "zero_crossings",
        "waveform_length",
        "median_frequency",
    ]

    X = np.array(
        [
            [features[name] for name in feature_names]
            for features in feature_dicts
        ],
        dtype=float,
    )

    return X, feature_names


def train_baseline_model(
    feature_dicts,
    labels,
    test_size=0.2,
    random_state=42,
):
    """
    Train a simple baseline classifier for EMG activity.

    Parameters
    ----------
    feature_dicts : list of dict
        Feature values extracted from EMG windows.
    labels : array-like
        Class labels for each window.

    Returns
    -------
    dict
        Model, scaler, metrics, and feature names.
    """

    X, feature_names = build_feature_matrix(feature_dicts)
    y = np.asarray(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced",
    )

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
    )

    return {
        "model": model,
        "scaler": scaler,
        "accuracy": accuracy,
        "classification_report": report,
        "feature_names": feature_names,
        "y_test": y_test,
        "predictions": predictions,
    }


def predict_activity(
    model,
    scaler,
    feature_dict,
):
    """
    Predict the class of a single EMG window.
    """

    X, _ = build_feature_matrix([feature_dict])

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]

    probabilities = model.predict_proba(X_scaled)[0]

    return {
        "prediction": prediction,
        "probabilities": probabilities,
    }
