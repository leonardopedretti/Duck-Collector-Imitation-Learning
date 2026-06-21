"""Policy wrappers around the trained Keras classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import YellowMaskConfig, extract_single_features


class KerasFeaturePolicy:
    """Callable simulator policy backed by the saved MLP model."""

    def __init__(
        self,
        model_path: str | Path = "models/best_model.keras",
        feature_config: YellowMaskConfig = YellowMaskConfig(),
        rng=None,
    ) -> None:
        from tensorflow import keras

        self.model_path = Path(model_path)
        self.model = keras.models.load_model(self.model_path)
        self.feature_config = feature_config
        self.rng = rng

    def predict_proba(self, observation: np.ndarray) -> np.ndarray:
        features = extract_single_features(observation, config=self.feature_config, rng=self.rng)
        return self.model.predict(features[None, :], verbose=0)[0]

    def __call__(self, observation: np.ndarray) -> int:
        return int(np.argmax(self.predict_proba(observation)))
