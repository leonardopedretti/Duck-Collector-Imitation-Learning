"""Training utilities for reproducing the final neural-network policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .features import extract_features


@dataclass(frozen=True)
class TrainingConfig:
    hidden1: int = 32
    hidden2: int = 16
    weight_decay: float = 1e-4
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 200
    patience: int = 10
    test_size: float = 0.2
    random_state: int = 42
    class_weight_alpha: float = 0.5


def build_model(input_dim: int, num_classes: int, config: TrainingConfig):
    """Build the final MLP architecture."""

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import BatchNormalization, Dense, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    model = Sequential(
        [
            Input(shape=(input_dim,)),
            BatchNormalization(),
            Dense(config.hidden1, activation="relu", kernel_regularizer=l2(config.weight_decay)),
            Dense(config.hidden2, activation="relu", kernel_regularizer=l2(config.weight_decay)),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def softened_class_weights(labels: np.ndarray, alpha: float = 0.5) -> dict[int, float]:
    """Compute balanced class weights with a configurable damping factor."""

    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels).astype(
        np.float32
    )
    weights = weights / weights.mean()
    weights_soft = 1.0 + alpha * (weights - 1.0)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights_soft)}


def train_final_model(
    observations: np.ndarray,
    actions: np.ndarray,
    room_ids: np.ndarray | None = None,
    output_path: str | Path = "models/best_model.keras",
    config: TrainingConfig = TrainingConfig(),
):
    """Train and save the final all-rooms model."""

    from tensorflow.keras.callbacks import EarlyStopping

    features = extract_features(observations)
    labels = actions.astype(int)

    if room_ids is None:
        x_train, x_val, y_train, y_val = train_test_split(
            features,
            labels,
            test_size=config.test_size,
            stratify=labels,
            random_state=config.random_state,
        )
    else:
        train_parts = []
        val_parts = []
        train_label_parts = []
        val_label_parts = []
        for room_id in sorted(np.unique(room_ids.astype(int))):
            mask = room_ids.astype(int) == room_id
            room_features = features[mask]
            room_labels = labels[mask]
            stratify = room_labels if np.min(np.bincount(room_labels)) >= 2 else None
            x_room_train, x_room_val, y_room_train, y_room_val = train_test_split(
                room_features,
                room_labels,
                test_size=config.test_size,
                stratify=stratify,
                random_state=config.random_state,
            )
            train_parts.append(x_room_train)
            val_parts.append(x_room_val)
            train_label_parts.append(y_room_train)
            val_label_parts.append(y_room_val)

        x_train = np.concatenate(train_parts, axis=0)
        x_val = np.concatenate(val_parts, axis=0)
        y_train = np.concatenate(train_label_parts, axis=0)
        y_val = np.concatenate(val_label_parts, axis=0)

    model = build_model(
        input_dim=x_train.shape[1],
        num_classes=int(np.max(labels)) + 1,
        config=config,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        restore_best_weights=True,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        class_weight=softened_class_weights(y_train, alpha=config.class_weight_alpha),
        callbacks=[early_stopping],
        verbose=0,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)

    y_pred = model.predict(x_val, verbose=0).argmax(axis=1)
    metrics = {
        "macro_f1": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_val, y_pred, digits=3, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
        "epochs_ran": len(history.history.get("loss", [])),
    }
    return model, metrics
