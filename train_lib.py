"""Shared training utilities: 1D CNN from the paper, augmentation, int8 export."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

FLOOR_DB = -100.0


def set_seed(seed: int):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


class Scaler:
    """Global mean/std standardisation fitted on the training fold only."""

    def fit(self, X):
        X = np.maximum(X.astype(np.float32), FLOOR_DB)
        self.mu, self.sd = float(X.mean()), float(X.std() + 1e-6)
        return self

    def __call__(self, X):
        return ((np.maximum(X.astype(np.float32), FLOOR_DB) - self.mu) / self.sd).astype(np.float32)


def build_cnn(n_classes: int, input_shape=(98, 40)) -> tf.keras.Model:
    """Paper Ch. 3.4.2: two conv blocks (8, 16 filters, kernel 3), max-pool, dropout 0.25, dense softmax.

    Input is (time frames, mel bands): Conv1D slides over time, the 40 mel bands are channels.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(2, padding="same"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(2, padding="same"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])


def augment(x, y):
    """SpecAugment-style masking + random gain on standardised log-mel (training folds only)."""
    T, F = 98, 40
    x = x + tf.random.uniform([], -0.3, 0.3)  # gain shift (~±6 dB after scaling)
    t0 = tf.random.uniform([], 0, T - 15, dtype=tf.int32)
    tw = tf.random.uniform([], 0, 15, dtype=tf.int32)
    tmask = tf.logical_or(tf.range(T) < t0, tf.range(T) >= t0 + tw)
    f0 = tf.random.uniform([], 0, F - 5, dtype=tf.int32)
    fw = tf.random.uniform([], 0, 5, dtype=tf.int32)
    fmask = tf.logical_or(tf.range(F) < f0, tf.range(F) >= f0 + fw)
    mask = tf.cast(tmask[:, None] & fmask[None, :], x.dtype)
    x = x * mask + tf.random.normal(tf.shape(x), stddev=0.05)
    return x, y


def fit(X, y, n_classes, groups, class_weight=True, epochs=40, batch=128, seed=0, verbose=0):
    """Train with a grouped 10% validation split for early stopping."""
    set_seed(seed)
    inner = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    tr, va = next(inner.split(X, y, groups))
    sw = np.ones(len(tr), dtype=np.float32)
    if class_weight:  # weighted cross-entropy via per-sample weights w_c = N / (K * n_c)
        present = np.unique(y[tr])
        w = compute_class_weight("balanced", classes=present, y=y[tr])
        lut = dict(zip(present, w))
        sw = np.array([lut[c] for c in y[tr]], dtype=np.float32)
    ds_tr = (tf.data.Dataset.from_tensor_slices((X[tr], y[tr].astype(np.int32), sw))
             .shuffle(len(tr), seed=seed)
             .map(lambda x, y, w: (*augment(x, y), w), num_parallel_calls=tf.data.AUTOTUNE)
             .batch(batch).prefetch(2))
    ds_va = tf.data.Dataset.from_tensor_slices((X[va], y[va].astype(np.int32))).batch(512)
    model = build_cnn(n_classes, X.shape[1:])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(ds_tr, validation_data=ds_va, epochs=epochs, verbose=verbose,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
    return model


def to_int8(model, X_rep) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    rep = X_rep[np.random.choice(len(X_rep), min(300, len(X_rep)), replace=False)]
    conv.representative_dataset = lambda: ([r[None]] for r in rep)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def predict_int8(tflite: bytes, X) -> np.ndarray:
    it = tf.lite.Interpreter(model_content=tflite)
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    it.resize_tensor_input(inp["index"], [len(X), *X.shape[1:]])
    it.allocate_tensors()
    s, z = inp["quantization"]
    it.set_tensor(inp["index"], np.clip(np.round(X / s + z), -128, 127).astype(np.int8))
    it.invoke()
    so, zo = out["quantization"]
    return (it.get_tensor(out["index"]).astype(np.float32) - zo) * so
