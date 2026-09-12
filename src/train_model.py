"""
Trains the Stage 3 classifier: MobileNetV2 (frozen ImageNet base) + a small
new head, fine-tuned on our own category folders.  (Member B)

Input: data/train/<category>/*.jpg  (produced by scripts/prepare_dataset.py)
Output: models/classifier.keras, models/class_names.json, and a printed
        validation accuracy + confusion matrix — the number you need to
        quote in the demo ("we measured >=80% on a held-out split").

Run:
    python src/train_model.py --data data/train --epochs 12
"""

import argparse
import json
import os

from classification import INPUT_SIZE


def build_model(num_classes: int, input_shape=(160, 160, 3)):
    import tensorflow as tf
    from tensorflow.keras import layers

    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False  # <-- this is the "frozen base" line for the viva

    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1),
    ], name="augmentation")

    inputs = tf.keras.Input(shape=input_shape)
    x = augmentation(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(data_dir: str, epochs: int, batch_size: int, out_dir: str):
    import numpy as np
    import tensorflow as tf

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=42,
        image_size=INPUT_SIZE, batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=42,
        image_size=INPUT_SIZE, batch_size=batch_size,
    )
    class_names = train_ds.class_names
    print(f"Classes ({len(class_names)}): {class_names}")

    # MobileNetV2 expects [-1, 1] inputs; do the same scaling used at
    # inference time in classification.preprocess_for_model.
    normalize = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)
    train_ds = train_ds.map(lambda x, y: (normalize(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds_raw = val_ds  # keep an unmapped copy around for the confusion matrix
    val_ds = val_ds.map(lambda x, y: (normalize(x), y)).prefetch(tf.data.AUTOTUNE)

    model = build_model(num_classes=len(class_names), input_shape=INPUT_SIZE + (3,))
    model.summary()

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    val_accuracy = history.history["val_accuracy"][-1]
    print(f"\nFinal validation accuracy: {val_accuracy:.1%}")
    if val_accuracy < 0.80:
        print("Below the 80% target — see the 'Risks & fallbacks' notes: try "
              "fewer/more-distinct categories before reaching for a bigger model.")

    # Confusion matrix, for the "correctness of categories" discussion
    y_true, y_pred = [], []
    for images, labels in val_ds_raw:
        normed = normalize(images)
        preds = model.predict(normed, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    try:
        from sklearn.metrics import confusion_matrix, classification_report
        print("\nConfusion matrix (rows=true, cols=predicted):")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
    except ImportError:
        pass

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "classifier.keras")
    classes_path = os.path.join(out_dir, "class_names.json")
    model.save(model_path)
    with open(classes_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"\nSaved model to {model_path} and class names to {classes_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/train")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    main(args.data, args.epochs, args.batch_size, args.out)
