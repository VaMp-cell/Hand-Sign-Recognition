import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sys

RANDOM_SEED = 42

dataset = 'model/keypoint_classifier/keypoint.csv'
model_save_path = 'model/keypoint_classifier/keypoint_classifier.keras'
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'
label_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'

# Read labels to determine NUM_CLASSES
try:
    with open(label_path, encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f) if row]
    NUM_CLASSES = len(keypoint_classifier_labels)
    print(f"Training for {NUM_CLASSES} classes: {keypoint_classifier_labels}")
except Exception as e:
    print(f"Error reading labels: {e}")
    sys.exit(1)

# Dataset reading
try:
    X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
except Exception as e:
    print(f"Error reading dataset: {e}")
    sys.exit(1)

X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

# Model building
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# Callbacks
cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save_path, verbose=1, save_weights_only=False)
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

# Compilation
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(
    X_train, y_train, 
    epochs=1000, 
    batch_size=128, 
    validation_data=(X_test, y_test), 
    callbacks=[cp_callback, es_callback]
)

# Evaluation
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
print(f"\nValidation Loss: {val_loss}, Validation Accuracy: {val_acc}\n")

# Load saved model for testing and conversion
model = tf.keras.models.load_model(model_save_path)
Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

print('Classification Report:')
print(classification_report(y_test, y_pred))

# Save for inference
model.save(model_save_path, include_optimizer=False)

# Transform model (quantization) to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

with open(tflite_save_path, 'wb') as f:
    f.write(tflite_quantized_model)

print(f"\nModel saved successfully to {tflite_save_path}!")
