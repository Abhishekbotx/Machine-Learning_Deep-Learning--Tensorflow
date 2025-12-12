# Example 3 - Sentiment Analysis (Binary Classification) - Without Embedding Layer
# Classify whether a review is positive or negative

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# IMDB reviews have different lengths, but Dense expects a fixed number of neurons
# We are ensuring every input is exactly 200 words.

# Pad sequences to ensure fixed-length inputs
x_train = pad_sequences(x_train, maxlen=200)
x_test = pad_sequences(x_test, maxlen=200)

# Build model using only Dense layers
model = models.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128, input_length=200),  # Embedding layer to convert word indices to dense vectors
    layers.Flatten(input_shape=(200,)),  # Convert 2D sequences to 1D
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Remember last layer activation for binary classification
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5, batch_size=512)

# Save & Load
model.save("models/sentiment_analysis_model.keras")
loaded_model = tf.keras.models.load_model("models/sentiment_analysis_model.keras")

# Evaluate model
test_loss, test_accuracy = loaded_model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Make a prediction on the first test sample
predicted_sentiment = model.predict(x_test[0:1])
print("Predicted Sentiment:", "Positive" if predicted_sentiment[0][0] > 0.5 else "Negative")
# Note: Using an Embedding layer helps convert word indices into dense vectors that capture semantic meaning.

# Example: Before vs After Embedding

# Before Embedding (Word Indices)
# [10, 482, 21, 345, 7, 99, 4]  # Just numbers

# After Embedding (Word Vectors, Each of Size 128)
# [
#   [0.12, -0.25, 0.78, ..., 0.05],   # Word 10
#   [0.34, 0.67, -0.12, ..., -0.89],  # Word 482
#   [0.08, 0.15, -0.32, ..., 0.40],   # Word 21
#   ...
# ]
# Each word index is now represented by a 128-dimensional vector capturing semantic meaning