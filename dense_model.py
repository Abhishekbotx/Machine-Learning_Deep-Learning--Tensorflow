import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

#Why dividing by 255? Because MNIST images are grayscale images, and each pixel value is between:
#0 → black    255 → white
# We divide by 255 to scale pixel values into a smaller range (0–1), which helps the neural network train faster, converge smoother, and avoid exploding gradients.

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# Visualize samples
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i], cmap='gray') #28 x 28 matrix 
    # imshow() = display an image
    # x_train[i] = the i-th handwritten digit from the MNIST dataset (it will be random)

    plt.title(f"Label: {y_train[i]}") # prints the actual digit value (0–9) from the dataset
    plt.axis('off') #off is used to remove the axes, ticks, borders, so image looks clean
plt.show()

# Build model: Define structure of neural network :-) You decide how many brain layers you'll use
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # neuural netowrk expects 1D input
    keras.layers.Dense(128, activation='relu'), # hidden layer
    keras.layers.Dropout(0.2),                  # used dropout to avoid overfitting
    keras.layers.Dense(64, activation='relu'),  # hidden layer
    keras.layers.Dense(10, activation='softmax')# output layer, softmax converts output numbers into probabilities that sum to 1.
])      

# Eg: [0.02, 0.01, 0.95, 0.01, ...]

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# Compile: Choose optimization + loss + metrics :-) You choose how you'll learn (speed, rules, scoring)
model.compile(
    optimizer='adam', # Algorithm that adjusts weights during training
    loss=loss_fn, # Loss function for multi-class classification
    metrics=['accuracy'] # Track how often predictions are correct
)
# Train: Model learns patterns and adjusts weights :-) You practice writing digits many times
history = model.fit(
    x_train, y_train,
    epochs=5, # Model sees the entire dataset 5 times
    validation_split=0.2, # 20% of training data used as validation
    batch_size=32 # Model updates weights after every 32 samples
)

# Evaluate: Test how well it performs on unseen data :-) Someone tests you on new handwriting
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f} {test_loss}")


# Save model
model.save("mnist_dense_model.keras")
print("Model saved!")

# Load model
loaded_model = keras.models.load_model("mnist_dense_model.keras")
print("Model loaded!")

# Pick a test sample
sample = x_test[0:1] #🌟 shape becomes (1, 28, 28) x_test[0:1] → shape (1, 28, 28) ✔ model understands this as:Batch of 1 image
                    # if x_test[0] → shape (28, 28) ❌ model will complain
# Predict
prediction = loaded_model.predict(sample)
print("Predicted Digit:", np.argmax(prediction))
