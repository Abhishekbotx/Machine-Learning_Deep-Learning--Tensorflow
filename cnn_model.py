import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# MNIST already comes built-in with Keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# RESHAPE DATA FOR CNN:
# CNNs expect input in the format: (batch_size, height, width, channels)
# -1 → lets TensorFlow calculate batch size automatically (-1 means auto-calc batch size),
# 28,28 → image height & width
# 1 → grayscale channel 3.->RGB 4.->RGBA
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1) / 255.0

# Why divide by 255?
# Pixel values range from 0→255. Scaling to (0→1) makes training faster and stable.


#Convo 2D shrinks size
# BUILD CNN MODEL::
model = keras.Sequential([
    
    # 1️. Convolution layer (detects edges, strokes, patterns)
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    
    # 2️. Max pooling — reduces image size while keeping important features
    keras.layers.MaxPooling2D((2,2)), #convert shape to 14 x 14 x 32
    # Note: Pooling halves width and height:

    # 3️. Second convolution layer — detects more complex features(patterns)
    keras.layers.Conv2D(64, (3,3), activation='relu'),

    # 4️. Pool again to compress information
    keras.layers.MaxPooling2D((2,2)), #convert shape to 7 x 7 x 64
    

    # 5️. Flatten — convert 2D feature maps → 1D vector
    keras.layers.Flatten(),

    # 6️. Dense layer — learns combinations of extracted features
    keras.layers.Dense(64, activation='relu'),

    # 7️. Output layer — 10 neurons for digits (0–9), softmax → probabilities
    keras.layers.Dense(10, activation='softmax')
])



# COMPILE THE MODEL ::
model.compile(
    optimizer="adam",                      # How weights update
    loss="sparse_categorical_crossentropy",# Best for multi-class classification
    metrics=["accuracy"]                   # Track model performance
)


# TRAIN THE MODEL
# history = model.fit(
#     x_train, y_train,
#     epochs=5,                # How many times the model sees entire dataset
#     validation_split=0.2     # 20% of training data reserved for validation
# )



# # EVALUATE ON TEST DATA
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print(f" Test accuracy: {test_acc:.4f}")
# print(f" Test loss: {test_loss:.4f}")


# Save + Load
# model.save("mnist_cnn_model.keras") #Saves the entire trained model into a file named:mnist_cnn_model.keras
loaded_model = keras.models.load_model("mnist_cnn_model.keras") #Load the saved model back
# Now, loaded_model is the same as your model, but:
# • You didn’t retrain it
# • It has learned weights already
# • You can use it for predictions directly

# Prediction test
sample = x_test[0].reshape(1, 28, 28, 1)  # Batch size = 1 (predicting 1 image at a time)
prediction = loaded_model.predict(sample) # Make prediction using the loaded model
print(f"prediction: {prediction}")
# Model processes the image and outputs a prediction vector like: 
#prediction: [[2.7433719e-08 1.7853939e-09 1.5923648e-05 1.2158064e-05 1.9324667e-10 
#              1.7967013e-09  4.8422772e-12 9.9996984e-01 1.0685101e-08 1.9827498e-06]]

# Display
plt.imshow(sample.reshape(28, 28), cmap='gray')
plt.title("Sample Image")
plt.axis('off')
plt.show()

print("\n🧠 Predicted Digit:", np.argmax(prediction))