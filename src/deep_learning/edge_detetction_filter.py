import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("./assets/cat.jpg").convert("L")  # grayscale
image = image.resize((256, 256))                  # resize for consistency
image = np.array(image, dtype=np.float32) / 255.0 # normalize [0,1]
image = np.expand_dims(image, axis=(0, -1))       # shape: (1, 256, 256, 1)
# Original loaded image	(256, 256)
# After expand_dims(axis=0) → batch dim	(1, 256, 256)
# After expand_dims(axis=-1) → channel dim	(1, 256, 256, 1)

#  Define Edge Detection Filters 

# Vertical Edge Filter (Sobel-like)
vertical_filter = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# Horizontal Edge Filter
horizontal_filter = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# Reshape filters for TensorFlow: (H, W, in_channels, out_channels)
vertical_filter = vertical_filter.reshape(3, 3, 1, 1)
horizontal_filter = horizontal_filter.reshape(3, 3, 1, 1)

# Apply Convolution 
vertical_edges = tf.nn.conv2d(image, vertical_filter, strides=[1,1,1,1], padding="SAME")
horizontal_edges = tf.nn.conv2d(image, horizontal_filter, strides=[1,1,1,1], padding="SAME")

"Why is it written as [1, 1, 1, 1]?"
"Because in TensorFlow’s low-level operation tf.nn.conv2d, the stride must match the tensor shape"
"[ batch_stride, height_stride, width_stride, channel_stride ] "
"means do not skip any images in the batch (batch_stride=1), move 1 pixel at a time vertically "
"(height_stride=1), move 1 pixel at a time horizontally (width_stride=1), and do not skip any channels (channel_stride=1)."

  
# Convert tensor → numpy for visualization
vertical_edges = np.abs(vertical_edges.numpy().squeeze())   # abs() to fix negatives
horizontal_edges = np.abs(horizontal_edges.numpy().squeeze())

# Plot Results for Original and Edge Detected Images
plt.figure(figsize=(12, 4))


plt.subplot(1, 3, 1)
plt.imshow(image.squeeze(), cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(vertical_edges, cmap="gray")
plt.title("Vertical Edges")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(horizontal_edges, cmap="gray")
plt.title("Horizontal Edges")
plt.axis("off")

plt.show()
