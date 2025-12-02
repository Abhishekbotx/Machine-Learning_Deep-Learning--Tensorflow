import tensorflow as tf
print("TF:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU:", tf.config.list_physical_devices('GPU'))
