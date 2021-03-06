import tensorflow as tf
from tensorflow.keras.layers import Conv2D
 
# The inputs are 28x28 RGB images with `channels_last` and the batch
# size is 4.
input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D( 2, 3, activation='relu', input_shape=input_shape[1:])(x)
print(y.shape)
#   (4, 26, 26, 2)

# With `dilation_rate` as 2.
input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(
2, 3, activation='relu', dilation_rate=2, input_shape=input_shape[1:])(x)
print(y.shape)
#   (4, 24, 24, 2)

# With `padding` as "same".
input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D( 2, 3, activation='relu', padding="same", input_shape=input_shape[1:])(x)
print(y.shape)
#   (4, 28, 28, 2)

# With extended batch shape [4, 7]:
input_shape = (4, 7, 28, 28, 3)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D( 2, 3, activation='relu', input_shape=input_shape[2:])(x)
print(y.shape)
#   (4, 7, 26, 26, 2)