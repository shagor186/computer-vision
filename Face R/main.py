import tensorflow as tf

# Automatic GPU use
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])
c = a + b
print(c)

# Force CPU
with tf.device('/CPU:0'):
    d = a * b
    print(d)
