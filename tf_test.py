import tensorflow as tf

# Assert GPU exists
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
assert len(physical_devices) > 0

# Dataset
mnist = tf.keras.datasets.mnist

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalise
x_train, x_test = x_train / 255.0, x_test / 255.0

# Input shape (m, 28, 28) -> (m, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Build a minimal CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
        input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training
history = model.fit(x_train, y_train, epochs=5)

# Examine loss
loss = history.history['loss']
assert loss[0] > loss[-1]
