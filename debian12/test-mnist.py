#!/usr/bin/env python3
"""Train the small MNIST network from the tensorflow-32-bits-linux README.

Run this after debian12/install.sh. It switches to the private Python 3.6
that install.sh created, then trains for 5 epochs and prints test accuracy.
"""
import os
import sys

PREFIX = os.environ.get(
    "TENSORFLOW_I686_PREFIX",
    os.path.expanduser("~/.local/opt/tensorflow-i686"),
)
PY = os.path.join(PREFIX, "bin", "python")


def _reexec():
    if not os.path.exists(PY):
        sys.stderr.write(
            "TensorFlow is not installed at %s\n"
            "From this folder, run: ./install.sh\n" % PREFIX
        )
        sys.exit(1)
    if os.path.realpath(sys.executable) != os.path.realpath(PY):
        os.execv(PY, [PY, os.path.abspath(__file__)] + sys.argv[1:])


_reexec()

import tensorflow as tf  # noqa: E402

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
