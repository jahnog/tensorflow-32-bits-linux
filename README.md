# Install Tensorflow in a 32 bits linux system

I used the following steps to install tensorflow in a old Asus Eee-Pc 1000H. Granted, it has been upgraded from the original 1 GB of RAM and an 80 GB HDD, to 2 GB of RAM and to 480 GB of SSD storage, that runs Ubuntu Xenial 32 bits without problems.

I also has been able to install it in a Debian 9 (stretch) 32 bits system, and the instructions are the same.

## Choose a convenient linux system

I have tested both the Ubuntu 16.04 (Xenial) and Debian 9.11 (Stretch) systems with 2 GB of RAM.

I set up the system to have 4 GB of SWAP space. With only 1 GB of SWAP, some compilations failed.

It's critical that the distribution has the version 8 of the Java SDK: openjdk-8-jdk

## Install the Java 8 SDK and build tools

``` bash
sudo apt-get update
sudo apt-get install openjdk-8-jdk
sudo apt-get install git zip unzip autoconf automake libtool curl zlib1g-dev swig build-essential
```

## Install Python libraries

Next, we install python 3 development libraries and the keras module that will be required by tensorflow.

``` bash
sudo apt-get install python3-dev python3-pip python3-wheel
sudo python3 -m pip install --upgrade pip
python3 -m pip install --user keras
```

You can use eithr python 3 or python 2 and compile tensorflow for that version. 

## Install and compile Bazel from sources

We need the source code bazel 0.19.2 distribution. We can obtain it and install in a new folder.

``` bash
wget https://github.com/bazelbuild/bazel/releases/download/0.19.2/bazel-0.19.2-dist.zip
mkdir Bazel-0-19.2
cd Bazel-0-19.2
unzip ../bazel-0.19.2-dist.zip
```

Before compiling, we need to remove line 30 of ./src/tools/singlejar/mapped_file_posix.inc file (**#error This code for 64 bit Unix.**) that throws an error if we are not in a 64 bit machine. This bazel version works ok in 32 bits. 

Also we need to increase the java memory available to Bazel and start compiling it.

``` bash
export BAZEL_JAVAC_OPTS="-J-Xmx1g"
./compile.sh
```

When it finishes (It can take several hours), we move the bazel compiled executable to some location in the current user's path

``` bash
cp output/bazel /home/user/.local/bin
```

## Compile Tensorflow from sources

Create a folder and clone tensorflow's 1.13.2 version to it. Starting from version 1.14, tensorflow uses the Intel MKL DNN optimization library that it only works in 64 bits systems. So 1.13.2 is the last version that runs in 32 bits.

``` bash
mkdir Tensorflow-1.13.2
cd Tensorflow-1.13.2
git clone -b v1.13.2 --depth=1 https://github.com/tensorflow/tensorflow .
```

Before compiling, we replace the references to 64 bit libraries to the 32 bit ones.

``` bash
grep -Rl "lib64"| xargs sed -i 's/lib64/lib/g'
```

We start the tensorflow configuration. We need to explicity disable the use of several optional libraries that are not available or not supported on 32 bit systems.

``` bash
export TF_NEED_CUDA=0
export TF_NEED_AWS=0
./configure
```

We have to take the following considerations:
* When asked to specify the location of python. [Default is /usr/bin/python]: We should respond **/usr/bin/python3** to use python 3.
* When asked to input the desired Python library path to use.  Default is [/usr/local/lib/python3.5/dist-packages] we just hit **Enter**
* We should respond **N** to all the Y/N questions.
* When asked to specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]: Just hit **Enter** 

Now we start compiling tensorflow disabling optional components like aws, kafka, etc.

``` bash
bazel build --config=noaws --config=nohdfs --config=nokafka --config=noignite --config=nonccl -c opt --verbose_failures //tensorflow/tools/pip_package:build_pip_package
```

If everything went ok, now we generate the pip package.

``` bash
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

And we install the pip package

``` bash
python3 -m pip install --user /tmp/tensorflow_pkg/tensorflow-1.13.2-cp35-cp35m-linux_i686.whl
``` 

## Test tensorflow

Now we run a small test to check that it works. We create a test.py file with the following contents:

``` python
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```

And we run the test

``` bash
python3 test.py
```
Here is the output

``` bash
Epoch 1/5
60000/60000 [==============================] - 87s 1ms/sample - loss: 0.2202 - acc: 0.9348
Epoch 2/5
60000/60000 [==============================] - 131s 2ms/sample - loss: 0.0963 - acc: 0.9703
Epoch 3/5
60000/60000 [==============================] - 135s 2ms/sample - loss: 0.0685 - acc: 0.9785
Epoch 4/5
60000/60000 [==============================] - 128s 2ms/sample - loss: 0.0526 - acc: 0.9828
Epoch 5/5
60000/60000 [==============================] - 128s 2ms/sample - loss: 0.0436 - acc: 0.9863
10000/10000 [==============================] - 3s 273us/sample - loss: 0.0666 - acc: 0.9800
```

Enjoy you new Tensorflow !!