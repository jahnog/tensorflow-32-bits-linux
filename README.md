# TensorFlow 1.13.2 for 32-bit Linux

Unofficial build of [TensorFlow 1.13.2](https://github.com/tensorflow/tensorflow/tree/v1.13.2) for 32-bit PCs.

1.13.2 is the last TensorFlow release that can run on i686. Starting with 1.14, TensorFlow requires 64-bit MKL-DNN.

This repository is not affiliated with Google or the TensorFlow authors.

| What you want | Where to go |
|---|---|
| Debian 12 (bookworm), 32-bit, **no compile** | [Install on Debian 12](#install-on-debian-12-without-compiling) |
| Build it yourself on Debian 12 | [Compile on Debian 12](#compile-tensorflow-1132-on-debian-12-i386) |
| Older Debian 9 or Ubuntu 16.04 / 18.04 | [OLD_README.md](OLD_README.md) and [`dist/`](dist/) |

The install scripts in this repository are MIT (see `LICENSE`). The TensorFlow wheel is Apache 2.0 (see `debian12/NOTICE`).

---

## Install on Debian 12 without compiling

You need:

- Debian 12 bookworm, **32-bit** (`uname -m` is `i686` or `i386`)
- A CPU with SSE2 (Pentium 4, Atom, Core 2, and later)
- About 1 GB of free disk
- glibc 2.36 or newer (that is what Debian 12 ships)

Debian 12’s normal Python is 3.11. TensorFlow 1.13.2 only works with Python 3.6, so the installer unpacks a private Python 3.6. It does not replace `/usr/bin/python3`.

```bash
git clone https://github.com/jahnog/tensorflow-32-bits-linux
cd tensorflow-32-bits-linux/debian12
./install.sh
./test-mnist.py
```

`./install.sh` copies Python and the wheels to `~/.local/opt/tensorflow-i686`. Pass `--prefix DIR` to put them somewhere else. If a library such as `libffi8` is missing, the script prints an `apt-get` line. `./install.sh --apt` installs those packages for you (it will ask for your sudo password).

When the install works, this prints `1.13.2`:

```bash
~/.local/opt/tensorflow-i686/bin/python -c 'import tensorflow as tf; print(tf.__version__)'
```

Remove it with `debian12/uninstall.sh`.

### Test: MNIST, 5 epochs

`debian12/test-mnist.py` is the small network below. It downloads the digits (about 11 MB) and trains for 5 epochs. On a 1.6 GHz Atom with 2 GB of RAM, one epoch takes about 3–4 minutes.

```python
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
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

Result on Debian 12 i386 (Atom N270, 2 GB RAM), using the wheel in `debian12/wheels/`:

```text
Epoch 1/5
60000/60000 [==============================] - 207s 3ms/sample - loss: 0.2202 - acc: 0.9344
Epoch 2/5
60000/60000 [==============================] - 219s 4ms/sample - loss: 0.0947 - acc: 0.9710
Epoch 3/5
60000/60000 [==============================] - 217s 4ms/sample - loss: 0.0696 - acc: 0.9779
Epoch 4/5
60000/60000 [==============================] - 232s 4ms/sample - loss: 0.0518 - acc: 0.9834
Epoch 5/5
60000/60000 [==============================] - 220s 4ms/sample - loss: 0.0438 - acc: 0.9858
10000/10000 [==============================] - 11s 1ms/sample - loss: 0.0642 - acc: 0.9810
```

You may see a warning that SSE “is not available”. On 32-bit Linux that check is wrong (see [Problems on Debian 12](#problems-on-debian-12)). Training still runs. `/proc/cpuinfo` on the machine above lists `sse`, `sse2`, and `ssse3`.

---

## Compile TensorFlow 1.13.2 on Debian 12 i386

Do this only if you want to rebuild. A machine with 2 GB of RAM and a slow disk needs about a day. The prebuilt files in `debian12/` are the result of this procedure.

### 1. Packages and swap

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential swig autoconf automake libtool patchelf \
  curl git zip unzip pkg-config \
  libffi-dev libbz2-dev libsqlite3-dev libreadline-dev \
  libncursesw5-dev liblzma-dev uuid-dev zlib1g-dev
```

Add about 4 GB of swap. With only 1 GB of swap, the Bazel and TensorFlow builds run out of memory.

If you run `earlyoom`, stop it before the compile and start it again when you are done. A default Debian install does not have `earlyoom`.

### 2. Java 8, OpenSSL 1.1, Python 3.6

Debian 12 does not ship OpenJDK 8. Install an i686 Java 8 JDK, for example [Azul Zulu 8](https://www.azul.com/downloads/?package=jdk#zulu), and put `bin` on `PATH`.

Debian 12’s OpenSSL is 3.x, and Python 3.6 cannot use it. Build OpenSSL 1.1.1w and install it into the same prefix you will use for Python, then build CPython 3.6.15:

```bash
# from the CPython 3.6.15 source tree, after OpenSSL 1.1.1w is installed
# into $PREFIX (include/ and lib/)
export CPPFLAGS="-I$PREFIX/include"
export LDFLAGS="-Wl,--enable-new-dtags -Wl,-rpath,\$ORIGIN/../lib -L$PREFIX/lib"
./configure --prefix="$PREFIX" --enable-shared --with-ensurepip=install
make -j1
make install
ln -sfn python3 "$PREFIX/bin/python"
```

`$PREFIX/bin/python` must exist. Bazel’s scripts run `python`, and Debian 12 does not provide that name.

Create a virtualenv with that interpreter and install the last pip that still supports Python 3.6. Do not upgrade pip to a current release; it will refuse Python 3.6.

```bash
"$PREFIX/bin/python3" -m venv ~/tf-venv
~/tf-venv/bin/python -m pip install 'pip==21.3.1' 'setuptools==59.6.0' 'wheel==0.37.1'
~/tf-venv/bin/pip install 'numpy>=1.16,<1.17' 'h5py==2.10.0' six keras-applications keras-preprocessing
```

`h5py` must come from a wheel. If pip tries to compile it, the build is missing HDF5 and will fail. `h5py==2.10.0` still publishes an i686 wheel.

### 3. Bazel 0.19.2

Download the [0.19.2 dist zip](https://github.com/bazelbuild/bazel/releases/download/0.19.2/bazel-0.19.2-dist.zip), unzip it, and compile with a small Java heap. Bazel 0.19 does not accept `--local_ram_resources`. With the default heap (about 32 MB) the analysis step never finishes.

```bash
export JAVA_HOME=/path/to/zulu8
export JAVA_TOOL_OPTIONS=-Xmx384m
export EXTRA_BAZEL_ARGS="--jobs=1 --cxxopt=-include --cxxopt=limits"
export BAZEL_JAVAC_OPTS="-J-Xmx512m"
./compile.sh
```

`--cxxopt=-include limits` is required on GCC 12. See below.

Copy the resulting `output/bazel` onto your `PATH`.

### 4. TensorFlow

```bash
git clone -b v1.13.2 --depth=1 https://github.com/tensorflow/tensorflow
cd tensorflow

# 32-bit Debian keeps libraries in lib/, not lib64.
# Do not follow bazel-* symlinks. They point into ~/.cache/bazel.
find . \( -path './.git' -o -path './bazel-*' \) -prune -o -type f -print0 \
  | xargs -0 grep -l lib64 | xargs -r sed -i 's/lib64/lib/g'

export PYTHON_BIN_PATH="$HOME/tf-venv/bin/python"
export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')"
export TF_NEED_CUDA=0 TF_NEED_AWS=0 TF_NEED_GCP=0 TF_NEED_HDFS=0
export TF_NEED_OPENCL=0 TF_NEED_OPENCL_SYCL=0 TF_NEED_ROCM=0 TF_NEED_TENSORRT=0
export TF_NEED_JEMALLOC=0 TF_NEED_KAFKA=0 TF_NEED_NGRAPH=0 TF_NEED_MKL=0
export TF_NEED_MPI=0 TF_NEED_VERBS=0 TF_NEED_GDR=0
export TF_ENABLE_XLA=0 TF_DOWNLOAD_CLANG=0 TF_SET_ANDROID_WORKSPACE=0
export CC_OPT_FLAGS="-march=i686 -msse2 -mfpmath=sse -Wno-sign-compare"
./configure   # accept the defaults; the variables above answer the questions

# If .tf_configure.bazelrc contains host_copt=-march=native, change it to -march=i686.
bazel build --jobs=1 \
  --config=opt -c opt \
  --copt=-march=i686 --copt=-msse2 --copt=-mfpmath=sse \
  --copt=-Wno-error --host_copt=-Wno-error \
  --host_copt=-march=i686 --host_copt=-msse2 \
  --cxxopt=-include --cxxopt=limits \
  --host_cxxopt=-include --host_cxxopt=limits \
  --config=noaws --config=nohdfs --config=nokafka --config=noignite --config=nonccl \
  --verbose_failures \
  //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
~/tf-venv/bin/pip install /tmp/tensorflow_pkg/tensorflow-1.13.2-cp36-cp36m-linux_i686.whl
```

Set `LD_LIBRARY_PATH` to the Python prefix `lib/` and `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt` before you run Python.

On a 1.6 GHz Atom the TensorFlow build took about 22 hours (`Elapsed time: 81112s`). Bazel itself, after the fixes below, took a few hours.

### 5. Smoke test

```bash
python - <<'PY'
import tensorflow as tf
print(tf.__version__)
h = tf.constant("ok")
with tf.Session() as s:
    print(s.run(h))
PY
```

Then run `debian12/test-mnist.py` (or the same script against your virtualenv).

---

## Problems on Debian 12

Each of these stopped a real build on Debian 12 with GCC 12 and glibc 2.36.

**`std::numeric_limits` is not a member of `std`.** GCC 12 no longer includes `<limits>` for you. Bazel’s ijar (`third_party/ijar/zlib_client.h`, `mapped_file_unix.cc`) and Abseil (`graphcycles.cc`) hit this. Pass `--cxxopt=-include limits` and `--host_cxxopt=-include limits`.

**`gettid` is already declared.** glibc 2.30 added `gettid()`. Bazel’s grpc and TensorFlow’s grpc still define `static long gettid()`. Rename those functions to `sys_gettid` in:

- `third_party/grpc/src/core/support/log_linux.c` (inside the Bazel source)
- `@grpc//` sources `src/core/lib/gpr/log_linux.cc` and `src/core/lib/iomgr/ev_epollex_linux.cc` (downloaded while TensorFlow configures)

**`std::rel_ops` is not a member of `std`.** Same GCC 12 change, this time for `<utility>`. google-cloud-cpp headers such as `google/cloud/iam_bindings.h` need `#include <utility>`.

**Downloaded files no longer match the checksums in TensorFlow 1.13.**

- ICU `release-62-1.tar.gz`: the old sha256 is `e15ffd84606323cbad5515bf9ecdf8061cc3bf80fb883b9e6aa162e485aa9761`. GitHub repacked the file. The current sha256 is `86b85fbf1b251d7a658de86ce5a0c8f34151027cc60b01e1b76f167379acf181`. Update `third_party/icu/workspace.bzl`.
- Python 2.7 `license.rst.txt`: the old sha256 is `7ca8f169368827781684f7f20876d17b4415bbc5cb28baa4ca4652f0dda05e9f`. The file on docs.python.org is now `629431b6e4f268457ec1c1a9c9032506e3b720312e22cc8f679f76c2decacad2`. Update `tensorflow/workspace.bzl`.

**“Compiled to use SSE, but these aren’t available”.** On i686, CPUID clobbers `%ebx`, which position-independent code uses as the GOT pointer, so TensorFlow’s CPU check is wrong. `/proc/cpuinfo` can list `sse` and the check still aborts. In `tensorflow/core/platform/cpu_feature_guard.cc`, make `CheckFeatureOrDie` a warning on 32-bit (`!defined(__x86_64__)`), the same way the Android branch already does. Also replace `host_copt=-march=native` in `.tf_configure.bazelrc` with `-march=i686`. The wheel in this repository was built with `-march=i686 -msse2`.

**`yes | ./configure` fails under `set -o pipefail`.** `yes` gets a broken pipe after configure has already finished. That non-zero status is not a configure failure.

**Do not search `bazel-*` for `lib64`.** Those names are symlinks into `~/.cache/bazel`. A recursive replace there corrupts Bazel’s own install, and the next build stops with “corrupt installation”.

**Do not point Bazel 0.19 `--distdir` at a folder unless every download is in that folder.** It will not look on the network for the rest.

---

## Older Debian 9 and Ubuntu instructions

The original install and compile steps for Debian 9 and Ubuntu 16.04 / 18.04 are in [OLD_README.md](OLD_README.md). The wheels from that guide are still in `dist/`:

- `dist/tensorflow-1.13.2-cp35-cp35m-linux_i686.whl`
- `dist/tensorflow-1.13.2-cp36-cp36m-linux_i686.whl`

Those files are not the Debian 12 build. On Debian 12 use `debian12/`.
