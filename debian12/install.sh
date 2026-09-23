#!/bin/bash
# Install TensorFlow 1.13.2 for Debian 12 bookworm, 32-bit.
# No compile. Does not change /usr/bin/python3.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorflow-i686}"
DO_APT=0
SKIP_SMOKE=0

usage() {
  cat <<EOF
Install TensorFlow 1.13.2 and a private Python 3.6 on Debian 12 (32-bit).

Usage: $0 [--prefix DIR] [--apt] [--skip-smoke]

  --prefix DIR   where to install (default: ~/.local/opt/tensorflow-i686)
  --apt          install missing Debian libraries with sudo apt-get
  --skip-smoke   do not run the short import test at the end

After it finishes, run:  ./test-mnist.py
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --prefix) PREFIX=$2; shift 2 ;;
    --apt) DO_APT=1; shift ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

echo "This installs TensorFlow 1.13.2 for 32-bit Debian 12."
echo "Files go to: $PREFIX"
echo "System Python is not changed."
echo

mach=$(uname -m)
[[ $mach == i686 || $mach == i386 ]] || die "This build is for 32-bit PCs (i686). uname -m is $mach."
grep -qw sse2 /proc/cpuinfo || die "This CPU has no SSE2. The wheel will not run here."

# head closes the pipe early; without "|| true", pipefail treats that as failure.
libc_line=$(ldd --version 2>&1 | head -n1 || true)
libc_ver=$(printf '%s\n' "$libc_line" | grep -oE '[0-9]+\.[0-9]+' | head -n1 || true)
[[ -n $libc_ver ]] || die "Could not read the glibc version."
lowest=$(printf '%s\n' 2.36 "$libc_ver" | sort -V | head -1)
[[ $lowest == 2.36 ]] || die "glibc $libc_ver is older than 2.36. Install Debian 12 (bookworm) or newer."

if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  if [[ ${ID:-} != debian || ${VERSION_ID:-} != 12 ]]; then
    echo "Warning: this build was made for Debian 12. This machine is ${PRETTY_NAME:-unknown}." >&2
  fi
fi

APT_PKGS=(
  libffi8 zlib1g libbz2-1.0 liblzma5 libsqlite3-0 libreadline8
  libncursesw6 libstdc++6 libuuid1 ca-certificates
)
missing=()
for p in "${APT_PKGS[@]}"; do
  if ! dpkg-query -W -f '${Status}' "$p" 2>/dev/null | grep -q 'install ok installed'; then
    missing+=("$p")
  fi
done
if ((${#missing[@]})); then
  echo "Missing packages: ${missing[*]}"
  echo "  sudo apt-get install -y ${missing[*]}"
  if [[ $DO_APT == 1 ]]; then
    sudo apt-get install -y "${missing[@]}"
  else
    die "Install those packages, or run this script again with --apt."
  fi
fi

WHEEL=$(ls -1 "$HERE/wheels"/tensorflow-1.13.2-cp36-cp36m-linux_i686.whl 2>/dev/null | head -1 || true)
[[ -n $WHEEL ]] || die "Missing wheels/tensorflow-1.13.2-cp36-cp36m-linux_i686.whl"
RUNTIME="$HERE/runtime/cpython-3.6.15-i686-bookworm.tar.gz"
[[ -f $RUNTIME ]] || die "Missing runtime/cpython-3.6.15-i686-bookworm.tar.gz"

echo "Unpacking Python 3.6..."
mkdir -p "$PREFIX/python" "$PREFIX/bin"
tar -xzf "$RUNTIME" -C "$PREFIX/python"
[[ -x $PREFIX/python/bin/python3 ]] || die "The Python archive did not contain bin/python3."

export LD_LIBRARY_PATH="$PREFIX/python/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_DIR=/etc/ssl/certs

echo "Creating a virtual environment and installing TensorFlow from local wheels..."
"$PREFIX/python/bin/python3" -m venv "$PREFIX/venv"
# ensurepip's pip is too old to see every wheel. Install the pinned 3.6 toolchain first.
"$PREFIX/venv/bin/python" -m pip install --no-index --find-links "$HERE/wheels" \
  'pip==21.3.1' 'setuptools==59.6.0' 'wheel==0.37.1'
"$PREFIX/venv/bin/python" -m pip install --no-index --find-links "$HERE/wheels" "$WHEEL"

{
  printf '%s\n' '#!/bin/sh'
  printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$PREFIX/python/lib"
  printf 'export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt\n'
  printf 'export SSL_CERT_DIR=/etc/ssl/certs\n'
  printf 'exec %q "$@"\n' "$PREFIX/venv/bin/python"
} >"$PREFIX/bin/python"
chmod +x "$PREFIX/bin/python"

if [[ $SKIP_SMOKE != 1 ]]; then
  echo "Checking that TensorFlow imports..."
  "$PREFIX/bin/python" - <<'PY'
import tensorflow as tf
print("version", tf.__version__)
hello = tf.constant("ok")
with tf.Session() as sess:
    print("session", sess.run(hello))
PY
fi

echo
echo "Installed."
echo "  $HERE/test-mnist.py"
echo "  $PREFIX/bin/python -c 'import tensorflow as tf; print(tf.__version__)'"
