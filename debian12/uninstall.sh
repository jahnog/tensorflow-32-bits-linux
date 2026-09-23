#!/bin/bash
# Remove the prefix created by install.sh. Does not touch system Python.
set -euo pipefail

PREFIX="${PREFIX:-$HOME/.local/opt/tensorflow-i686}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --prefix)
      PREFIX=$2
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--prefix DIR]"
      echo "Removes the TensorFlow 1.13.2 prefix (default: ~/.local/opt/tensorflow-i686)."
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -d $PREFIX ]]; then
  echo "Nothing to remove: $PREFIX"
  exit 0
fi
if [[ ! -x $PREFIX/bin/python && ! -d $PREFIX/venv ]]; then
  echo "Refusing to delete $PREFIX (it does not look like this TensorFlow install)." >&2
  exit 1
fi
rm -rf "$PREFIX"
echo "Removed $PREFIX"
