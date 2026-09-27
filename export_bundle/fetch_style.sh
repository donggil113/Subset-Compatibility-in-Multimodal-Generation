#!/bin/sh
# Fetch the official ICML 2026 style files into paper/ and verify them.
# They are not vendored: icml2026.sty carries no redistribution licence.
#
#   sh fetch_style.sh                      # download from the official URL
#   ICML_STYLE_ZIP=/path/icml2026.zip sh fetch_style.sh   # use a local copy
#
# The expected SHA-256 is that of the zip retrieved on 2026-09-26. If the
# official file has changed since then, this script stops. Do not build with
# a different style silently; record the new hash and the date instead.
set -eu

URL="https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip"
EXPECTED="8b29290f5828e176debb57ea9cc00252502973d55ea561a2f18a7f0a326bfc6c"
HERE=$(cd "$(dirname "$0")" && pwd)
DEST="$HERE/paper"
TMP=$(mktemp "${TMPDIR:-/tmp}/icml2026.XXXXXX")
trap 'rm -f "$TMP"' EXIT

if [ -n "${ICML_STYLE_ZIP:-}" ]; then
  cp "$ICML_STYLE_ZIP" "$TMP"
elif command -v curl >/dev/null 2>&1; then
  curl -fsSL -o "$TMP" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -q -O "$TMP" "$URL"
else
  echo "need curl or wget (or set ICML_STYLE_ZIP)" >&2; exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  GOT=$(sha256sum "$TMP" | cut -d' ' -f1)
else
  GOT=$(shasum -a 256 "$TMP" | cut -d' ' -f1)
fi
if [ "$GOT" != "$EXPECTED" ]; then
  echo "SHA-256 mismatch: got $GOT, expected $EXPECTED" >&2
  exit 1
fi

FILES="icml2026.sty icml2026.bst algorithm.sty algorithmic.sty fancyhdr.sty"
if command -v unzip >/dev/null 2>&1; then
  unzip -o -j -q "$TMP" $FILES -d "$DEST"
else
  python3 - "$TMP" "$DEST" $FILES <<'PY'
import sys, zipfile, os
zf, dest, names = sys.argv[1], sys.argv[2], sys.argv[3:]
with zipfile.ZipFile(zf) as z:
    for n in names:
        with open(os.path.join(dest, n), "wb") as f:
            f.write(z.read(n))
PY
fi
echo "OK: style files verified (SHA-256 $EXPECTED) and placed in $DEST"
