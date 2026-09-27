#!/bin/sh
# Clone the benchmarks of the model runs at the commits we used.
# Usage: sh tools/get_benchmarks.sh [--kramabench]
#   InfiAgent-DABench, QRData and DiscoveryBench: about 540 MB.
#   --kramabench also clones KramaBench: about 1.6 GB more.
set -e
DEST=${BENCH:-data}
mkdir -p "$DEST"

get() {
  name=$1
  url=$2
  commit=$3
  if [ ! -d "$DEST/$name/.git" ]; then
    git clone -q "$url" "$DEST/$name"
  fi
  git -C "$DEST/$name" checkout -q "$commit"
  echo "$DEST/$name at $(git -C "$DEST/$name" rev-parse HEAD)"
}

get InfiAgent https://github.com/InfiAgent/InfiAgent.git 3d6c4a70198e0a41fadf539f5b43c88b8c1a2d9c
get QRData https://github.com/xxxiaol/QRData.git de450af45ff7101b328bb064c6b475f73414a7ed
get discoverybench https://github.com/allenai/discoverybench.git c31fcf011e070f021a5f5b906896d0821f6880e8
if [ "$1" = "--kramabench" ]; then
  get Kramabench https://github.com/mitdbg/Kramabench.git b2e0d77540263f8b6119f977fe79a8c4386b5a02
fi
(cd "$DEST/QRData/benchmark" && unzip -q -o data.zip)
echo "unpacked the QRData tables to $DEST/QRData/benchmark/data"
