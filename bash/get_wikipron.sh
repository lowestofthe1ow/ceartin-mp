#!/bin/bash

OUT_PATH="data/wikipron/"
FILE_URL="https://drive.google.com/uc?export=view&id=1cJmdkvvJNQb-xHS_fYafEwMetu5IWdoQ"
FILE_NAME="wikipron_tl.tsv"

mkdir -p $OUT_PATH

wget -O $OUT_PATH$FILE_NAME $FILE_URL

echo "Wrote to file $OUT_PATH$FILE_NAME."
