#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../data" && pwd )"
cd $DIR

FILE=hico_20160224_det.tar.gz
ID=1dUByzVzM6z1Oq4gENa1-t0FLhr0UtDaS

# Kiểm tra nếu tệp đã tồn tại
if [ -f $FILE ]; then
  echo "File already exists..."
  exit 0
fi

echo "Downloading HICO-DET (7.5G)..."

# Tải tệp từ Google Drive sử dụng gdown
gdown --id $ID --output $FILE

echo "Unzipping..."

# Giải nén tệp tar.gz
tar zxvf $FILE

echo "Done."
