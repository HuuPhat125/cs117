#!/bin/bash

# Xác định đường dẫn thư mục output
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../output" && pwd )"
cd $DIR

FILE=precomputed_ho_rcnn_models.tar.gz
ID=1NjXjR4IG6pxtRZqnU1XHfyZPivd6GE7X

# Kiểm tra nếu file đã tồn tại
if [ -f $FILE ]; then
  echo "File already exists..."
  exit 0
fi

echo "Downloading precomputed HO-RCNN models (4.0G)..."

# Tải file từ Google Drive sử dụng gdown
gdown --id $ID --output $FILE

echo "Unzipping..."

# Giải nén file tar.gz
tar zxvf $FILE

echo "Done."
