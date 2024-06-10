#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.

EXAMPLE=examples/mnist
DATA=data/mnist
BUILD=build/examples/mnist

BACKEND="lmdb"

echo "Creating ${BACKEND}..."

rm -rf $EXAMPLE/mnist_train_sigmoid_${BACKEND}
rm -rf $EXAMPLE/mnist_test_sigmoid_${BACKEND}

$BUILD/convert_mnist_data_sigmoid.bin $DATA/train-images-idx3-ubyte \
  $DATA/train-labels-idx1-ubyte $EXAMPLE/mnist_train_sigmoid_${BACKEND} --backend=${BACKEND}
$BUILD/convert_mnist_data_sigmoid.bin $DATA/t10k-images-idx3-ubyte \
  $DATA/t10k-labels-idx1-ubyte $EXAMPLE/mnist_test_sigmoid_${BACKEND} --backend=${BACKEND}

echo "Done."
