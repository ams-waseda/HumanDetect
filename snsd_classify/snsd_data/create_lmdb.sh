#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
set -e

EXAMPLE=examples/snsd_classify
DATA=/home/akentu/caffe/examples/snsd_classify/
BUILD=/home/akentu/caffe/build/tools
SIZE=64
BACKEND="lmdb"
echo "Creating ${BACKEND}..."


rrm -rf test.txt
rm -rf train.txt

echo "make txtfile..."
python labelTxt_create.py png test etc kanban
python labelTxt_create.py png train etc kanban

echo "resize..."
mogrify -geometry ${SIZE}x${SIZE}! test/*/*/*.png


echo "Creating ${BACKEND}..."



rm -rf ../snsd_cifar10_train_${BACKEND}
rm -rf ../snsd_cifar10_test_${BACKEND}


$BUILD/convert_imageset /home/akentu/caffe/examples/snsd_classify/ train.txt ../snsd_cifar10_train_${BACKEND} 1 -backend ${BACKEND} ${SIZE} ${SIZE}

$BUILD/convert_imageset /home/akentu/caffe/examples/snsd_classify/ test.txt ../snsd_cifar10_test_${BACKEND} 1 -backend ${BACKEND} ${SIZE} ${SIZE}
echo "make mean.binaryproto"

cd $CAFFE_ROOT
build/tools/compute_image_mean.bin -backend=${BACKEND} ./examples/snsd_classify/snsd_cifar10_train_${BACKEND} ./examples/snsd_classify/snsd_mean.binaryproto

echo "start learning"
build/tools/caffe train --solver examples/snsd_classify/snsd_cifar10_full_solver.prototxt

echo "Done."
