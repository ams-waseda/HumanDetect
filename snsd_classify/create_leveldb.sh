#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
set -e

EXAMPLE=examples/snsd_classify
DATA=examples/snsd_classify/snsd_data
BUILD=/home/akentu/caffe/build/tools

BACKEND="lmdb"
cd train/etc/etc
for file in `ls`; do convert ${file} -equalize ${file}; done
cd ../kanban
for file in `ls`; do convert ${file} -equalize ${file}; done
cd../../../test/etc/etc
for file in `ls`; do convert ${file} -equalize ${file}; done
cd ../kanban
for file in `ls`; do convert ${file} -equalize ${file}; done
cd ../../../
echo "Creating ${BACKEND}..."

rm -rf snsd_classify_train_${BACKEND}
rm -rf snsd_classify_test_${BACKEND}
rm -rf mean.binaryproto

$BUILD/convert_imageset /home/akentu/caffe/examples/snsd_classify/ train.txt snsd_classify_train_${BACKEND} 1 -backend ${BACKEND} 64 64

$BUILD/convert_imageset /home/akentu/caffe/examples/snsd_classify/ test.txt snsd_classify_test_${BACKEND} 1 -backend ${BACKEND} 64 64

$BUILD/compute_image_mean -backend=${BACKEND} snsd_classify_train_${BACKEND} mean.binaryproto


echo "Done."
