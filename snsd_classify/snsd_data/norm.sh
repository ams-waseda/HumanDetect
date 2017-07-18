#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
set -e



cd train/etc/etc
for file in `ls`; do convert ${file} -equalize ${file}; done
cd ../../kanban/kanban
for file in `ls`; do convert ${file} -equalize ${file}; done
cd../../../test/etc/etc
for file in `ls`; do convert ${file} -equalize ${file}; done
cd ../../kanban/kanban
for file in `ls`; do convert ${file} -equalize ${file}; done
cd ../../../

