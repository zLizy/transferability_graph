#!/bin/sh

# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

if [ $# -nq 1 ]; then
  echo "Usage: ./download_imagenet.sh DATA_ROOT"
  exit 1
fi

DATA_DIR="../dataset" #$1
TASK2VEC_REPO="./"


mkdir -p "$DATA_DIR"/imagenet
cd "$DATA_DIR"/imagenet || exit 1

wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
tar -xzf ILSVRC2012_img_val.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz
tar -xzf ILSVRC2012_bbox_val_v3.tgz
mv lists/* ILSVRC2012_img_val/
cp $TASK2VEC_REPO/support_files/cub/*.json "$DATA_DIR"/cub/CUB_200_2011