#!/bin/bash

TARGE_DIR=$1

if [ "$TARGET_DIR" == "" ]; then
    TARGET_DIR=.
fi
TARGET_DIR=`readlink -f "$TARGET_DIR"`

TMP_LIST_PATH=`echo "$TARGET_DIR" | md5sum | awk '{print $1}'``date | md5sum | awk '{print $1}'`
TMP_LIST_PATH="/tmp/caffe_wipe_test_"$TMP_LIST_PATH

find "$TARGET_DIR" -type f -name 'eval_acc.txt' | while read EVAL_PATH ; do
    CACHE_DIR=`dirname "$EVAL_PATH"`
    caffe_wipe_cache.sh "$CACHE_DIR" | sort > $TMP_LIST_PATH.cache
    cat "$EVAL_PATH" | awk '{print $1}' | while read ITER; do
        ls "$CACHE_DIR"/*_iter_"$ITER".* 2>/dev/null
    done | sort > $TMP_LIST_PATH.tested
    comm -12 $TMP_LIST_PATH.tested $TMP_LIST_PATH.cache
done


rm -f $TMP_LIST_PATH.*

