#!/bin/bash

TIMEOUT=3600

if [ -z "$1" ] ; then
    echo Usage: batch_eval.sh PROTOTXT_FILE SUB_FOLDER BATCH_NUM GPU_ID OUTPUT_NAME BN_CONVERT_BATCH BN_CONVERT_PROTOTXT_FILE
    echo
    exit
fi

pushd `dirname $0` > /dev/null ; SCRIPT_DIR=`pwd`; popd > /dev/null

CAFFE="$SCRIPT_DIR/caffe.bin"
PROTOTXT_FILE=$1
SUB_FOLDER=$2
BATCH_NUM=$3
GPU_ID=$4
OUTPUT_NAME="$5"
BN_CONVERT_BATCH=$6
BN_CONVERT_PROTOTXT_FILE=$7
if [ -z "$OUTPUT_NAME" ] ; then
    OUTPUT_NAME="accuracy";
fi
if [ -z "$BN_CONVERT_PROTOTXT_FILE" ] ; then
    BN_CONVERT_PROTOTXT_FILE="$PROTOTXT_FILE"
fi

if [ "$BN_CONVERT_BATCH" == "" ]; then
    BN_CONVERT_BATCH=2000
fi

echo OUTPUT_NAME ":" $OUTPUT_NAME

OUTPUT_FILE=$SUB_FOLDER/eval_acc.txt

ls $SUB_FOLDER/*.caffemodel | sed -e 's/^.*_\([0-9]*\)\.caffemodel$/\1/' \
                | sort -u > $SUB_FOLDER/cur_remain.txt

if [ -a "$OUTPUT_FILE" ]; then
    cat $OUTPUT_FILE | awk  '{print $1}' | sort -u > $SUB_FOLDER/cur_exist.txt
    cp $SUB_FOLDER/cur_remain.txt $SUB_FOLDER/cur_full.txt
    comm -23 $SUB_FOLDER/cur_full.txt $SUB_FOLDER/cur_exist.txt > $SUB_FOLDER/cur_remain.txt
fi

BN_STR=`cat $PROTOTXT_FILE | grep 'type:[ \t]*"BN"\|type:[ \t]*"BatchNorm"'`
if [ "$BN_STR" == "" ]; then
    IS_BN=0
else
    IS_BN=1
fi

cat $SUB_FOLDER/cur_remain.txt | while read ITER
do
    line=`ls $SUB_FOLDER/*_$ITER.caffemodel | tail -n1` #if multiple, only get the first
    printf '%16d : ' $ITER
    if [ "$IS_BN" -eq "0" ]; then
        TEST_WEIGHTS="$line"
    else
        TEST_WEIGHTS="$line".bn
        if [ -e "$TEST_WEIGHTS" ]; then
            printf '%s' "bn model existed : "
        else
            printf '%s' "convert to bn model : "
            bn4inference.sh $BN_CONVERT_PROTOTXT_FILE $line $TEST_WEIGHTS $BN_CONVERT_BATCH $GPU_ID > "$line".log-bn 2>&1
        fi
    fi
    printf '%s' "test : "
    ACC=`timeout $TIMEOUT $CAFFE test -model $PROTOTXT_FILE -weights $TEST_WEIGHTS -iterations $BATCH_NUM -gpu $GPU_ID 2>&1 \
        | tee "$TEST_WEIGHTS".log-test | egrep "\] $OUTPUT_NAME"' =' | sed -e 's/^.*'"$OUTPUT_NAME"' = \([0-9\.enan]*\)[ \t]*$/\1/'`
    echo $ACC
#printf '\t%f\n' $ACC
    if [ "$ACC" == "" ]; then
        echo "Failed to get accuracy" >&2
    else
        chmod +w $OUTPUT_FILE 2>/dev/null
        echo $ITER $ACC >> $OUTPUT_FILE
        chmod -w $OUTPUT_FILE 2>/dev/null
    fi
done


