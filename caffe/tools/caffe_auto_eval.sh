#!/bin/bash

if [ -z "$4" ]; then
    echo "Usage: $0 EMAIL_ADDR TASK_NAME GPU_ID TRAIN_ITER_1 [TRAIN_ITER_2 ...]" >&2
    exit
fi
EMAIL_ADDR="$1"
TASK_NAME="$2"
GPU_ID="$3"

if [ "$CAFFE_TEST_ITER" == "" ]; then
    CAFFE_TEST_ITER=1000
fi

if [ "$CAFFE_DUMP_ITER" == "" ]; then
    CAFFE_DUMP_ITER=4
fi

pushd `dirname $0` > /dev/null ; SCRIPT_DIR=`pwd`; popd > /dev/null
CAFFE="$SCRIPT_DIR/caffe.bin"

CAFFEMODEL_EXT="caffemodel"
TESTLOG_EXT="test_log"

TRAIN_ITERS=`echo ${@:4}`

declare -A FIND_ARG=()
for c in $CAFFEMODEL_EXT $TESTLOG_EXT; do
    FIND_ARG[$c]=""
    for k in $TRAIN_ITERS; do
        if [ -z "${FIND_ARG[$c]}" ]; then
            FIND_ARG_k="-name '""*_$k.$c""'"
        else
            FIND_ARG_k="-or -name '""*_$k.$c""'"
        fi
        FIND_ARG[$c]="${FIND_ARG[$c]} $FIND_ARG_k"
    done
    #printf '%12s:   %s\n' "$c"  "${FIND_ARG[$c]}"
done

echo

WDIR=`pwd`

mkdir -p _auto_test

eval "find . ${FIND_ARG[$CAFFEMODEL_EXT]}" | sed -e 's/\.'"$CAFFEMODEL_EXT"'$//' | sort -u > _auto_test/auto_test.avail.list
eval "find . ${FIND_ARG[$TESTLOG_EXT]}" | sed -e 's/\.'"$TESTLOG_EXT"'$//' | sed -e 's/.test_log$//' | sort -u > _auto_test/auto_test.log.list

comm -23 _auto_test/auto_test.avail.list _auto_test/auto_test.log.list > _auto_test/auto_test.pending.list

cat _auto_test/auto_test.pending.list | while read line; do
    line=`readlink -f "$line"`
    echo '************************************************'
    echo "$line"
    echo '************************************************'
    cd "$WDIR"
    if [ ! -e "$line.caffemodel" ]; then
        continue;
    fi
    output_folder=`dirname "$line"`
    task_folder=`dirname "$output_folder"`
    mkdir -p "$line"
    cd "$line"
    if [ -e "$task_folder/train_val-dump.prototxt" ]; then
        echo "*** DUMP:"
        $CAFFE test \
            -model $task_folder/train_val-dump.prototxt -weights $line.caffemodel -iterations $CAFFE_DUMP_ITER -gpu $GPU_ID 2>&1 \
            | grep '\] Batch [0-9*]\,'
    fi
    echo "*** BATCH:"$'\t'$CAFFE_TEST_ITER" iterations"
    $CAFFE test \
        -model $task_folder/train_val.prototxt -weights $line.caffemodel -iterations $CAFFE_TEST_ITER -gpu $GPU_ID 2>&1 \
        | tee $line.test_tmp | grep '\] Batch [0-9]*\,'
    mv $line.test_tmp $line.test_log

    train_iter=`echo $line | sed -e 's/^.*[^0-9]\([0-9]*\)$/\1/'`
    echo "TASK: " $task_folder > report.txt
    echo "TRAIN_ITER: " $train_iter >> report.txt
    echo >> report.txt
    echo $line >> report.txt
    echo >> report.txt
    tail -n10 $line.test_log >> report.txt
    echo >> report.txt
    echo "==============================" >> report.txt
    echo >> report.txt
    cat $line.test_log >> report.txt
    if [ ! -z "$EMAIL_ADDR" ]; then
        mail -s "[$TASK_NAME] $line (caffe:auto_eval)" "$EMAIL_ADDR" < report.txt
    fi

    cd "$WDIR"
done

