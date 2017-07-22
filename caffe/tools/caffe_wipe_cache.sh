#!/bin/bash

function gen_cur_list() {
    if [ ! -f "$TMP_LIST_PATH.cur_base" ]; then 
        return;
    fi
    CUR_BASE=`cat $TMP_LIST_PATH.cur_base`
    cat $TMP_LIST_PATH.cur | sort -n | sed -e '$ d' | sort > $TMP_LIST_PATH.cur.1
    comm -23 $TMP_LIST_PATH.cur.1 $TMP_LIST_PATH.keep | sort -n > $TMP_LIST_PATH.cur.2
    cat $TMP_LIST_PATH.cur.2 | while read line; do
        ls "$CUR_BASE""$line".* >>  $TMP_LIST_PATH.wipe
#echo "$CUR_BASE""$line.caffemodel" >>  $TMP_LIST_PATH.wipe
#echo "$CUR_BASE""$line.solverstate" >>  $TMP_LIST_PATH.wipe
    done
}

TARGET_DIR=$1
if [ "$TARGET_DIR" == "" ]; then
    TARGET_DIR=`pwd`
fi
TARGET_DIR=`readlink -f "$TARGET_DIR"`

TMP_LIST_PATH=`echo "$TARGET_DIR" | md5sum | awk '{print $1}'``date | md5sum | awk '{print $1}'`
TMP_LIST_PATH="/tmp/caffe_wipe_cache_"$TMP_LIST_PATH

rm -f $TMP_LIST_PATH.keep.0
touch $TMP_LIST_PATH.keep.0
args=("$@")
ELEMENTS=${#args[@]}
for (( i=0;i<$ELEMENTS;i++)); do
    arg_i="${args[${i}]}";
    echo "$arg_i" >> $TMP_LIST_PATH.keep.0
done
cat $TMP_LIST_PATH.keep.0 | sort > $TMP_LIST_PATH.keep

find "$TARGET_DIR" -name '*.caffemodel' | sort > $TMP_LIST_PATH.all

cat $TMP_LIST_PATH.all | sed -e 's/^.*_\([0-9]*\).caffemodel$/\1/' > $TMP_LIST_PATH.iter

rm -f $TMP_LIST_PATH.wipe
touch $TMP_LIST_PATH.wipe
rm -f $TMP_LIST_PATH.cur_base
CUR_BASE=""
cat $TMP_LIST_PATH.all | while read line; do
    NEW_BASE=`echo "$line" | sed -e 's/^\(.*_\)[0-9]*.caffemodel$/\1/'`
    CUR_ITER=`echo "$line" | sed -e 's/^.*_\([0-9]*\).caffemodel$/\1/'`
    if [ "$CUR_BASE" != "$NEW_BASE" ]; then
        gen_cur_list
        rm -f $TMP_LIST_PATH.cur
        CUR_BASE="$NEW_BASE"
        echo "$CUR_BASE" > $TMP_LIST_PATH.cur_base
    fi
    echo $CUR_ITER >> $TMP_LIST_PATH.cur
done
gen_cur_list

cat $TMP_LIST_PATH.wipe | while read FN; do
    if [ -w "$FN" ]; then
        echo "$FN"
    fi
done

rm -f $TMP_LIST_PATH.*

