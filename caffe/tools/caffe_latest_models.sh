#!/bin/bash

function gen_cur_list() {
    if [ ! -f "$TMP_LIST_PATH.cur_base" ]; then 
        return;
    fi
    CUR_BASE=`cat $TMP_LIST_PATH.cur_base`
    MAX_ITER=`cat $TMP_LIST_PATH.cur | sort -n | tail -n1`
    echo "$CUR_BASE""$MAX_ITER.caffemodel"
}

TMP_LIST_PATH=`pwd | md5sum | awk '{print $1}'``date | md5sum | awk '{print $1}'`
TMP_LIST_PATH="/tmp/caffe_wipe_cache_"$TMP_LIST_PATH

if [ "$1" == "" ]; then
    find . -name '*.caffemodel' | sort > $TMP_LIST_PATH.all
else
    find "$@" -name '*.caffemodel' | sort > $TMP_LIST_PATH.all
fi


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

rm -f $TMP_LIST_PATH.*

