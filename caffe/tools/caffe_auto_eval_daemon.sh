#!/bin/bash

pushd `dirname $0` > /dev/null ; SCRIPT_DIR=`pwd`; popd > /dev/null
while true; do
    $SCRIPT_DIR/caffe_auto_eval.sh "$@"
    date
    echo sleep for 20min
    echo
    sleep 1200
done

