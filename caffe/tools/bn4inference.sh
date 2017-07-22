#!/bin/bash

AVE_MASS=1e6    # use a large number
DECAY_VALUE=`echo "print $AVE_MASS / ( $AVE_MASS + 1. ) " | python `

PROTOTXT_PATH=$1
LEARN_WEIGHT_PATH=$2
INFERENCE_WEIGHT_PATH=$3
NUM_ITER=$4
GPU_ID=$5

if [ "$1" == "" ]; then
    echo `basename $0` PROTOTXT_PATH
    echo `basename $0` PROTOTXT_PATH LEARN_WEIGHT_PATH INFERENCE_WEIGHT_PATH NUM_ITER GPU_ID
    echo "Remark: it works only for refactored BN with running average layers"
    exit 0
fi

if [ "$NUM_ITER" == "" ]; then
    NUM_ITER=1000
fi
if [ "$GPU_ID" == "" ]; then
    GPU_ID=0
fi

PROTOTXT_PATH_INFERENCE=$PROTOTXT_PATH.bn4inference.TMP_FILE.NOT_FOR_EVALUATION

#cat $PROTOTXT_PATH | sed -e 's/update_test:[ \t]*false//g' | \
#        sed -e 's/mass:[ \t]*[0-9e\-]*//g' | \
#        sed -e 's/reset_history:[ \t]*false//g' | \
#        sed -e 's/reset_history:[ \t]*true//g' | \
#        sed -e 's/AUTO_NORM/NORM/g' | \
#        sed -e 's/AUTO_CORRECT/INFERENCE/g' | \
#        sed -e 's/running_average_param[ \t]*{/\0 mass: '$AVE_MASS' reset_history: true /g' | \
#        sed -e 's/phase:[ \t]*TEST/phase: DUMMY/g' | \
#        sed -e 's/phase:[ \t]*TRAIN/phase: TEST/g' | \
#        sed -e 's/use_global_stats:[ \t]*[a-z0-9]*[ \t]*/ /g' | \
#        sed -e 's/moving_average_fraction:[ \t]*[0-9\.]*[ \t]*/ /g' | \
#        sed -e 's/batch_norm_param[ \t]*{/\0 use_global_stats: false  moving_average_fraction: '"$DECAY_VALUE"'  /g' | \
#        cat > $PROTOTXT_PATH_INFERENCE

cat $PROTOTXT_PATH | sed -e 's/update_test:[ \t]*false//g' | \
        sed -e 's/mass:[ \t]*[0-9e\-]*//g' | \
        sed -e 's/reset_history:[ \t]*false//g' | \
        sed -e 's/reset_history:[ \t]*true//g' | \
        sed -e 's/AUTO_NORM/NORM/g' | \
        sed -e 's/AUTO_CORRECT/INFERENCE/g' | \
        sed -e 's/running_average_param[ \t]*{/\0 mass: '$AVE_MASS' reset_history: true /g' | \
        sed -e 's/phase:[ \t]*TEST/phase: DUMMY/g' | \
        sed -e 's/phase:[ \t]*TRAIN/phase: TEST/g' | \
        sed -e 's/$/__LINEBREAK__/' | tr '\n' ' ' | sed -e 's/batch_norm_param[ \t]*{[^}]*}//g' | sed -e 's/__LINEBREAK__/\n/g' | \
        sed -e 's/type[ \t]*:[ \t]*"BatchNorm"/\0 \n batch_norm_param { use_global_stats: false  moving_average_fraction: '"$DECAY_VALUE"' } /g' | \
        cat > $PROTOTXT_PATH_INFERENCE

echo "PROTOTXT at: $PROTOTXT_PATH_INFERENCE"
if [ "$3" == "" ]; then
    echo caffe.bin test -model '"'"$PROTOTXT_PATH_INFERENCE"'" -weights LEARN_WEIGHT_PATH -output_weights INFERENCE_WEIGHT_PATH ...'
    exit 0
fi

caffe.bin test -model "$PROTOTXT_PATH_INFERENCE" -weights "$LEARN_WEIGHT_PATH" -output_weights "$INFERENCE_WEIGHT_PATH" -iterations $NUM_ITER -gpu $GPU_ID

