#!/bin/bash

# get the file directory
dir=$(cd $(dirname "$0"); pwd)

# get parent directory
parentdir=$(dirname "$dir")


step=${1:-1}

# if step equals 1
if [ "$step" -eq 1 ]; then
    # step 1: prepare data
    echo "step 1: create semantic data"
    PYTHONPATH=$parentdir:$dir python3 "$dir"/create_data.py

    # step++
    step=2
fi


# if step equals 2
if [ "$step" -eq 2 ]; then
    # step 2: create audio files
    echo "step 2: create audio files"
    PYTHONPATH=$parentdir:$dir python3 "$dir"/create_wavs.py

    # step++
    step=3
fi


# if step equals 3
if [ "$step" -eq 3 ]; then
    # step 3: preprocess traning data
    echo "step 3: preprocess train data"
    PYTHONPATH=$parentdir:$dir python3 "$dir"/process.py --mode=prepare
    PYTHONPATH=$parentdir:$dir python3 "$dir"/process.py --mode=prepare2

    # step++
    step=4
fi


# if step equals 4
if [ "$step" -eq 4 ]; then
    # step 4: train model
    echo "step 4: train model"
    PYTHONPATH=$parentdir:$dir python3 "$dir"/process.py --mode=train

    # step++
    step=5
fi


# if step equals 5
if [ "$step" -eq 5 ]; then
    # step 5: test model
    echo "step 5: test model"
    PYTHONPATH=$parentdir:$dir python3 "$dir"/process.py --mode=test

    # step++
    step=6
fi

echo "finished model training"