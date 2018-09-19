##!/usr/bin/env bash
#
#PBS -N demo_with_modules
#

module load cs/keras/2.1.0-tensorflow-1.4-python-3.5
python ./work/bwhpc-container-demo/binary_classifier_lstm.py

