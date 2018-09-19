##!/usr/bin/env bash
#
#PBS -N demo_with_container
#
export SINGULARITY_CACHEDIR="/beegfs/work/.singularity"

module load devel/singularity/2.5.2

singularity exec --nv \
	shub://schanzel/tensorflow-keras-py3:gpu \
	python ./work/bwhpc-container-demo/binary_classifier_lstm.py

