##!/usr/bin/env bash
#
#PBS -N demo_with_container
#
export SINGULARITY_CACHEDIR="/beegfs/work/.singularity"

module load devel/singularity/2.5.2

singularity run --nv --pwd /opt/code shub://schanzel/bwhpc-container-demo

