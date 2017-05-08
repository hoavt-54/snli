#!/bin/sh
#
# Select Q
#$ -q main.q
#
# Your job name
#$ -N test_gpu
#
# Use current working directory
#$ -cwd
#
# Join stdout and stderr
#$ -j y
#
# Run job through bash shell
#$ -S /bin/bash
#

# Activate virtualenv
#source /opt/deeplearningenv/bin/activate
source ~/hoavt/hoa_env/snli/bin/activate
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 ~/hoavt/hoa_env/snli/bin/python test_gpu.py
