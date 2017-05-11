#!/bin/sh
#
# Select Q
#$ -q albert.q
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
#source ~/hoavt/hoa_env/snli/bin/activate
source /users/ud2017/hoavt/hoa_env/tensorflow0.12-gpu/bin/activate
~/hoavt/hoa_env/snli/bin/python test_gpu.python
