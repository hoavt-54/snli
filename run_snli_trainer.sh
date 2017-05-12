#!/bin/sh
#
# Select Q
#$ -q albert.q
#
# Your job name
#$ -N nli_BiMPM_dep_v800_beginning
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
~/hoavt/hoa_env/snli/bin/python SentenceMatchTrainer.py
