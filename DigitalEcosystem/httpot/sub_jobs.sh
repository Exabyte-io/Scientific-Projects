#!/bin/bash


cwd=$( pwd )
for trainfile in $(find all_rows/all_cols -name "train.parquet"); do
    job_dir=$(dirname $trainfile)

    cp -v $cwd/job.pbs $job_dir/job.pbs
    cp -v $cwd/run_job.py $job_dir/run_job.py

    cd $job_dir
    qsub job.pbs

    cd $cwd
done
