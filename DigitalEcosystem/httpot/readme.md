HT screening of TPOT models. To reproduce the results:

1. Execute `setup_jobs.py`. This will generate an `httopt_runs` directory.
2. Copy the `httpot_runs` directory to a cluster running Torque (Slurm also is fine if the submission script is
   modified). Additionally, copy the `tpot_config.py`, `requirements.txt`, `job.pbs`, `run_jobs.py`, `sub_jobs.sh`,
   and `result_header.csv` files to the newly-created `httpot_runs` directory on the cluster.
3. SSH into the cluster, and navigate to the newly-created `httpot_runs` directory.
4. Update $BASE_DIR inside `job.pbs` with the path to the `httpot_runs` directory on the cluster.
5. Create a new virtual environment with `virtualenv .env && .env/bin/pip install -r requirements.txt`
6. Run the `sub_jobs.sh` script.
7. After the jobs have run (this will take at least 1 hour, depending on how many compute nodes are available), the runs
   can be summarized by running `gather.sh`.
