sbatch --exclusive -J rh_test_presubmit test.slurm presubmit
sbatch --exclusive -J rh_test_12b test.slurm 12b
sbatch --exclusive -J rh_test_50b test.slurm 50b
sbatch --exclusive -J rh_test_150b test.slurm 150b