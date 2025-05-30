id=$(date +"%Y%m%d_%H%M%S")
sbatch --exclusive -J rh_test_presubmit test.slurm presubmit $id
sbatch --exclusive -J rh_test_12b test.slurm 12b $id
sbatch --exclusive -J rh_test_50b test.slurm 50b $id
sbatch --exclusive -J rh_test_150b test.slurm 150b $id