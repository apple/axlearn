#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name rh_test
#SBATCH --output=test_logs/%x_%j.out
#SBATCH --cpus-per-task 127

srun -l setup_node.sh /fsx/huilgolr/apr-artifacts-shrihari/

SUITE_ARG=${1:-"all"}

mkdir -p test_logs
if [ $SUITE_ARG = "all" ]; then
    for suite in "presubmit" "12b" "50b" "150b"; do
        mkdir -p test_logs/${suite}
        srun -l unit_test.sh unit $suite 2>&1 | tee test_logs/${suite}/unit.log
        mv test-results/testing.xml test_logs/${suite}/unit.xml
        srun -l unit_test.sh integ $suite 2>&1 | tee test_logs/${suite}/integ.log
        mv test-results/testing.xml test_logs/${suite}/integ.xml
        if [ "$suite" = "presubmit" ]; then
            srun -l unit_test.sh 150b $suite 2>&1 | tee test_logs/${suite}/150b.log
        fi
    done
else
    suite=$SUITE_ARG
    mkdir -p test_logs/${suite}
    srun -l unit_test.sh unit $suite 2>&1 | tee test_logs/${suite}/unit.log
    srun -l unit_test.sh integ $suite 2>&1 | tee test_logs/${suite}/integ.log
    if [ "$SUITE_ARG" = "presubmit" ]; then
        srun -l unit_test.sh 150b $suite 2>&1 | tee test_logs/${suite}/150b.log
    fi
fi
