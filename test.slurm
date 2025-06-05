#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name rh_test
#SBATCH --output=test_logs/%x_%j.out
#SBATCH --cpus-per-task 127

srun -l setup_node.sh /fsx/huilgolr/may-artifacts/

SUITE_ARG=${1:-"all"}
TID=${2:-$SLURM_JOB_ID}

TEST_LOGDIR="test_logs/$TID/"
mkdir -p $TEST_LOGDIR

function summary() {
    echo "---------------------------------"
    suite=$1
    num_passed=$(grep -re 'PASSED' $TEST_LOGDIR/$suite/*.log | wc -l)
    num_failed=$(grep -re 'FAILED' $TEST_LOGDIR/$suite/*.log | wc -l)
    total_num=$(($num_passed + $num_failed))
    echo "Test suite: $suite, Total tests $total_num"
    echo "Number of tests passed: $num_passed"
    echo "Number of tests failed: $num_failed"
    if [ $num_failed -gt 0 ]; then
        echo "Failed tests:"
        grep -hre 'FAILED' $TEST_LOGDIR/$suite/*.log
    fi
}

function run_suite() {
    suite=$1
    echo "Running test suite: $suite"
    mkdir -p $TEST_LOGDIR/${suite}
    srun -l unit_test.sh unit $suite $TEST_LOGDIR 2>&1 | tee $TEST_LOGDIR/${suite}/unit.log
    srun -l unit_test.sh integ $suite $TEST_LOGDIR 2>&1 | tee $TEST_LOGDIR/${suite}/integ.log
    if [ "$suite" = "150b" ]; then
        srun -l unit_test.sh 150b ${suite} $TEST_LOGDIR 2>&1 | tee $TEST_LOGDIR/${suite}/unit_and_integ.log
    fi
    summary $suite
}

if [ $SUITE_ARG = "all" ]; then
    for suite in "presubmit" "12b" "50b" "150b"; do
        run_suite $suite
    done
else
    run_suite $SUITE_ARG
fi
