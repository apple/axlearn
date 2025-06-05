id=$(date +"%Y%m%d_%H%M%S")

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

sbatch -W --exclusive -J rh_test_presubmit test.slurm presubmit $id &
sbatch -W --exclusive -J rh_test_12b test.slurm 12b $id &
sbatch -W --exclusive -J rh_test_50b test.slurm 50b $id &
sbatch -W --exclusive -J rh_test_150b test.slurm 150b $id &
echo "All tests launched with ID: $id"
wait
echo "All tests finished"
TEST_LOGDIR=test_logs/$id
# TEST_LOGDIR=test_logs/20250604_180636
for suite in "presubmit" "12b" "50b" "150b"; do
    summary $suite
done