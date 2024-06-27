"""
A script to split up test files into partitions for GitHub Actions

This allows us to run 10 partitions of tests at the same time on different
GitHub action runners.
"""
import glob
import json

partitions = 10
test_files = glob.glob("axlearn/**/*_test.py")
# TODO(samos123): Figure out why clip is broken
excluded_tests = ["axlearn/vision/clip_test.py"]
test_files = [f for f in test_files if f not in excluded_tests]

# Split the test files into partitions.
paritioned_test_files: list[str] = [
    " ".join(test_files[i : i + partitions]) for i in range(0, len(test_files), partitions)
]

print(
    json.dumps(
        {
            "pytest_files": paritioned_test_files,
        }
    )
)
