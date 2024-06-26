"""
A script to define split up the test files into partitions
"""
import glob
import json

partitions = 10
test_files = glob.glob("axlearn/**/*_test.py")
# Split the test files into partitions
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
