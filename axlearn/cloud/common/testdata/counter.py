# Copyright © 2023 Apple Inc.

"""A dummy script that spawns a subprocess which keeps updating a counter."""

import shlex
import subprocess
import sys
import time


def _child(path: str):
    print(f"emitting to {path}")
    for i in range(100):
        with open(path, "w", encoding="utf-8") as f:
            f.seek(0, 0)
            print(f"incrementing to {i}")
            f.write(str(i))
            f.flush()
        time.sleep(0.1)


if __name__ == "__main__":
    output_path, parent_or_child = sys.argv[1], sys.argv[2]

    if parent_or_child == "parent":
        # pylint: disable-next=consider-using-with
        p = subprocess.Popen(
            shlex.split(f"python3 {__file__} {output_path} child"), start_new_session=True
        )
        print("returncode:", p.wait())
    else:
        assert parent_or_child == "child"
        _child(output_path)
