# Usage:
# $ python axlearn/profiler/setup.py build_ext --inplace
# $ python axlearn/profiler/profiler_test.py
# $ pprof --text my_heap_profile.0001.heap
# File: python3.10
# Type: space
# Showing nodes accounting for 16B, 100% of 16B total
#       flat  flat%   sum%        cum   cum%
#        16B   100%   100%        16B   100%  std::allocator_traits::allocate (inline)
#          0     0%   100%        16B   100%  PyCFunction_Call

import profiler

# Start profiling to a file
profiler.start_profiling("my_heap_profile")

# Your Python code here
x = sorted(list(range(100000, 0, -1)))

# Optionally, dump intermediate profile data
profiler.dump_profile("checkpoint")

# Stop profiling
profiler.stop_profiling()

print(x[3])
