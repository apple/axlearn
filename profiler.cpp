#include <gperftools/heap-profiler.h>
#include <pybind11/pybind11.h>

void start_profiling(const char* filename) {
    HeapProfilerStart(filename);
}

void stop_profiling() {
    HeapProfilerStop();
}

void dump_profile(const char* reason) {
    HeapProfilerDump(reason);
}

namespace py = pybind11;

PYBIND11_MODULE(profiler, m) {
    m.def("start_profiling", &start_profiling, "Start the heap profiler");
    m.def("stop_profiling", &stop_profiling, "Stop the heap profiler");
    m.def("dump_profile", &dump_profile, "Dump heap profiler data");
}
