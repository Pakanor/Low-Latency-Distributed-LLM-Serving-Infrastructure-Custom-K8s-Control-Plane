#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "page_allocator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(llm_allocator_cpp, m) {
    m.doc() = "C++ High-Performance DRAM Page Allocator for KV-Cache";

    py::class_<llm_infra::PageAllocator>(m, "PageAllocator")
        .def(py::init<size_t, size_t>(), py::arg("total_blocks"), py::arg("block_size"))
        .def("allocate_block", &llm_infra::PageAllocator::allocate_block, py::call_guard<py::gil_scoped_release>())
        .def("allocate_blocks", &llm_infra::PageAllocator::allocate_blocks, py::arg("count"), py::call_guard<py::gil_scoped_release>())
        .def("share_block", &llm_infra::PageAllocator::share_block, py::arg("block_id"), py::call_guard<py::gil_scoped_release>())
        .def("free_block", &llm_infra::PageAllocator::free_block, py::arg("block_id"), py::call_guard<py::gil_scoped_release>())
        .def("get_block_size", &llm_infra::PageAllocator::get_block_size)
        .def("get_num_free_blocks", &llm_infra::PageAllocator::get_num_free_blocks, py::call_guard<py::gil_scoped_release>())
        .def("get_total_blocks", &llm_infra::PageAllocator::get_total_blocks)
        .def("get_utilization_percentage", &llm_infra::PageAllocator::get_utilization_percentage, py::call_guard<py::gil_scoped_release>());

    py::class_<llm_infra::SequenceBlockTable>(m, "SequenceBlockTable")
        .def(py::init<llm_infra::PageAllocator&>(), py::arg("allocator"), py::keep_alive<1, 2>())
        .def("append_token", &llm_infra::SequenceBlockTable::append_token, py::arg("allocator"), py::call_guard<py::gil_scoped_release>())
        .def("append_block", &llm_infra::SequenceBlockTable::append_block, py::arg("block_id"))
        .def("release", [](llm_infra::SequenceBlockTable& self, llm_infra::PageAllocator& alloc) {
            self.release(alloc);
        }, py::arg("allocator"), py::call_guard<py::gil_scoped_release>())
        .def("get_physical_blocks", &llm_infra::SequenceBlockTable::get_physical_blocks)
        .def("get_tokens_count", &llm_infra::SequenceBlockTable::get_tokens_count);
}