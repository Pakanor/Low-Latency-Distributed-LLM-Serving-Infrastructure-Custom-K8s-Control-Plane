#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "page_allocator.hpp"
#include "paged_kernels.h"

namespace py = pybind11;

static void register_paged_attention(py::module_& m) {
    m.def("paged_attention", [](
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        py::array_t<float, py::array::c_style | py::array::forcecast> key_cache,
        py::array_t<float, py::array::c_style | py::array::forcecast> value_cache,
        int64_t layer_idx,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> gather_slots,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> attention_mask,
        float scaling
    ) -> py::array_t<float> {
        auto q_buf  = query.request();
        auto k_buf  = key_cache.request();
        auto v_buf  = value_cache.request();
        auto s_buf  = gather_slots.request();
        auto m_buf  = attention_mask.request();

        if (q_buf.ndim != 4)
            throw std::runtime_error("query must be 4-D (1, heads, seq, dim)");
        if (k_buf.ndim != 5)
            throw std::runtime_error("key_cache must be 5-D (layers, blocks, block_size, heads, dim)");
        if (s_buf.ndim != 1)
            throw std::runtime_error("gather_slots must be 1-D");
        if (m_buf.ndim != 4)
            throw std::runtime_error("attention_mask must be 4-D (1, 1, query, context)");

        PagedAttentionDims dims{
            static_cast<int32_t>(q_buf.shape[1]),
            static_cast<int32_t>(q_buf.shape[2]),
            static_cast<int32_t>(q_buf.shape[3]),
            static_cast<int32_t>(k_buf.shape[0]),
            static_cast<int32_t>(k_buf.shape[1]),
            static_cast<int32_t>(k_buf.shape[2]),
            static_cast<int32_t>(k_buf.shape[3]),
            static_cast<int32_t>(s_buf.shape[0]),
        };

        if (dims.num_query_heads % dims.num_kv_heads != 0)
            throw std::runtime_error("num_query_heads must be a multiple of num_kv_heads for GQA");
        if (layer_idx < 0 || layer_idx >= dims.num_layers)
            throw std::runtime_error("layer_idx out of range");

        py::array_t<float> output({1, dims.num_query_tokens, dims.num_query_heads, dims.head_dim});
        py::array_t<float> scratch({dims.num_query_tokens, dims.num_query_heads, dims.num_context_tokens});

        auto out_buf     = output.request();
        auto scratch_buf = scratch.request();

        {
            py::gil_scoped_release release;
            paged_attention(
                static_cast<float*>(q_buf.ptr),
                static_cast<float*>(k_buf.ptr),
                static_cast<float*>(v_buf.ptr),
                static_cast<int32_t>(layer_idx),
                dims,
                static_cast<int32_t*>(s_buf.ptr),
                static_cast<uint8_t*>(m_buf.ptr),
                scaling,
                static_cast<float*>(out_buf.ptr),
                static_cast<float*>(scratch_buf.ptr),
                nullptr
            );
        }

        return output;
    }, py::arg("query"), py::arg("key_cache"), py::arg("value_cache"),
       py::arg("layer_idx"), py::arg("gather_slots"), py::arg("attention_mask"),
       py::arg("scaling"));
}

PYBIND11_MODULE(llm_allocator_cpp, m) {
    m.doc() = "C++ High-Performance DRAM Page Allocator for KV-Cache with Swapping";

    py::class_<llm_infra::PageAllocator>(m, "PageAllocator")
        .def(py::init<size_t, size_t, size_t>(),
             py::arg("total_blocks"),
             py::arg("block_size"),
             py::arg("total_cpu_blocks") = 32)
        .def("allocate_block", &llm_infra::PageAllocator::allocate_block, py::call_guard<py::gil_scoped_release>())
        .def("allocate_blocks", &llm_infra::PageAllocator::allocate_blocks, py::arg("count"), py::call_guard<py::gil_scoped_release>())
        .def("share_block", &llm_infra::PageAllocator::share_block, py::arg("block_id"), py::call_guard<py::gil_scoped_release>())
        .def("free_block", &llm_infra::PageAllocator::free_block, py::arg("block_id"), py::call_guard<py::gil_scoped_release>())
        .def("swap_out", &llm_infra::PageAllocator::swap_out, py::arg("gpu_block_id"), py::call_guard<py::gil_scoped_release>())
        .def("swap_in", &llm_infra::PageAllocator::swap_in, py::arg("gpu_block_id"), py::call_guard<py::gil_scoped_release>())
        .def("get_cpu_block_id", &llm_infra::PageAllocator::get_cpu_block_id, py::arg("gpu_block_id"), py::call_guard<py::gil_scoped_release>())
        .def("release_swapped_block", &llm_infra::PageAllocator::release_swapped_block, py::arg("gpu_block_id"), py::call_guard<py::gil_scoped_release>())
        .def("get_host_ptr", &llm_infra::PageAllocator::get_host_ptr, py::arg("gpu_block_id"), py::call_guard<py::gil_scoped_release>())
        .def("get_device_ptr", &llm_infra::PageAllocator::get_device_ptr, py::arg("gpu_block_id"), py::call_guard<py::gil_scoped_release>())
        .def("is_zero_copy", &llm_infra::PageAllocator::is_zero_copy)
        .def("get_block_size", &llm_infra::PageAllocator::get_block_size)
        .def("get_num_free_blocks", &llm_infra::PageAllocator::get_num_free_blocks, py::call_guard<py::gil_scoped_release>())
        .def("get_num_free_cpu_blocks", &llm_infra::PageAllocator::get_num_free_cpu_blocks, py::call_guard<py::gil_scoped_release>())
        .def("get_total_blocks", &llm_infra::PageAllocator::get_total_blocks)
        .def("get_utilization_percentage", &llm_infra::PageAllocator::get_utilization_percentage, py::call_guard<py::gil_scoped_release>());

    py::class_<llm_infra::SequenceBlockTable>(m, "SequenceBlockTable")
        .def(py::init<llm_infra::PageAllocator&>(), py::arg("allocator"), py::keep_alive<1, 2>())
        .def("append_token", &llm_infra::SequenceBlockTable::append_token, py::arg("allocator"), py::call_guard<py::gil_scoped_release>())
        .def("append_block", &llm_infra::SequenceBlockTable::append_block, py::arg("block_id"))
        .def("release", [](llm_infra::SequenceBlockTable& self, llm_infra::PageAllocator& alloc) {
            self.release(alloc);
        }, py::arg("allocator"), py::call_guard<py::gil_scoped_release>())
        .def("replace_block", &llm_infra::SequenceBlockTable::replace_block,
             py::arg("old_block_id"), py::arg("new_block_id"))
        .def("get_physical_blocks", &llm_infra::SequenceBlockTable::get_physical_blocks)
        .def("get_tokens_count", &llm_infra::SequenceBlockTable::get_tokens_count);

    register_paged_attention(m);
}
