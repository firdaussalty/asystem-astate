#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "astate/astate.h"

namespace py = pybind11;

PYBIND11_MODULE(astate_cpp, m) {
    m.doc() = "Astate C++ Module";

    // Export TensorTableType enum
    py::enum_<astate::TensorTableType>(m, "TensorTableType")
        .value("IN_MEMORY", astate::TensorTableType::IN_MEMORY)
        .value("REMOTE", astate::TensorTableType::REMOTE)
        .export_values();

    // Export ARole enum
    py::enum_<astate::ARole>(m, "ARole")
        .value("TRAINING", astate::ARole::TRAINING)
        .value("INFERENCE", astate::ARole::INFERENCE)
        .export_values();

    // Export AParallelConfig struct
    py::class_<astate::AParallelConfig>(m, "AParallelConfig")
        .def(py::init<>())
        .def(
            py::init<
                astate::ARole,
                int32_t,
                int32_t,
                int32_t,
                int32_t,
                int32_t,
                int32_t,
                int32_t,
                int32_t,
                int32_t,
                int32_t,
                int32_t,
                int32_t,
                int32_t,
                int32_t>(),
            py::arg("role"),
            py::arg("role_size"),
            py::arg("role_rank"),
            py::arg("dp_size") = 1,
            py::arg("dp_rank") = 0,
            py::arg("tp_size") = 1,
            py::arg("tp_rank") = 0,
            py::arg("pp_size") = 1,
            py::arg("pp_rank") = 0,
            py::arg("cp_size") = 1,
            py::arg("cp_rank") = 0,
            py::arg("ep_size") = 1,
            py::arg("ep_rank") = 0,
            py::arg("etp_size") = 1,
            py::arg("etp_rank") = 0)
        .def_readwrite("role", &astate::AParallelConfig::role)
        .def_readwrite("role_size", &astate::AParallelConfig::role_size)
        .def_readwrite("role_rank", &astate::AParallelConfig::role_rank)
        .def_readwrite("dp_size", &astate::AParallelConfig::dp_size)
        .def_readwrite("dp_rank", &astate::AParallelConfig::dp_rank)
        .def_readwrite("tp_size", &astate::AParallelConfig::tp_size)
        .def_readwrite("tp_rank", &astate::AParallelConfig::tp_rank)
        .def_readwrite("pp_size", &astate::AParallelConfig::pp_size)
        .def_readwrite("pp_rank", &astate::AParallelConfig::pp_rank)
        .def_readwrite("cp_size", &astate::AParallelConfig::cp_size)
        .def_readwrite("cp_rank", &astate::AParallelConfig::cp_rank)
        .def_readwrite("ep_size", &astate::AParallelConfig::ep_size)
        .def_readwrite("ep_rank", &astate::AParallelConfig::ep_rank)
        .def_readwrite("etp_size", &astate::AParallelConfig::etp_size)
        .def_readwrite("etp_rank", &astate::AParallelConfig::etp_rank)
        .def("is_training", &astate::AParallelConfig::IsTraining)
        .def("is_inference", &astate::AParallelConfig::IsInference)
        .def("to_string", &astate::AParallelConfig::ToString)
        .def("__str__", &astate::AParallelConfig::ToString)
        .def("__repr__", &astate::AParallelConfig::ToString);

    // Export ShardedKey struct
    py::class_<astate::ShardedKey>(m, "ShardedKey")
        .def(py::init<>())
        .def_readwrite("key", &astate::ShardedKey::key)
        .def_readwrite("globalShape", &astate::ShardedKey::global_shape)
        .def_readwrite("globalOffset", &astate::ShardedKey::global_offset);

    // Export TorchTensorMeta struct
    py::class_<astate::TorchTensorMeta>(m, "TorchTensorMeta")
        .def(py::init<>())
        .def_readwrite("dtype", &astate::TorchTensorMeta::dtype)
        .def_readwrite("size", &astate::TorchTensorMeta::size)
        .def_readwrite("device", &astate::TorchTensorMeta::device);

    // Export abstract TensorTable interface with proper shared_ptr handling
    py::class_<astate::TensorTable, std::shared_ptr<astate::TensorTable>>(m, "TensorTable")
        .def("put", &astate::TensorTable::Put, "Store a single tensor to the table")
        .def("get", &astate::TensorTable::Get, "Retrieve a single tensor from the table (in-place update)")
        .def("get_tensor", &astate::TensorTable::GetTensor, "Get tensor object based on metadata")
        .def("multi_put", &astate::TensorTable::MultiPut, "Store multiple tensors in batch")
        .def("multi_get", &astate::TensorTable::MultiGet, "Retrieve multiple tensors in batch (in-place update)")
        .def(
            "multi_get_tensor",
            &astate::TensorTable::MultiGetTensor,
            "Get multiple tensor objects based on metadata list")
        .def("complete", &astate::TensorTable::Complete, "Complete all operations for the specified sequence ID")
        .def(
            "scan_tensor_meta",
            &astate::TensorTable::ScanTensorMeta,
            "Scan all tensor metadata for the specified sequence ID")
        .def("name", &astate::TensorTable::Name, "Get the name of the tensor table");

    // Export TensorStorage with proper return value policy
    py::class_<astate::ATensorStorage, std::shared_ptr<astate::ATensorStorage>>(m, "TensorStorage")
        .def_static("create_tensor_storage", &astate::CreateATensorStorage, "Create a tensor storage instance")
        .def(
            "register_table",
            &astate::ATensorStorage::RegisterTable,
            "Create a tensor table with specified type, name and parallel "
            "configuration");
}
