# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum

from cutlass_library import *

#
#   Extend cutlass library with custom types, and missing values
#


class APHRODITEDataType(enum.Enum):
    u4b8 = enum_auto()
    u8b128 = enum_auto()


class MixedInputKernelScheduleType(enum.Enum):
    TmaWarpSpecialized = enum_auto()
    TmaWarpSpecializedPingpong = enum_auto()
    TmaWarpSpecializedCooperative = enum_auto()


APHRODITEDataTypeNames: dict[APHRODITEDataType | DataType, str] = {
    **DataTypeNames,  # type: ignore
    **{
        APHRODITEDataType.u4b8: "u4b8",
        APHRODITEDataType.u8b128: "u8b128",
    },
}

APHRODITEDataTypeTag: dict[APHRODITEDataType | DataType, str] = {
    **DataTypeTag,  # type: ignore
    **{
        APHRODITEDataType.u4b8: "cutlass::aphrodite_uint4b8_t",
        APHRODITEDataType.u8b128: "cutlass::aphrodite_uint8b128_t",
    },
}

APHRODITEDataTypeSize: dict[APHRODITEDataType | DataType, int] = {
    **DataTypeSize,  # type: ignore
    **{
        APHRODITEDataType.u4b8: 4,
        APHRODITEDataType.u8b128: 8,
    },
}

APHRODITEDataTypeAPHRODITEScalarTypeTag: dict[APHRODITEDataType | DataType, str] = {
    APHRODITEDataType.u4b8: "aphrodite::kU4B8",
    APHRODITEDataType.u8b128: "aphrodite::kU8B128",
    DataType.u4: "aphrodite::kU4",
    DataType.u8: "aphrodite::kU8",
    DataType.s4: "aphrodite::kS4",
    DataType.s8: "aphrodite::kS8",
    DataType.f16: "aphrodite::kFloat16",
    DataType.bf16: "aphrodite::kBfloat16",
}

APHRODITEDataTypeTorchDataTypeTag: dict[APHRODITEDataType | DataType, str] = {
    DataType.u8: "torch::headeronly::ScalarType::Byte",
    DataType.s8: "torch::headeronly::ScalarType::Char",
    DataType.e4m3: "torch::headeronly::ScalarType::Float8_e4m3fn",
    DataType.s32: "torch::headeronly::ScalarType::Int",
    DataType.f16: "torch::headeronly::ScalarType::Half",
    DataType.bf16: "torch::headeronly::ScalarType::BFloat16",
    DataType.f32: "torch::headeronly::ScalarType::Float",
}

APHRODITEKernelScheduleTag: dict[MixedInputKernelScheduleType | KernelScheduleType, str] = {
    **KernelScheduleTag,  # type: ignore
    **{
        MixedInputKernelScheduleType.TmaWarpSpecialized: "cutlass::gemm::KernelTmaWarpSpecialized",  # noqa: E501
        MixedInputKernelScheduleType.TmaWarpSpecializedPingpong: "cutlass::gemm::KernelTmaWarpSpecializedPingpong",  # noqa: E501
        MixedInputKernelScheduleType.TmaWarpSpecializedCooperative: "cutlass::gemm::KernelTmaWarpSpecializedCooperative",  # noqa: E501
    },
}
