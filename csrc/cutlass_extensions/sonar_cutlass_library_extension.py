# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum

from cutlass_library import (
    DataType,
    DataTypeNames,
    DataTypeSize,
    DataTypeTag,
    KernelScheduleTag,
    KernelScheduleType,
    enum_auto,
)

#
#   Extend cutlass library with custom types, and missing values
#


class SONARDataType(enum.Enum):
    u4b8 = enum_auto()
    u8b128 = enum_auto()


class MixedInputKernelScheduleType(enum.Enum):
    TmaWarpSpecialized = enum_auto()
    TmaWarpSpecializedPingpong = enum_auto()
    TmaWarpSpecializedCooperative = enum_auto()


SONARDataTypeNames: dict[SONARDataType | DataType, str] = {
    **DataTypeNames,  # type: ignore
    **{
        SONARDataType.u4b8: "u4b8",
        SONARDataType.u8b128: "u8b128",
    },
}

SONARDataTypeTag: dict[SONARDataType | DataType, str] = {
    **DataTypeTag,  # type: ignore
    **{
        SONARDataType.u4b8: "cutlass::sonar_uint4b8_t",
        SONARDataType.u8b128: "cutlass::sonar_uint8b128_t",
    },
}

SONARDataTypeSize: dict[SONARDataType | DataType, int] = {
    **DataTypeSize,  # type: ignore
    **{
        SONARDataType.u4b8: 4,
        SONARDataType.u8b128: 8,
    },
}

SONARDataTypeSONARScalarTypeTag: dict[SONARDataType | DataType, str] = {
    SONARDataType.u4b8: "sonar::kU4B8",
    SONARDataType.u8b128: "sonar::kU8B128",
    DataType.u4: "sonar::kU4",
    DataType.u8: "sonar::kU8",
    DataType.s4: "sonar::kS4",
    DataType.s8: "sonar::kS8",
    DataType.f16: "sonar::kFloat16",
    DataType.bf16: "sonar::kBfloat16",
}

SONARDataTypeTorchDataTypeTag: dict[SONARDataType | DataType, str] = {
    DataType.u8: "torch::headeronly::ScalarType::Byte",
    DataType.s8: "torch::headeronly::ScalarType::Char",
    DataType.e4m3: "torch::headeronly::ScalarType::Float8_e4m3fn",
    DataType.s32: "torch::headeronly::ScalarType::Int",
    DataType.f16: "torch::headeronly::ScalarType::Half",
    DataType.bf16: "torch::headeronly::ScalarType::BFloat16",
    DataType.f32: "torch::headeronly::ScalarType::Float",
}

SONARKernelScheduleTag: dict[MixedInputKernelScheduleType | KernelScheduleType, str] = {
    **KernelScheduleTag,  # type: ignore
    **{
        MixedInputKernelScheduleType.TmaWarpSpecialized: "cutlass::gemm::KernelTmaWarpSpecialized",  # noqa: E501
        MixedInputKernelScheduleType.TmaWarpSpecializedPingpong: "cutlass::gemm::KernelTmaWarpSpecializedPingpong",  # noqa: E501
        MixedInputKernelScheduleType.TmaWarpSpecializedCooperative: "cutlass::gemm::KernelTmaWarpSpecializedCooperative",  # noqa: E501
    },
}
