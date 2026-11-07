// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Protobuf messages and gRPC clients and servers for Sonar's Inference and Control APIs.

tonic::include_proto!("sonar");

/// Schema version of this crate, reported by servers as `ServerInfo.proto_version`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
