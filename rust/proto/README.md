# gRPC protocol

This directory contains Sonar's gRPC schema. Rust builds use these local files.
The Buf configuration supports local schema linting. Sonar does not configure
automatic publication to a Buf registry or crates.io.

The `sonar-proto` workspace crate exposes the Prost message types and Tonic
client and server modules for the inference and control APIs. It includes the
canonical `.proto` files and generates bindings using `protox` and
`tonic-prost-build`; a separately installed `protoc` is not required.

`Control.GetServerInfo` exposes effective attention block-size metadata for
context-parallel deployments. Review both protobuf wire compatibility and the
generated Rust API when updating the schema: an additive wire-compatible field
can still require changes to Rust struct initializers.

## Sonar sampler extensions

The OpenAI chat and completion APIs accept Sonar's sampler fields at the top
level, including DRY, XTC, dynamic temperature, no-repeat n-grams, top-a,
tail-free, eta/epsilon, typical, smoothing, top-nsigma, Mirostat, and skew.
They are forwarded through the same text lowering path as engine-native
requests, including tokenizer-based DRY sequence-breaker resolution.

gRPC clients set the corresponding fields in
`GenerateRequest.sonar_sampling`. Omitted scalar fields use the shared
Python sampling defaults, not protobuf zero values. The
`dry_sequence_breakers` wrapper distinguishes an omitted list from an
explicitly empty one; `dry_sequence_breaker_ids` accepts already-tokenized
breakers.
