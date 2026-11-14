// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde_default::DefaultFromSerde;
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use crate::protocol::OpaqueValue;
use crate::protocol::sampling::RequestOutputKind;

/// Parameters of a pooling request, as the Python frontend sends them after
/// `PoolingParams.verify()` resolved their defaults.
///
/// Python declares the struct `array_like`, so it travels as an array of its
/// fields in declaration order.
///
/// Original Python definition: `PoolingParams` in `sonar/pooling_params.py`.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct PoolingParams {
    /// Whether the pooler applies its activation; `None` is the pooler's default.
    #[serde(default)]
    pub use_activation: Option<bool>,
    /// Matryoshka dimensions of an embedding.
    #[serde(default)]
    pub dimensions: Option<u32>,
    /// Step pooling's tag token.
    #[serde(default)]
    pub step_tag_id: Option<u32>,
    /// Step pooling's returned tokens.
    #[serde(default)]
    pub returned_token_ids: Option<Vec<u32>>,
    /// The pooling task, such as `classify` or `token_classify`.
    #[serde(default)]
    pub task: Option<String>,
    /// Set by the engine's pooler, not by frontends.
    #[serde(default)]
    pub requires_token_ids: bool,
    /// Whether the request skips prefix-cache hits. Token-wise tasks need the
    /// hidden states of every prompt token, so `verify()` sets it for them.
    #[serde(default)]
    pub skip_reading_prefix_cache: Option<bool>,
    /// Late-interaction scoring metadata, which no Rust route sends.
    #[serde(default)]
    pub late_interaction_params: Option<OpaqueValue>,
    /// Model-specific request data, such as Clef's `joint_schema`.
    #[serde(default)]
    pub extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    /// Pooling requests only return their final output.
    #[serde(default = "final_only")]
    pub output_kind: RequestOutputKind,
    /// Late-chunking parameters of `token_embed`, which no Rust route sends.
    #[serde(default)]
    pub late_chunking_params: Option<OpaqueValue>,
}

fn final_only() -> RequestOutputKind {
    RequestOutputKind::FinalOnly
}

impl PoolingParams {
    /// The parameters of `task` as `verify()` resolves them for a model
    /// whose pooler config sets no defaults: activation on unless
    /// `use_activation` turns it off, and prefix-cache hits skipped for
    /// token-wise tasks.
    pub fn for_task(task: &str, use_activation: Option<bool>) -> Self {
        Self {
            use_activation: Some(use_activation.unwrap_or(true)),
            task: Some(task.to_string()),
            skip_reading_prefix_cache: Some(matches!(task, "token_embed" | "token_classify")),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::protocol::{decode_msgpack, encode_msgpack};

    /// `msgspec.msgpack.encode(params)` of the D3 read's params after
    /// `verify()`: `PoolingParams(task="classify", use_activation=False)`.
    const D3_PARAMS: &str = "9bc2c0c0c0a8636c617373696679c2c2c0c002c0";
    /// The same for a Clef read's `PoolingParams(task="token_classify",
    /// extra_kwargs={"joint_schema": ...})`.
    const CLEF_PARAMS: &str = "9bc3c0c0c0ae746f6b656e5f636c617373696679c2c3c081ac6a6f696e745f73636865\
        6d6184ae7175657374696f6e5f7479706573920100ae7175657374696f6e5f7370616e739292d0d8d0da92eced\
        ac6f7074696f6e5f7370616e739492d0dcd0de92d0dfe192eff192f2f4ad6f7074696f6e5f636f756e74739202\
        0202c0";

    fn clef_params() -> PoolingParams {
        let schema = json!({
            "question_types": [1, 0],
            "question_spans": [[-40, -38], [-20, -19]],
            "option_spans": [[-36, -34], [-33, -31], [-17, -15], [-14, -12]],
            "option_counts": [2, 2],
        });
        PoolingParams {
            extra_kwargs: Some(serde_json::Map::from_iter([(
                "joint_schema".to_string(),
                schema,
            )])),
            ..PoolingParams::for_task("token_classify", None)
        }
    }

    #[test]
    fn pooling_params_encode_like_python() {
        let d3 = PoolingParams::for_task("classify", Some(false));
        assert_eq!(hex::encode(encode_msgpack(&d3).unwrap()), D3_PARAMS);
        assert_eq!(
            hex::encode(encode_msgpack(&clef_params()).unwrap()),
            CLEF_PARAMS
        );
    }

    #[test]
    fn pooling_params_decode_python_encoding() {
        let decoded: PoolingParams = decode_msgpack(&hex::decode(D3_PARAMS).unwrap()).unwrap();
        assert_eq!(decoded, PoolingParams::for_task("classify", Some(false)));
        let decoded: PoolingParams = decode_msgpack(&hex::decode(CLEF_PARAMS).unwrap()).unwrap();
        assert_eq!(decoded, clef_params());
        // msgspec leaves out trailing defaults of shorter structs.
        let decoded: PoolingParams = decode_msgpack(&[0x91, 0xc2]).unwrap();
        assert_eq!(decoded.use_activation, Some(false));
        assert_eq!(decoded.output_kind, RequestOutputKind::FinalOnly);
    }
}
