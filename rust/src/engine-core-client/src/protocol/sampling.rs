// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeSet, HashMap};
use std::num::NonZeroU32;

use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;
use serde_repr::{Deserialize_repr, Serialize_repr};

use crate::protocol::structured_outputs::EngineCoreStructuredOutputsParams;
use crate::protocol::tensor::WireNdArray;

fn default_top_p() -> f32 {
    1.0
}

fn default_repetition_penalty() -> f32 {
    1.0
}

fn default_temperature() -> f32 {
    1.0
}

fn default_watermarking() -> bool {
    true
}

fn default_max_tokens() -> u32 {
    16
}

///
/// Parameters for detecting repetitive N-gram patterns in output tokens.
///
/// Mirrors Python's `RepetitionDetectionParams`:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/sampling_params.py#L109-L144>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepetitionDetectionParams {
    /// Maximum N-gram size to check. 0 disables detection.
    pub max_pattern_size: u32,
    /// Minimum N-gram size to check. Defaults to 1 when zero.
    #[serde(default)]
    pub min_pattern_size: u32,
    /// Minimum number of repetitions to trigger detection (must be >= 2).
    pub min_count: u32,
}

impl RepetitionDetectionParams {
    /// Return `true` when the params are effectively disabled (max_pattern_size
    /// is 0).
    pub fn is_disabled(&self) -> bool {
        self.max_pattern_size == 0
    }
}

/// Controls how intermediate outputs are returned to the frontend.
///
/// `Cumulative = 0` is intentionally not supported in Rust frontend.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/sampling_params.py#L146-L152>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum RequestOutputKind {
    /// Return only token deltas in each update.
    #[default]
    Delta = 1,
    /// Suppress intermediate updates and return only the final output.
    FinalOnly = 2,
}

/// Sonar sampler options shared by the HTTP, text, and engine-core boundaries.
///
/// Flattened into each sampling map so Python receives its native
/// `SamplingParams` keys, not an extension object or `extra_args`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SonarSamplingParams {
    #[serde(deserialize_with = "default_on_null")]
    pub no_repeat_ngram_size: u32,
    #[serde(deserialize_with = "default_on_null")]
    pub dynatemp_min: f32,
    #[serde(deserialize_with = "default_on_null")]
    pub dynatemp_max: f32,
    #[serde(deserialize_with = "one_on_null")]
    pub dynatemp_exponent: f32,
    #[serde(deserialize_with = "default_on_null")]
    pub temperature_last: bool,
    #[serde(deserialize_with = "default_on_null")]
    pub top_a: f32,
    #[serde(deserialize_with = "one_on_null")]
    pub tfs: f32,
    #[serde(deserialize_with = "default_on_null")]
    pub eta_cutoff: f32,
    #[serde(deserialize_with = "default_on_null")]
    pub epsilon_cutoff: f32,
    #[serde(deserialize_with = "one_on_null")]
    pub typical_p: f32,
    #[serde(deserialize_with = "default_on_null")]
    pub smoothing_factor: f32,
    #[serde(deserialize_with = "one_on_null")]
    pub smoothing_curve: f32,
    #[serde(deserialize_with = "xtc_threshold_on_null")]
    pub xtc_threshold: f32,
    #[serde(deserialize_with = "default_on_null")]
    pub xtc_probability: f32,
    #[serde(deserialize_with = "default_on_null")]
    pub nsigma: f32,
    #[serde(deserialize_with = "default_on_null")]
    pub mirostat_mode: u32,
    #[serde(deserialize_with = "default_on_null")]
    pub mirostat_tau: f32,
    #[serde(deserialize_with = "default_on_null")]
    pub mirostat_eta: f32,
    #[serde(deserialize_with = "default_on_null")]
    pub dry_multiplier: f32,
    #[serde(deserialize_with = "dry_base_on_null")]
    pub dry_base: f32,
    #[serde(deserialize_with = "dry_allowed_length_on_null")]
    pub dry_allowed_length: u32,
    pub dry_sequence_breakers: Option<Vec<String>>,
    #[serde(deserialize_with = "default_on_null")]
    pub dry_sequence_breaker_ids: Vec<u32>,
    #[serde(deserialize_with = "default_on_null")]
    pub dry_range: u32,
    #[serde(deserialize_with = "dry_max_ngram_on_null")]
    pub dry_max_ngram: u32,
    #[serde(deserialize_with = "dry_scan_limit_on_null")]
    pub dry_max_occurrences: u32,
    #[serde(deserialize_with = "dry_scan_limit_on_null")]
    pub dry_early_exit_match_len: u32,
    #[serde(deserialize_with = "default_on_null")]
    pub skew: f32,
}

impl Default for SonarSamplingParams {
    fn default() -> Self {
        Self {
            no_repeat_ngram_size: 0,
            dynatemp_min: 0.0,
            dynatemp_max: 0.0,
            dynatemp_exponent: 1.0,
            temperature_last: false,
            top_a: 0.0,
            tfs: 1.0,
            eta_cutoff: 0.0,
            epsilon_cutoff: 0.0,
            typical_p: 1.0,
            smoothing_factor: 0.0,
            smoothing_curve: 1.0,
            xtc_threshold: 0.1,
            xtc_probability: 0.0,
            nsigma: 0.0,
            mirostat_mode: 0,
            mirostat_tau: 0.0,
            mirostat_eta: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_sequence_breakers: None,
            dry_sequence_breaker_ids: Vec::new(),
            dry_range: 0,
            dry_max_ngram: 12,
            dry_max_occurrences: 8,
            dry_early_exit_match_len: 8,
            skew: 0.0,
        }
    }
}

// Python's from_optional treats explicit JSON null just like an omitted sampler
// override. Keep non-zero defaults as well as zero/false defaults on this path.
fn default_on_null<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

fn one_on_null<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
    Ok(Option::<f32>::deserialize(deserializer)?.unwrap_or(1.0))
}

fn xtc_threshold_on_null<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<f32, D::Error> {
    Ok(Option::<f32>::deserialize(deserializer)?.unwrap_or(0.1))
}

fn dry_base_on_null<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
    Ok(Option::<f32>::deserialize(deserializer)?.unwrap_or(1.75))
}

fn dry_allowed_length_on_null<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<u32, D::Error> {
    Ok(Option::<u32>::deserialize(deserializer)?.unwrap_or(2))
}

fn dry_max_ngram_on_null<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<u32, D::Error> {
    Ok(Option::<u32>::deserialize(deserializer)?.unwrap_or(12))
}

fn dry_scan_limit_on_null<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<u32, D::Error> {
    Ok(Option::<u32>::deserialize(deserializer)?.unwrap_or(8))
}

/// Engine-core-facing sampling parameters for text generation.
///
/// This is the normalized southbound subset used by the Rust frontend when it
/// talks to Python engine-core over the wire. User-facing request semantics
/// such as `stop` strings, `n`, `ignore_eos`, and output aggregation mode are
/// intentionally handled by higher layers before values reach this DTO.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/sampling_params.py#L155-L291>
// Python's SamplingParams is `omit_defaults=True`, so msgpack drops
// default-valued keys; default the whole struct. Per-field fns cover the
// non-zero defaults.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, DefaultFromSerde)]
#[serde(default)]
pub struct EngineCoreSamplingParams {
    /// Controls randomness. Lower values are more deterministic; zero means
    /// greedy sampling.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Whether to apply the engine's configured watermark to this request.
    #[serde(default = "default_watermarking")]
    pub watermarking: bool,
    /// Cumulative probability threshold for nucleus sampling.
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Maximum number of top tokens to consider. `0` means all tokens.
    pub top_k: u32,
    /// Random seed used by the sampler when present.
    pub seed: Option<i64>,
    /// Maximum number of tokens to generate per output sequence.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Minimum number of tokens to generate before EOS or stop-token handling.
    pub min_tokens: u32,
    /// Maximum number of reasoning ("thinking") tokens to emit before the
    /// reasoning section is force-closed. `None` means unlimited; the
    /// user-facing `-1` sentinel is normalized to `None` by the frontend before
    /// reaching this DTO, so only non-negative values are sent. Enforced
    /// engine-side (and only when a reasoning parser is configured).
    pub thinking_token_budget: Option<u64>,
    /// Number of log probabilities to return per generated token.
    ///
    /// `None` disables sample logprobs. `-1` requests the full vocabulary.
    pub logprobs: Option<i32>,
    /// Number of log probabilities to return per prompt token.
    ///
    /// `None` disables prompt logprobs. `-1` requests the full vocabulary.
    pub prompt_logprobs: Option<i32>,
    /// `[num_rows, num_ids]` candidate token IDs per scored causal prompt row,
    /// `-1` padding shorter rows.
    pub prompt_logprob_token_ids: Option<WireNdArray>,
    /// First causal prompt row to score; `None` scores from the first row.
    pub prompt_logprob_start: Option<u32>,
    /// Minimum probability threshold for token sampling.
    pub min_p: f32,
    /// Frequency penalty applied by the sampler.
    pub frequency_penalty: f32,
    /// Presence penalty applied by the sampler.
    pub presence_penalty: f32,
    /// Repetition penalty applied by the sampler.
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    /// Parameters for detecting repetitive N-gram patterns. `None` disables
    /// detection.
    pub repetition_detection: Option<RepetitionDetectionParams>,
    /// Token IDs that stop generation.
    pub stop_token_ids: Vec<u32>,
    /// Primary EOS token ID used by engine-core's dedicated EOS stop path.
    ///
    /// This mirrors Python's internal `_eos_token_id` field and is derived by
    /// the frontend from tokenizer/model metadata rather than supplied directly
    /// by end users.
    #[serde(rename = "_eos_token_id")]
    pub eos_token_id: Option<u32>,
    /// Complete stop-token set used by engine-core for `min_tokens` masking.
    ///
    /// This mirrors Python's internal `_all_stop_token_ids` field and should
    /// contain explicit `stop_token_ids` plus any frontend-derived EOS token
    /// IDs.
    #[serde(rename = "_all_stop_token_ids")]
    pub all_stop_token_ids: BTreeSet<u32>,
    /// Logit biases to apply during sampling.
    /// Keys are token IDs
    pub logit_bias: Option<HashMap<u32, f32>>,
    /// Restrict output to these token IDs only.
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Tokenized bad words to avoid during generation.
    #[serde(rename = "_bad_words_token_ids")]
    pub bad_words_token_ids: Option<Vec<Vec<u32>>>,
    /// Parameters for configuring structured outputs (guided decoding).
    pub structured_outputs: Option<EngineCoreStructuredOutputsParams>,
    /// Specific token IDs for which log probabilities should be returned at
    /// each position.
    ///
    /// When set, the engine returns logprobs for exactly these tokens in
    /// addition to the sampled/scored token. Mutually exclusive with the
    /// `logprobs` count field in practice.
    pub logprob_token_ids: Option<Vec<u32>>,
    /// If `Some(true)`, the request will not attempt to read from the prefix
    /// cache; newly computed blocks may still populate the cache. `None`
    /// defers to engine-core defaults.
    pub skip_reading_prefix_cache: Option<bool>,
    /// Additional request parameters for custom extensions (from `sonar_xargs`).
    pub extra_args: Option<HashMap<String, serde_json::Value>>,
    /// Number of prompt tokens to skip from returned routed-expert data.
    /// A value of zero returns routing data for the entire prompt.
    pub routed_experts_prompt_start: u32,
    /// Number of newly generated tokens to batch into each streamed output.
    /// The first and final outputs are always emitted immediately. Engine-core
    /// ignores it; [`EngineCoreClient::call`](crate::EngineCoreClient::call)
    /// batches output deliveries by it.
    pub stream_interval: Option<NonZeroU32>,
    #[serde(flatten)]
    pub sonar_sampling: SonarSamplingParams,
}

impl EngineCoreSamplingParams {
    /// Constructs a default sampling params for testing purposes only.
    pub fn for_test() -> Self {
        Self {
            temperature: 1.0,
            watermarking: true,
            top_p: 1.0,
            top_k: 0,
            seed: None,
            max_tokens: 65536,
            min_tokens: 0,
            thinking_token_budget: None,
            logprobs: None,
            prompt_logprobs: None,
            prompt_logprob_token_ids: None,
            prompt_logprob_start: None,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            repetition_detection: None,
            stop_token_ids: Vec::new(),
            eos_token_id: None,
            all_stop_token_ids: BTreeSet::new(),
            logit_bias: None,
            allowed_token_ids: None,
            bad_words_token_ids: None,
            structured_outputs: None,
            logprob_token_ids: None,
            skip_reading_prefix_cache: None,
            extra_args: None,
            routed_experts_prompt_start: 0,
            stream_interval: None,
            sonar_sampling: SonarSamplingParams::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::{EngineCoreSamplingParams, SonarSamplingParams};
    use crate::protocol::decode_msgpack;
    use crate::protocol::request::EngineCoreRequest;

    #[test]
    fn sonar_sampler_fields_and_phrase_retry_survive_the_engine_wire() {
        let params: EngineCoreSamplingParams = serde_json::from_value(serde_json::json!({
            "dry_multiplier": 0.8,
            "dry_sequence_breaker_ids": [13, 42],
            "xtc_probability": 0.25,
            "dynatemp_exponent": 1.5,
            "extra_args": {"_sonar_phrase_retry": {"rollback_tokens": 3}}
        }))
        .unwrap();
        let bytes = crate::protocol::encode_msgpack(&params).unwrap();
        let value: serde_json::Value = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(
            value["dry_sequence_breaker_ids"],
            serde_json::json!([13, 42])
        );
        assert_eq!(value["xtc_probability"], 0.25);
        assert_eq!(
            value["extra_args"]["_sonar_phrase_retry"]["rollback_tokens"],
            3
        );
        assert!(value.get("sonar_sampling").is_none());
        let decoded: EngineCoreSamplingParams = decode_msgpack(&bytes).unwrap();
        assert_eq!(decoded, params);
    }

    #[test]
    fn sonar_sampler_nulls_preserve_python_defaults() {
        let params: EngineCoreSamplingParams = serde_json::from_value(serde_json::json!({
            "dynatemp_exponent": null,
            "tfs": null,
            "typical_p": null,
            "smoothing_curve": null,
            "xtc_threshold": null,
            "dry_base": null,
            "dry_allowed_length": null,
            "dry_max_ngram": null,
            "dry_max_occurrences": null,
            "dry_early_exit_match_len": null,
            "temperature_last": null,
            "dry_sequence_breaker_ids": null
        }))
        .unwrap();
        assert_eq!(params.sonar_sampling, SonarSamplingParams::default());
    }

    #[test]
    fn decodes_sampling_params_with_omitted_defaults() {
        let sampling_params = Value::Map(vec![
            (
                Value::from("stop_token_ids"),
                Value::Array(vec![Value::from(151643u32)]),
            ),
            (Value::from("skip_reading_prefix_cache"), Value::from(false)),
        ]);
        let request = Value::Array(vec![
            Value::from("req-omit-defaults"),
            Value::Array(vec![
                Value::from(1u32),
                Value::from(2u32),
                Value::from(3u32),
            ]),
            Value::Nil,
            sampling_params,
            Value::Nil,
            Value::from(1.0f64),
        ]);

        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &request).unwrap();

        let decoded: EngineCoreRequest = decode_msgpack(&bytes)
            .expect("a real omit_defaults request must decode (regression: missing field)");

        assert_eq!(decoded.request_id, "req-omit-defaults");
        let sampling = decoded.sampling_params.expect("sampling params present");

        assert_eq!(sampling.stop_token_ids, vec![151643]);
        assert_eq!(sampling.skip_reading_prefix_cache, Some(false));

        // Omitted fields -> Python defaults.
        assert_eq!(sampling.temperature, 1.0);
        assert!(sampling.watermarking);
        assert_eq!(sampling.top_p, 1.0);
        assert_eq!(sampling.top_k, 0);
        assert_eq!(sampling.seed, None);
        assert_eq!(sampling.max_tokens, 16);
        assert_eq!(sampling.min_tokens, 0);
        assert_eq!(sampling.min_p, 0.0);
        assert_eq!(sampling.frequency_penalty, 0.0);
        assert_eq!(sampling.presence_penalty, 0.0);
        assert_eq!(sampling.repetition_penalty, 1.0);
        assert_eq!(sampling.repetition_detection, None);
        assert_eq!(sampling.logprobs, None);
        assert_eq!(sampling.prompt_logprobs, None);
        assert_eq!(sampling.eos_token_id, None);
        assert!(sampling.all_stop_token_ids.is_empty());
        assert_eq!(sampling.sonar_sampling, SonarSamplingParams::default());
    }
}
