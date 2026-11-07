// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Opt-in, weight-only receipts. These do not attest to inference execution.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use anyhow::{Result as AnyResult, bail};
use asynk_strim_attr::{TryYielder, try_stream};
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use futures::future::try_join_all;
use futures::{Stream, StreamExt as _, pin_mut};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sonar_chat::ChatEvent;
use sonar_text::{DecodedTextEvent, TextOutputStreamExt as _};
use validator::Validate;

use crate::error::{ApiError, chat_submit_error, invalid_request, server_error, text_submit_error};
use crate::routes::openai::chat_completions::{self as chat, ChatCompletionRequest};
use crate::routes::openai::completions::{self as text, CompletionRequest};
use crate::routes::openai::utils::types::Normalizable;
use crate::routes::openai::utils::validate_max_output_tokens;
use crate::state::AppState;
use crate::utils::{resolve_request_context, sse_response, unix_timestamp};

/// Public configuration only: never accepts a serving-side private key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DphnNetworkConfig {
    pub public_key: String,
    pub worker_id: String,
    pub model_manifest_id: String,
    pub key_id: String,
    pub num_raw_terms: u32,
    pub raw_term_bytes: u32,
}

impl Default for DphnNetworkConfig {
    fn default() -> Self {
        Self {
            public_key: String::new(),
            worker_id: String::new(),
            model_manifest_id: String::new(),
            key_id: "default".to_string(),
            num_raw_terms: 8,
            raw_term_bytes: 256,
        }
    }
}

impl DphnNetworkConfig {
    pub fn validate(&self) -> AnyResult<()> {
        for (name, value) in [
            ("public_key", &self.public_key),
            ("worker_id", &self.worker_id),
            ("model_manifest_id", &self.model_manifest_id),
            ("key_id", &self.key_id),
        ] {
            if value.trim().is_empty() || value.contains('\0') {
                bail!("dphn-network {name} must be nonempty and contain no NUL");
            }
        }
        if !(1..=64).contains(&self.num_raw_terms) || !(16..=4096).contains(&self.raw_term_bytes) {
            bail!("dphn-network sampling requires 1..64 terms and 16..4096 bytes per term");
        }
        Ok(())
    }
}

fn nonce(headers: &HeaderMap) -> Result<String, ApiError> {
    let values = headers.get_all("x-dphn-proof-nonce");
    if values.iter().count() != 1 {
        return Err(invalid_request!(
            "exactly one X-DPHN-Proof-Nonce header is required"
        ));
    }
    let value = values
        .iter()
        .next()
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| invalid_request!("X-DPHN-Proof-Nonce must be valid header text"))?;
    if value.trim().is_empty()
        || value.len() > 1024
        || !value.bytes().all(|b| (0x20..=0x7e).contains(&b))
    {
        return Err(invalid_request!(
            "X-DPHN-Proof-Nonce must be nonblank visible ASCII (0x20..0x7e), at most 1024 bytes"
        ));
    }
    Ok(value.to_string())
}

#[derive(Debug, Deserialize, Serialize)]
struct RankReceipt {
    rank: u32,
    dp_rank: u32,
    worker_id: String,
    proof: String,
}

async fn prove(
    state: &AppState,
    request_id: &str,
    nonce: &str,
    ids: &[u32],
) -> Result<Value, ApiError> {
    let config = state
        .dphn_network_config
        .as_ref()
        .ok_or_else(|| server_error!("dphn-network is not enabled"))?;
    let per_engine: Vec<Vec<RankReceipt>> = state
        .engine_core_client()
        .call_utility("dphn_prove", (request_id, nonce, ids))
        .await
        .map_err(|_| server_error!("dphn-network weight receipt generation failed"))?;
    if per_engine.is_empty() || per_engine.iter().any(Vec::is_empty) {
        return Err(server_error!("dphn-network returned missing rank receipts"));
    }
    let mut identities = std::collections::HashSet::new();
    let mut proofs = Vec::new();
    for receipt in per_engine.into_iter().flatten() {
        let expected = format!(
            "{}/dp/{}/rank/{}",
            config.worker_id, receipt.dp_rank, receipt.rank
        );
        if receipt.worker_id != expected
            || receipt.proof.is_empty()
            || !identities.insert((receipt.dp_rank, receipt.rank))
        {
            return Err(server_error!("dphn-network returned invalid rank receipts"));
        }
        proofs.push(receipt);
    }
    Ok(
        json!({"version": 2, "request_id": request_id, "nonce": nonce,
        "output_token_ids": ids, "proofs": proofs}),
    )
}

fn decode<T: serde::de::DeserializeOwned + Normalizable + Validate>(
    value: Value,
) -> Result<T, ApiError> {
    let mut request: T =
        serde_json::from_value(value).map_err(|error| invalid_request!("{error}"))?;
    request.normalize();
    request.validate().map_err(|error| invalid_request!("{error}"))?;
    Ok(request)
}

/// Python's `SamplingParams` treats temperatures below this as greedy.
const SAMPLING_EPS: f32 = 1e-5;

/// Reject greedy sampling with several choices, as Python's `SamplingParams`
/// does, before each choice is expanded into its own `n=1` request.
fn reject_greedy_choices(
    n: usize,
    temperature: Option<f32>,
    default_temperature: Option<f32>,
) -> Result<(), ApiError> {
    // Same fallback as lowering: request value, then model default, then 1.0.
    let temperature = temperature.or(default_temperature).unwrap_or(1.0);
    // Negative values are left to the regular temperature validation.
    if n > 1 && (0.0..SAMPLING_EPS).contains(&temperature) {
        return Err(invalid_request!(
            "n must be 1 when using greedy sampling, got {n}."
        ));
    }
    Ok(())
}

/// Python's `SONAR_MAX_N_SEQUENCES` limit on choices per prompt.
fn max_n_sequences() -> u64 {
    std::env::var("SONAR_MAX_N_SEQUENCES")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(16384)
}

/// Expand only the opt-in routes; legacy handlers and their output remain untouched.
fn expand(mut body: Value, chat: bool) -> Result<Vec<Value>, ApiError> {
    let object = body
        .as_object_mut()
        .ok_or_else(|| invalid_request!("request must be an object"))?;
    let max_n = max_n_sequences();
    let n = match object.get("n") {
        None | Some(Value::Null) => 1,
        Some(value) => value
            .as_u64()
            .filter(|n| (1..=max_n).contains(n))
            .ok_or_else(|| {
                invalid_request!(
                    "n must be an integer from 1 to {max_n}; set SONAR_MAX_N_SEQUENCES to raise the limit"
                )
            })?,
    };
    object.insert("n".to_string(), json!(1));
    // Like Python parallel sampling, choice `i` of an explicitly seeded request
    // uses `seed + i`; -1 means unseeded, as in `SamplingParams`.
    let seed = match object.get("seed") {
        None | Some(Value::Null) => None,
        Some(value) => value
            .as_i64()
            .filter(|seed| *seed != -1)
            .map(|seed| {
                seed.checked_add_unsigned(n - 1).map(|_| seed).ok_or_else(|| {
                    invalid_request!("seed + n - 1 must fit in a signed 64-bit integer")
                })
            })
            .transpose()?,
    };
    let prompts = if chat {
        vec![None]
    } else {
        let prompt =
            object.remove("prompt").ok_or_else(|| invalid_request!("prompt is required"))?;
        match prompt {
            Value::Array(values)
                if values.first().is_some_and(|v| v.is_string() || v.is_array()) =>
            {
                values.into_iter().map(Some).collect()
            }
            prompt => vec![Some(prompt)],
        }
    };
    if prompts.is_empty() {
        return Err(invalid_request!("prompt batch must be nonempty"));
    }
    let total = usize::try_from(n)
        .ok()
        .and_then(|n| prompts.len().checked_mul(n))
        .ok_or_else(|| invalid_request!("too many prompts and choices"))?;
    let mut requests = Vec::with_capacity(total);
    let last_prompt = prompts.len() - 1;
    let mut template = Some(body);
    for (prompt_index, mut prompt) in prompts.into_iter().enumerate() {
        for choice in 0..n {
            let mut request = if prompt_index == last_prompt && choice + 1 == n {
                template.take().expect("last expanded choice owns the template")
            } else {
                template.as_ref().expect("template remains until last choice").clone()
            };
            if let Some(prompt) = prompt.as_mut() {
                request["prompt"] = if choice + 1 == n {
                    prompt.take()
                } else {
                    prompt.clone()
                };
            }
            if let Some(seed) = seed {
                request["seed"] = json!(seed + choice as i64);
            }
            requests.push(request);
        }
    }
    Ok(requests)
}

enum PreparedChoice {
    Chat(chat::convert::PreparedRequest),
    Text(text::convert::PreparedRequest),
}

type TextStream = Pin<Box<dyn Stream<Item = sonar_text::Result<DecodedTextEvent>> + Send>>;

enum SubmittedChoice {
    Chat {
        stream: sonar_chat::ChatEventStream,
        request_id: String,
        response_model: String,
        options: chat::convert::ResponseOptions,
    },
    Text {
        stream: TextStream,
        request_id: String,
        response_model: String,
        options: text::convert::ResponseOptions,
    },
}

async fn submit_choice(
    state: &AppState,
    choice: PreparedChoice,
) -> Result<SubmittedChoice, ApiError> {
    match choice {
        PreparedChoice::Chat(prepared) => {
            let stream = state
                .chat
                .chat(prepared.chat_request)
                .await
                .map_err(|e| chat_submit_error("failed to submit chat request", e))?;
            Ok(SubmittedChoice::Chat {
                stream,
                request_id: prepared.request_id,
                response_model: prepared.response_model,
                options: prepared.options,
            })
        }
        PreparedChoice::Text(prepared) => {
            let stream = state
                .chat
                .text()
                .generate(prepared.text_request)
                .await
                .map_err(|e| text_submit_error("failed to submit completion request", e))?;
            Ok(SubmittedChoice::Text {
                stream: Box::pin(stream),
                request_id: prepared.request_id,
                response_model: prepared.response_model,
                options: prepared.options,
            })
        }
    }
}

type ChunkStream = Pin<Box<dyn Stream<Item = Result<Value, ApiError>> + Send>>;

pub(crate) async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    respond(state, headers, body, true)
        .await
        .unwrap_or_else(IntoResponse::into_response)
}

pub(crate) async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    respond(state, headers, body, false)
        .await
        .unwrap_or_else(IntoResponse::into_response)
}

async fn respond(
    state: Arc<AppState>,
    headers: HeaderMap,
    body: Value,
    is_chat: bool,
) -> Result<Response, ApiError> {
    // Validate every choice, and the nonce, before submitting any engine work.
    let nonce = nonce(&headers)?;
    let request_context =
        resolve_request_context(&headers, body.get("request_id").and_then(Value::as_str));
    let n = body.get("n").and_then(Value::as_u64).unwrap_or(1) as usize;
    let requests = expand(body, is_chat)?;
    let default_temperature = state
        .chat
        .text()
        .sampling_hints()
        .map_err(|_| server_error!("failed to load model sampling defaults"))?
        .default_temperature;
    let mut choices = Vec::with_capacity(requests.len());
    let mut stream = false;
    let mut include_usage = false;
    for (index, body) in requests.into_iter().enumerate() {
        if is_chat {
            let request: ChatCompletionRequest = decode(body)?;
            if !request.include_reasoning && state.chat.reasoning_parser_name().is_some() {
                return Err(invalid_request!(
                    "dphn-network cannot combine include_reasoning=false with a reasoning parser: weight-receipt bundles expose raw output token IDs"
                ));
            }
            stream = request.stream;
            let model = state
                .resolve_model_with_loras(request.model.as_deref().filter(|v| !v.is_empty()))
                .await;
            let mut prepared =
                chat::convert::prepare_chat_request(request, &model, request_context.clone())?;
            reject_greedy_choices(
                n,
                prepared.chat_request.sampling_params.temperature,
                default_temperature,
            )?;
            validate_max_output_tokens(
                "max_completion_tokens",
                prepared.chat_request.sampling_params.max_tokens,
                state.chat.engine_core_client().max_model_len(),
            )?;
            prepared.chat_request.request_id = format!("{}/choice/{index}", prepared.request_id);
            include_usage = prepared.options.include_usage;
            // Aggregate usage once across choices rather than emitting per-choice totals.
            prepared.options.include_usage = false;
            choices.push(PreparedChoice::Chat(prepared));
        } else {
            let request: CompletionRequest = decode(body)?;
            stream = request.stream;
            let model = state
                .resolve_model_with_loras(request.model.as_deref().filter(|v| !v.is_empty()))
                .await;
            let tokenizer = state.chat.text().tokenizer();
            let mut prepared = text::convert::prepare_completion_request(
                request,
                &model,
                request_context.clone(),
                tokenizer.as_ref(),
            )?;
            reject_greedy_choices(
                n,
                prepared.text_request.sampling_params.temperature,
                default_temperature,
            )?;
            validate_max_output_tokens(
                "max_tokens",
                prepared.text_request.sampling_params.max_tokens,
                state.chat.engine_core_client().max_model_len(),
            )?;
            prepared.text_request.request_id = format!("{}/choice/{index}", prepared.request_id);
            include_usage = prepared.options.include_usage;
            prepared.options.include_usage = false;
            choices.push(PreparedChoice::Text(prepared));
        }
    }
    // Submission is concurrent and completes before an HTTP 200/SSE response.
    // Dropping successful streams on any submission failure retains cancellation cleanup.
    let choices =
        try_join_all(choices.into_iter().map(|choice| submit_choice(&state, choice))).await?;
    let created = unix_timestamp();
    if stream {
        let chunks = stream_choices(state.clone(), choices, nonce, created, include_usage, n);
        Ok(sse_response(
            events(chunks),
            state.api_server_options.sse_keep_alive_interval,
        ))
    } else {
        let responses =
            try_join_all(choices.into_iter().enumerate().map(|(index, choice)| {
                collect_choice(&state, choice, &nonce, index as u32, created)
            }))
            .await?;
        let mut response = None;
        for (index, value) in responses.into_iter().enumerate() {
            merge_response(&mut response, value, index % n != 0)?;
        }
        Ok(
            Json(response.ok_or_else(|| server_error!("dphn-network produced no choices"))?)
                .into_response(),
        )
    }
}

async fn collect_choice(
    state: &AppState,
    choice: SubmittedChoice,
    nonce: &str,
    index: u32,
    created: u64,
) -> Result<Value, ApiError> {
    let (response, proof) = match choice {
        SubmittedChoice::Chat {
            stream,
            request_id,
            response_model,
            options,
        } => {
            let collected = stream
                .collect_message()
                .await
                .map_err(|_| server_error!("failed to collect chat completion"))?;
            let proof = prove(
                state,
                &format!("{request_id}/choice/{index}"),
                nonce,
                &collected.token_ids,
            )
            .await?;
            let response = chat::render_chat_completion(
                collected,
                request_id,
                response_model,
                created,
                state.api_server_options,
                state.chat.reasoning_parser_name().is_some(),
                options,
            )?;
            (serde_json::to_value(response), proof)
        }
        SubmittedChoice::Text {
            stream,
            request_id,
            response_model,
            options,
        } => {
            let collected = stream
                .collect_output()
                .await
                .map_err(|_| server_error!("failed to collect completion"))?;
            let proof = prove(
                state,
                &format!("{request_id}/choice/{index}"),
                nonce,
                &collected.token_ids,
            )
            .await?;
            let response = text::render_completion(
                collected,
                request_id,
                response_model,
                created,
                state.api_server_options,
                options,
            )?;
            (serde_json::to_value(response), proof)
        }
    };
    let mut response = response.map_err(|_| server_error!("failed to serialize completion"))?;
    let choice = response["choices"]
        .as_array_mut()
        .and_then(|v| v.first_mut())
        .ok_or_else(|| server_error!("completion returned no choice"))?;
    choice["index"] = json!(index);
    choice["x_dphn_proof"] = proof;
    Ok(response)
}

fn add_usage(total: &mut Value, next: &Value, repeated_prompt: bool) {
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"] {
        let value = next.get(key).and_then(Value::as_u64).unwrap_or(0);
        let value = if repeated_prompt && key == "prompt_tokens" {
            0
        } else if repeated_prompt && key == "total_tokens" {
            next.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0)
        } else {
            value
        };
        total[key] = json!(total.get(key).and_then(Value::as_u64).unwrap_or(0) + value);
    }
    for key in ["prompt_tokens_details", "completion_tokens_details"] {
        if let Some(details) = next.get(key).and_then(Value::as_object) {
            if repeated_prompt && key == "prompt_tokens_details" {
                continue;
            }
            if !total.get(key).is_some_and(Value::is_object) {
                total[key] = json!({});
            }
            for (field, value) in details {
                if let Some(n) = value.as_u64() {
                    let sum = total[key].get(field).and_then(Value::as_u64).unwrap_or(0) + n;
                    total[key][field] = json!(sum);
                }
            }
        }
    }
}

fn merge_response(
    total: &mut Option<Value>,
    mut next: Value,
    repeated_prompt: bool,
) -> Result<(), ApiError> {
    if let Some(total) = total {
        // Prompt accounting follows the serving API's per-prompt choice semantics.
        add_usage(&mut total["usage"], &next["usage"], repeated_prompt);
        let choices =
            next["choices"].as_array_mut().ok_or_else(|| server_error!("missing choices"))?;
        total["choices"]
            .as_array_mut()
            .ok_or_else(|| server_error!("missing choices"))?
            .append(choices);
    } else {
        *total = Some(next);
    }
    Ok(())
}

#[try_stream]
async fn stream_choice(
    state: Arc<AppState>,
    choice: SubmittedChoice,
    nonce: Arc<str>,
    index: usize,
    created: u64,
    mut y: TryYielder<Value, ApiError>,
) -> Result<(), ApiError> {
    let ids = Arc::new(Mutex::new(Vec::<u32>::new()));
    let recorder = ids.clone();
    let (chunks, request_id): (ChunkStream, String) = match choice {
        SubmittedChoice::Chat {
            mut stream,
            request_id,
            response_model,
            mut options,
        } => {
            let prompt_text = stream.take_prompt_text();
            let stream = stream.inspect(move |event| {
                if let Ok(ChatEvent::LogprobsDelta { token_ids, .. }) = event {
                    recorder
                        .lock()
                        .expect("receipt token recorder poisoned")
                        .extend_from_slice(token_ids);
                }
            });
            options.include_usage = true;
            let chunks = chat::chat_completion_chunk_stream(
                stream,
                prompt_text,
                request_id.clone(),
                response_model,
                created,
                state.api_server_options,
                state.chat.reasoning_parser_name().is_some(),
                options,
            )
            .map(|chunk| {
                chunk.and_then(|c| {
                    serde_json::to_value(c)
                        .map_err(|_| server_error!("failed to serialize chat chunk"))
                })
            });
            (Box::pin(chunks), request_id)
        }
        SubmittedChoice::Text {
            stream,
            request_id,
            response_model,
            mut options,
        } => {
            let stream = stream.inspect(move |event| {
                if let Ok(DecodedTextEvent::TextDelta { sampled, .. }) = event {
                    recorder
                        .lock()
                        .expect("receipt token recorder poisoned")
                        .extend_from_slice(&sampled.token_ids);
                }
            });
            options.include_usage = true;
            let chunks = text::completion_chunk_stream(
                stream,
                request_id.clone(),
                response_model,
                created,
                state.api_server_options,
                options,
            )
            .map(|chunk| {
                chunk.and_then(|c| {
                    serde_json::to_value(c)
                        .map_err(|_| server_error!("failed to serialize completion chunk"))
                })
            });
            (Box::pin(chunks), request_id)
        }
    };
    pin_mut!(chunks);
    let mut finished = false;
    while let Some(chunk) = chunks.next().await {
        let mut chunk = chunk?;
        if let Some(choices) = chunk["choices"].as_array_mut() {
            for choice in choices {
                choice["index"] = json!(index);
                if choice.get("finish_reason").is_some_and(|v| !v.is_null()) {
                    let output_ids =
                        std::mem::take(&mut *ids.lock().expect("receipt token recorder poisoned"));
                    choice["x_dphn_proof"] = prove(
                        &state,
                        &format!("{request_id}/choice/{index}"),
                        &nonce,
                        &output_ids,
                    )
                    .await?;
                    finished = true;
                }
            }
        }
        y.yield_ok(chunk).await;
    }
    if !finished {
        return Err(server_error!("generation ended without a weight receipt"));
    }
    Ok(())
}

#[try_stream]
async fn stream_choices(
    state: Arc<AppState>,
    choices: Vec<SubmittedChoice>,
    nonce: String,
    created: u64,
    include_usage: bool,
    n: usize,
    mut y: TryYielder<Value, ApiError>,
) -> Result<(), ApiError> {
    let mut usage_by_choice = vec![json!({}); choices.len()];
    let nonce: Arc<str> = nonce.into();
    let streams = choices.into_iter().enumerate().map(|(index, choice)| {
        let chunks: ChunkStream = Box::pin(stream_choice(
            state.clone(),
            choice,
            nonce.clone(),
            index,
            created,
        ));
        chunks.map(move |chunk| chunk.map(|chunk| (index, chunk)))
    });
    let mut chunks = futures::stream::select_all(streams);
    let mut usage_envelope = None;
    while let Some(chunk) = chunks.next().await {
        let (index, mut chunk) = chunk?;
        if chunk.get("usage").is_some_and(|v| !v.is_null()) {
            usage_by_choice[index] = chunk["usage"].take();
            let mut total = json!({});
            for (index, usage) in usage_by_choice.iter().enumerate() {
                add_usage(&mut total, usage, index % n != 0);
            }
            chunk["usage"] = total;
            if chunk["choices"].as_array().is_some_and(Vec::is_empty) {
                usage_envelope = Some(chunk);
                continue;
            }
        }
        y.yield_ok(chunk).await;
    }
    if include_usage && let Some(chunk) = usage_envelope {
        y.yield_ok(chunk).await;
    }
    Ok(())
}

#[try_stream]
async fn events(
    stream: impl Stream<Item = Result<Value, ApiError>>,
    mut y: TryYielder<Event, Infallible>,
) -> Result<(), Infallible> {
    pin_mut!(stream);
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(chunk) => {
                y.yield_ok(Event::default().json_data(chunk).expect("JSON value serializes"))
                    .await
            }
            Err(error) => {
                y.yield_ok(
                    Event::default()
                        .json_data(error.to_error_response())
                        .expect("API error serializes"),
                )
                .await;
                return Ok(()); // Fail closed: no successful terminal choice or [DONE].
            }
        }
    }
    y.yield_ok(Event::default().data("[DONE]")).await;
    Ok(())
}

pub(crate) async fn unsupported() -> ApiError {
    invalid_request!(
        "dphn-network supports only /v1/chat/completions and /v1/completions generation; receipts attest sampled runtime weights only"
    )
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use super::*;

    #[test]
    fn nonce_is_required_bounded_and_preserved() {
        let mut headers = HeaderMap::new();
        assert!(nonce(&headers).is_err());
        headers.insert("x-dphn-proof-nonce", "   ".parse().unwrap());
        assert!(nonce(&headers).is_err());
        headers.insert("x-dphn-proof-nonce", "literal-nonce".parse().unwrap());
        assert_eq!(nonce(&headers).unwrap(), "literal-nonce");
        headers.insert("x-dphn-proof-nonce", " a b ".parse().unwrap());
        assert_eq!(nonce(&headers).unwrap(), " a b ");
        headers.insert(
            "x-dphn-proof-nonce",
            axum::http::HeaderValue::from_bytes(b"a\tb").unwrap(),
        );
        assert!(nonce(&headers).is_err());
        headers.insert(
            "x-dphn-proof-nonce",
            axum::http::HeaderValue::from_bytes(b"a\x80b").unwrap(),
        );
        assert!(nonce(&headers).is_err());
        headers.insert("x-dphn-proof-nonce", "a".repeat(1024).parse().unwrap());
        assert_eq!(nonce(&headers).unwrap().len(), 1024);
        headers.insert("x-dphn-proof-nonce", "a".repeat(1025).parse().unwrap());
        assert!(nonce(&headers).is_err());
        headers.insert("x-dphn-proof-nonce", "one".parse().unwrap());
        headers.append("x-dphn-proof-nonce", "two".parse().unwrap());
        assert!(nonce(&headers).is_err());
    }

    #[test]
    fn expansion_preserves_token_prompts_and_choice_order() {
        let expanded = expand(json!({"prompt": [[1, 2], [3]], "n": 2}), false).unwrap();
        assert_eq!(expanded.len(), 4);
        assert_eq!(expanded[0]["prompt"], json!([1, 2]));
        assert_eq!(expanded[1]["prompt"], json!([1, 2]));
        assert_eq!(expanded[2]["prompt"], json!([3]));
        assert_eq!(expand(json!({"prompt": [1, 2]}), false).unwrap().len(), 1);
        assert!(expand(json!({"prompt": "a", "n": 0}), false).is_err());
    }

    #[test]
    fn expansion_enforces_python_choice_limit_before_allocating() {
        // Python accepts 1..=SONAR_MAX_N_SEQUENCES (default 16384) for both APIs.
        assert_eq!(
            expand(json!({"messages": [], "n": 11}), true).unwrap().len(),
            11
        );
        for chat in [false, true] {
            for n in [
                json!(16385),
                json!(u32::MAX),
                json!(u64::MAX),
                json!(-1),
                json!(1.5),
            ] {
                let body = json!({"prompt": "a", "messages": [], "n": n});
                let error = expand(body, chat).expect_err("oversized n must be rejected");
                assert_eq!(error.into_response().status(), StatusCode::BAD_REQUEST);
            }
        }
        assert_eq!(
            expand(json!({"prompt": "a", "n": 16384}), false).unwrap().len(),
            16384
        );
    }

    #[test]
    fn expansion_offsets_explicit_seed_per_choice_like_python() {
        let seeds = |body: Value| -> Vec<Value> {
            expand(body, false).unwrap().into_iter().map(|r| r["seed"].clone()).collect()
        };
        // Child seed = parent seed + index within each prompt's n choices.
        assert_eq!(
            seeds(json!({"prompt": [[1], [2]], "n": 3, "seed": 7})),
            [7, 8, 9, 7, 8, 9].map(|s| json!(s))
        );
        assert_eq!(
            seeds(json!({"prompt": "a", "n": 2, "seed": i64::MIN})),
            [json!(i64::MIN), json!(i64::MIN + 1)]
        );
        // -1 and null mean "unseeded"; every choice samples independently.
        assert_eq!(
            seeds(json!({"prompt": "a", "n": 2, "seed": -1})),
            [json!(-1), json!(-1)]
        );
        assert_eq!(
            seeds(json!({"prompt": "a", "n": 2, "seed": null})),
            [Value::Null, Value::Null]
        );
        assert_eq!(
            seeds(json!({"prompt": "a", "n": 1, "seed": i64::MAX})),
            [json!(i64::MAX)]
        );
        // A child seed past int64 cannot be sampled by the engine; reject up front.
        assert!(expand(json!({"prompt": "a", "n": 2, "seed": i64::MAX}), false).is_err());
    }

    #[test]
    fn greedy_sampling_requires_a_single_choice() {
        assert!(reject_greedy_choices(2, Some(0.0), None).is_err());
        // An omitted temperature inherits the model's generation-config default.
        assert!(reject_greedy_choices(2, None, Some(0.0)).is_err());
        assert!(reject_greedy_choices(2, Some(0.7), Some(0.0)).is_ok());
        assert!(reject_greedy_choices(2, None, None).is_ok());
        assert!(reject_greedy_choices(1, Some(0.0), None).is_ok());
        assert!(reject_greedy_choices(2, Some(SAMPLING_EPS), None).is_ok());
    }

    #[test]
    fn config_rejects_private_key_fields() {
        assert!(
            serde_json::from_value::<DphnNetworkConfig>(json!({"private_key": "secret"})).is_err()
        );
    }
}
