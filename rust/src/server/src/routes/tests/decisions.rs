// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! HTTP round trips of `/v1/decisions` and `/v1/systemone` against a mock
//! engine, with the answers the Python frontend computes for the same
//! logprobs.

use std::sync::{Arc, Mutex};

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use serial_test::serial;
use sonar_engine_core_client::protocol::logprobs::{Logprobs, PositionLogprobs, TokenLogprob};
use sonar_engine_core_client::protocol::output::{EngineCoreFinishReason, RequestBatchOutputs};
use sonar_engine_core_client::protocol::request::EngineCoreRequest;
use tower::Service as _;

use super::*;
use crate::config::{DecisionsConfig, LogprobsMode};

const USER: u32 = 300;
const SYSTEM: u32 = 301;
const ASSISTANT: u32 = 302;
const YES: u32 = 310;
const NO: u32 = 311;
const CHANNEL_OPEN: u32 = 312;
const CHANNEL_CLOSE: u32 = 313;
const TURN_END: u32 = 314;
const PAD: u32 = 315;
const VOCAB: usize = 1000;

fn tokenizer() -> TestTokenizer {
    TestTokenizer::new()
        .with_special_token("<|user|>", USER)
        .with_special_token("<|system|>", SYSTEM)
        .with_special_token("<|assistant|>", ASSISTANT)
        .with_regular_token("yes", YES)
        .with_regular_token("no", NO)
        .with_special_token("<|channel>", CHANNEL_OPEN)
        .with_special_token("<channel|>", CHANNEL_CLOSE)
        .with_special_token("<turn|>", TURN_END)
        .with_special_token("<pad>", PAD)
        .with_vocab_size(VOCAB)
}

/// A chat backend that renders `<|system|>`, `<|user|>` and `<|assistant|>`
/// turns and serves the given architecture.
#[derive(Clone, Debug)]
struct DecisionsBackend {
    hf_config: Value,
}

impl DecisionsBackend {
    fn new(architecture: &str) -> Arc<Self> {
        Arc::new(Self {
            hf_config: json!({"architectures": [architecture], "canvas_length": 256}),
        })
    }
}

impl TextBackend for DecisionsBackend {
    fn tokenizer(&self) -> DynTokenizer {
        Arc::new(tokenizer())
    }

    fn model_id(&self) -> &str {
        "test-model"
    }

    fn model_vocab_size(&self) -> usize {
        VOCAB
    }

    fn hf_config(&self) -> &Value {
        &self.hf_config
    }

    fn pad_token_id(&self) -> Option<u32> {
        Some(PAD)
    }
}

impl ChatBackend for DecisionsBackend {
    fn chat_renderer(&self) -> DynChatRenderer {
        Arc::new(self.clone())
    }

    fn new_chat_output_processor(
        &self,
        request: &mut ChatRequest,
        options: NewChatOutputProcessorOptions<'_>,
    ) -> sonar_chat::Result<DynChatOutputProcessor> {
        Ok(Box::new(DefaultChatOutputProcessor::new(
            request,
            "test-model",
            self.tokenizer(),
            options.tool_call_parser,
            options.reasoning_parser,
            options.tool_strict_level,
        )?))
    }
}

impl ChatRenderer for DecisionsBackend {
    fn render(&self, request: &ChatRequest) -> sonar_chat::Result<sonar_chat::RenderedPrompt> {
        let mut prompt = String::new();
        for message in &request.messages {
            let (turn, content) = match message {
                ChatMessage::System { content } => ("<|system|>", content),
                ChatMessage::User { content } => ("<|user|>", content),
                _ => unreachable!("decisions render system and user turns"),
            };
            prompt.push_str(turn);
            prompt.push_str(&render_fake_content(content, "<image>")?);
            prompt.push('\n');
        }
        if request.chat_options.add_generation_prompt() {
            prompt.push_str("<|assistant|>\n");
        }
        Ok(sonar_chat::RenderedPrompt {
            prompt: Prompt::Text(prompt),
            media_order: None,
            effective_template_kwargs: request.chat_options.template_kwargs.clone(),
        })
    }
}

/// The logprob the mock engine reports for each token.
fn logprob(token_id: u32) -> f32 {
    match token_id {
        65 => -0.25, // A
        66 => -1.5,  // B
        67 => -2.5,  // C
        YES => -0.5,
        NO => -2.0,
        48 => -3.0,  // 0
        49 => -0.75, // 1
        50 => -1.25, // 2
        id => -0.1 - (id % 7) as f32 / 2.0,
    }
}

/// Every position of the reply holds each requested label's logprob.
fn respond(request: &EngineCoreRequest) -> EngineCoreOutput {
    let params = request.sampling_params.as_ref().expect("a generate request");
    let ids = params.logprob_token_ids.clone().unwrap_or_default();
    let width = params.max_tokens as usize;
    let position = PositionLogprobs {
        entries: ids
            .iter()
            .map(|&token_id| TokenLogprob {
                token_id,
                logprob: logprob(token_id),
                rank: 1,
            })
            .collect(),
    };
    let first = ids.first().copied().unwrap_or(0);
    request_output_with_logprobs(
        &request.request_id,
        vec![first; width],
        Some(EngineCoreFinishReason::Length),
        None,
        Some(Logprobs {
            positions: vec![position; width],
        }),
        None,
    )
}

type Requests = Arc<Mutex<Vec<EngineCoreRequest>>>;

async fn decisions_app(
    backend: Arc<DecisionsBackend>,
    config: DecisionsConfig,
    respond: fn(&EngineCoreRequest) -> EngineCoreOutput,
) -> (axum::Router, MockEngineTask, Requests) {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let requests: Requests = Arc::default();
    let seen = requests.clone();
    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        b"engine-decisions".to_vec(),
        move |dealer, push| {
            boxed_test_future(async move {
                loop {
                    let frames = recv_engine_message(dealer).await;
                    if frames[0].as_ref() != [0] {
                        continue;
                    }
                    let request = decode_add_request(&frames);
                    let output = respond(&request);
                    seen.lock().unwrap().push(request);
                    send_outputs(
                        push,
                        RequestBatchOutputs {
                            outputs: vec![output],
                            ..Default::default()
                        }
                        .into(),
                    )
                    .await;
                }
            })
        },
    ));
    let client = EngineCoreClient::connect(
        EngineCoreClientConfig::new_single(handshake_address)
            .with_model_name("test-model")
            .with_local_input_output_addresses(
                Some(ipc.input_endpoint()),
                Some(ipc.output_endpoint()),
            ),
    )
    .await
    .expect("connect client");
    let chat = ChatLlm::from_shared_backend(test_llm(client), backend);
    let state = AppState::new(vec!["test-model".to_string()], chat).with_decisions_config(config);
    (build_router(Arc::new(state)), engine_task, requests)
}

async fn post(app: &axum::Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .header("X-Request-Id", "test-request")
                .body(Body::from(body.to_string()))
                .expect("build request"),
        )
        .await
        .expect("call app");
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    (status, serde_json::from_slice(&body).expect("JSON body"))
}

/// Assert that `actual` is `expected`. serde_json's literal parser can miss
/// the last bit of a float, so numbers compare within one rounding error.
fn assert_json_matches(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) if a.is_f64() || e.is_f64() => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            assert!((a - e).abs() <= e.abs() * 4.0 * f64::EPSILON, "{a} != {e}");
        }
        (Value::Object(a), Value::Object(e)) => {
            let keys =
                |map: &serde_json::Map<String, Value>| map.keys().cloned().collect::<Vec<_>>();
            assert_eq!(keys(a), keys(e));
            for (key, value) in a {
                assert_json_matches(value, &e[key]);
            }
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(a.len(), e.len());
            a.iter().zip(e).for_each(|(a, e)| assert_json_matches(a, e));
        }
        (a, e) => assert_eq!(a, e),
    }
}

fn prompt_text(request: &EngineCoreRequest) -> String {
    let ids = decode_prompt_token_ids(request).expect("token prompt");
    tokenizer().decode(&ids, false).expect("decode prompt")
}

fn three_questions() -> Value {
    json!({
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payments", "shipping": null}},
        "urgent": {"type": "noul", "instructions": "Urgent?"},
        "tone": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]},
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn systemone_next_token_reads_match_python() {
    let (app, _engine, requests) = decisions_app(
        DecisionsBackend::new("Qwen3ForCausalLM"),
        DecisionsConfig::default(),
        respond,
    )
    .await;
    let (status, mut body) = post(
        &app,
        "/v1/systemone",
        json!({
            "model": "test-model",
            "state": {"team": "billing", "language": "Français"},
            "questions": three_questions(),
            "instructions": "Be brief.",
            "cache_salt": "tenant-a",
            "priority": 3,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["id"], "decision-test-request");
    assert!(body["created"].is_u64());
    body.as_object_mut().unwrap().retain(|key, _| key != "id" && key != "created");
    // The Python frontend's answers for the same label logprobs.
    let expected = json!({
        "object": "structured_decision",
        "model": "test-model",
        "answers": {
            "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.7772998611746911, "shipping": 0.22270013882530884}, "confidence": 0.7788007830714048},
            "urgent": {"type": "noul", "noul": 0.8175744761936437, "probabilities": {"yes": 0.8175744761936437, "no": 0.18242552380635632}, "confidence": 0.6065306597126334},
            "tone": {"type": "score", "score": 1.2927289486293096, "legend": {"0": "calm", "1": "annoyed", "2": "furious"}, "probabilities": {"0": 0.061567489141958455, "1": 0.5841360730867737, "2": 0.3542964377712679}, "confidence": 0.47236655274101474},
        },
        "usage": {"input_tokens": body["usage"]["input_tokens"], "output_tokens": 3},
        "diagnostics": {
            // serde_json's literal parser can miss the last digit, so parse
            // Python's value exactly.
            "team": {"label_mass": "1.0019309432198347".parse::<f64>().unwrap(), "argmax_is_label": true},
            "urgent": {"label_mass": 0.7418659429492461, "argmax_is_label": true},
            "tone": {"label_mass": 0.8086584179690687, "argmax_is_label": true},
        },
        "joint": null,
    });
    assert_json_matches(&body, &expected);
    // Keys keep the Python order.
    let keys: Vec<_> = body["answers"].as_object().unwrap().keys().cloned().collect();
    assert_eq!(keys, ["team", "urgent", "tone"]);

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    let mut input_tokens = 0;
    for (i, request) in requests.iter().enumerate() {
        let params = request.sampling_params.as_ref().unwrap();
        assert_eq!(params.max_tokens, 1);
        assert_eq!(params.temperature, 0.0);
        assert_eq!(request.cache_salt.as_deref(), Some("tenant-a"));
        assert_eq!(request.priority, 3);
        assert_eq!(
            request.external_req_id.as_deref(),
            Some(format!("decision-test-request-{i}").as_str())
        );
        input_tokens += decode_prompt_token_ids(request).unwrap().len();
    }
    assert_eq!(body["usage"]["input_tokens"], input_tokens);
    let label_ids =
        |i: usize| requests[i].sampling_params.as_ref().unwrap().logprob_token_ids.clone();
    assert_eq!(label_ids(0), Some(vec![65, 66]));
    assert_eq!(label_ids(1), Some(vec![YES, NO]));
    assert_eq!(label_ids(2), Some(vec![48, 49, 50]));
    assert_eq!(
        prompt_text(&requests[0]),
        "<|system|>Be brief.\n<|user|>{\"team\": \"billing\", \"language\": \"Français\"}\n\n\
         Question: Which team?\nA: billing - payments\nB: shipping\n\
         Answer with the letter of one option only.\n<|assistant|>\n"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn decisions_next_token_reads_match_python() {
    let (app, _engine, requests) = decisions_app(
        DecisionsBackend::new("Qwen3ForCausalLM"),
        DecisionsConfig::default(),
        respond,
    )
    .await;
    let (status, body) = post(
        &app,
        "/v1/decisions",
        json!({
            "model": "test-model",
            "input": [{"role": "user", "content": "evidence"}],
            "safety_identifier": "test-user",
            "questions": [
                {"type": "predicate", "instructions": "Damaged?"},
                {"type": "choice", "name": "decision", "instructions": "Pick.", "choices": [{"value": true}, {"value": "true"}]},
                {"type": "score", "instructions": "Rate.", "levels": [{"label": "low"}, {"label": "mid"}, {"label": "high", "description": "worst"}]},
            ],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // Every Decisions question reads letters, so the predicate reads A and B.
    let requests = requests.lock().unwrap();
    let input_tokens: usize =
        requests.iter().map(|r| decode_prompt_token_ids(r).unwrap().len()).sum();
    let p_true = 0.22270013882530884;
    let (p0, p1, p2) = (0.7184403411510649, 0.20583660399765155, 0.07572305485128351);
    assert_json_matches(
        &body,
        &json!({
            "model": "test-model",
            "answers": [
                {"type": "predicate", "name": null, "probability": p_true},
                {"type": "choice", "name": "decision", "choice": true, "probabilities": [
                    {"value": true, "probability": 0.7772998611746911},
                    {"value": "true", "probability": p_true},
                ], "confidence": 0.7788007830714048},
                {"type": "score", "name": null, "score": 0.3572827137002186, "probabilities": [
                    {"value": 0, "label": "low", "probability": p0},
                    {"value": 1, "label": "mid", "probability": p1},
                    {"value": 2, "label": "high", "probability": p2},
                ], "confidence": 0.7788007830714048},
            ],
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": input_tokens + 3,
            },
        }),
    );
    assert!(prompt_text(&requests[1]).contains("A: true\nB: \"true\"\n"));
    assert!(prompt_text(&requests[2]).contains("C: high - worst\n"));
    assert!(
        prompt_text(&requests[0])
            .starts_with("<|user|>evidence\n\nQuestion: Damaged?\nA: false\nB: true\n")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn liquid_reads_with_the_checkpoint_prompt() {
    let (app, _engine, requests) = decisions_app(
        DecisionsBackend::new("Lfm2VlForConditionalGeneration"),
        DecisionsConfig::default(),
        respond,
    )
    .await;
    let (status, body) = post(
        &app,
        "/v1/systemone",
        json!({"state": {"k": "é"}, "questions": three_questions()}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["answers"]["team"]["choice"], "billing");
    {
        let requests = requests.lock().unwrap();
        assert_eq!(
            prompt_text(&requests[0]),
            "<|user|>{\n  \"k\": \"é\"\n}\n\n\nQUESTION:\nWhich team?\n\nOptions:\nA payments\nB shipping\n\n\
             Reply with the option code only.\n<|assistant|>\n"
        );
        // Only "yes" and "no" are one token here, so Liquid's other spellings
        // drop out of the noul's label groups.
        assert_eq!(
            requests[1].sampling_params.as_ref().unwrap().logprob_token_ids,
            Some(vec![YES, NO])
        );
    }

    // Liquid shrinks pictures before its processor, which the Rust frontend
    // does not do.
    let (status, body) = post(
        &app,
        "/v1/systemone",
        json!({"state": "x", "questions": three_questions(), "images": ["data:image/png;base64,AA=="]}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("SONAR_USE_RUST_FRONTEND=0"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn diffusion_gemma_reads_canvases() {
    let (app, _engine, requests) = decisions_app(
        DecisionsBackend::new("DiffusionGemmaForBlockDiffusion"),
        DecisionsConfig::default(),
        respond,
    )
    .await;
    let body = json!({"state": "Charged twice.", "questions": three_questions(), "joint": true});
    let (status, joint) = post(&app, "/v1/systemone", body.clone()).await;
    assert_eq!(status, StatusCode::OK, "{joint}");
    let canvas = &joint["joint"]["canvases"][0];
    assert_eq!(canvas["questions"], json!(["team", "urgent", "tone"]));
    // Three lines of four tokens, the end of the turn and a pad row round up
    // to two steps of 16.
    assert_eq!(canvas["width"], 32);
    let slots: Vec<usize> = canvas["slots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_u64().unwrap() as usize)
        .collect();
    {
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        let params = request.sampling_params.as_ref().unwrap();
        let extra = params.extra_args.as_ref().unwrap();
        let seed: Vec<u32> =
            serde_json::from_value(extra["diffusion_seed_canvas"].clone()).unwrap();
        assert_eq!(extra["diffusion_canvas_length"], 32);
        assert_eq!(extra["diffusion_max_steps"], 1);
        assert_eq!(extra["diffusion_read_only"], true);
        assert_eq!(params.max_tokens as usize, slots[2] + 1);
        let template: Vec<u32> =
            (0..slots[2] + 3).filter(|i| !slots.contains(i)).map(|i| seed[i]).collect();
        assert_eq!(
            tokenizer().decode(&template, false).unwrap(),
            "1: \n2: \n3: \n<turn|>"
        );
        assert_eq!(*seed.last().unwrap(), PAD);
        let prompt = prompt_text(request);
        assert!(prompt.ends_with("Reply with one line per question, in order, formatted as \"number: answer\".\n<|assistant|>\n<|channel>thought\n<channel|>"), "{prompt}");
        assert_eq!(
            params.logprob_token_ids,
            Some(vec![48, 49, 50, 65, 66, YES, NO])
        );
    }
    assert_eq!(
        joint["answers"]["team"]["probabilities"]["billing"],
        0.7772998611746911
    );
    assert_eq!(joint["usage"]["output_tokens"], slots[2] + 1);

    // Without joint, each question reads its own canvas with the same answers.
    let (status, alone) = post(
        &app,
        "/v1/systemone",
        json!({"state": "Charged twice.", "questions": three_questions()}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{alone}");
    assert_eq!(alone["answers"], joint["answers"]);
    assert_eq!(alone["joint"], Value::Null);
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 4);
    let params = requests[1].sampling_params.as_ref().unwrap();
    assert_eq!(params.max_tokens, 2);
    let seed = &params.extra_args.as_ref().unwrap()["diffusion_seed_canvas"];
    assert_eq!(seed.as_array().unwrap().len(), 16);
    assert_eq!(seed[1], TURN_END);
}

fn missing_label(request: &EngineCoreRequest) -> EngineCoreOutput {
    let mut output = respond(request);
    if let Some(MaybeWireLogprobs::Direct(logprobs)) = &mut output.new_logprobs {
        logprobs.positions[0].entries.pop();
    }
    output
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn decisions_errors_match_python() {
    let one = json!({"q": {"type": "choice", "criteria": {"yes": null, "no": null}}});
    let (app, _engine, requests) = decisions_app(
        DecisionsBackend::new("Qwen3ForCausalLM"),
        DecisionsConfig::default(),
        missing_label,
    )
    .await;
    let cases = [
        (
            json!({"model": "other", "state": "x", "questions": one}),
            StatusCode::NOT_FOUND,
            "does not exist",
        ),
        (
            json!({"state": "x", "questions": one, "chat_template_kwargs": {"chat_template": "{{ 1 }}"}}),
            StatusCode::BAD_REQUEST,
            "--trust-request-chat-template",
        ),
        (
            json!({"state": "x", "questions": one, "joint": true}),
            StatusCode::BAD_REQUEST,
            "joint reads need DiffusionGemma",
        ),
        (
            json!({"state": "x", "questions": one, "samples": 4}),
            StatusCode::BAD_REQUEST,
            "samples",
        ),
        (
            json!({"state": "x", "questions": one, "videos": ["v"]}),
            StatusCode::BAD_REQUEST,
            "this model does not take video inputs",
        ),
        (
            json!({"state": "x", "questions": one, "images": ["i"]}),
            StatusCode::BAD_REQUEST,
            "this model does not take image inputs",
        ),
        (
            json!({"state": "x", "questions": one, "chat_template_kwargs": {"enable_thinking": true}}),
            StatusCode::BAD_REQUEST,
            "thinking must be off",
        ),
        (
            json!({"state": "x", "questions": {"q": {"type": "noul", "options": 1}}}),
            StatusCode::BAD_REQUEST,
            "question 'q': unknown field(s) ['options']",
        ),
    ];
    for (body, status, message) in cases {
        let (actual, response) = post(&app, "/v1/systemone", body.clone()).await;
        assert_eq!(actual, status, "{body}: {response}");
        assert!(
            response["error"]["message"].as_str().unwrap().contains(message),
            "{body}: {response}"
        );
    }
    let many: serde_json::Map<String, Value> =
        (0..65).map(|i| (i.to_string(), one["q"].clone())).collect();
    let (status, response) = post(
        &app,
        "/v1/systemone",
        json!({"state": "x", "questions": many}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(response["error"]["message"].as_str().unwrap().contains("at most 64"));
    let questions = vec![json!({"type": "predicate", "instructions": "x"}); 65];
    let (status, response) = post(
        &app,
        "/v1/decisions",
        json!({"model": "test-model", "input": "x", "questions": questions}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(response["error"]["message"].as_str().unwrap().contains("at most 64"));
    assert!(requests.lock().unwrap().is_empty());

    // A read without a label's logprob is a server error, like Python's
    // GenerationError.
    let (status, response) = post(
        &app,
        "/v1/systemone",
        json!({"state": "x", "questions": one}),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(response["error"]["type"], "InternalServerError");
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not found in logprobs for item 0")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn unsupported_models_answer_not_implemented() {
    let systemone =
        json!({"state": "x", "questions": {"q": {"type": "choice", "criteria": {"a": null}}}});
    let decisions = json!({"model": "test-model", "input": "x", "questions": [{"type": "predicate", "instructions": "x"}]});
    for (architecture, config, systemone_message) in [
        (
            "D3ForSequenceClassification",
            DecisionsConfig::default(),
            "Structured decisions on D3ForSequenceClassification use D3Strategy, which reads through the model's classify head with pooling requests. The Rust frontend does not send pooling requests yet. Set SONAR_USE_RUST_FRONTEND=0 to use the Python frontend.",
        ),
        (
            "LlamaForCausalLM",
            DecisionsConfig::default(),
            "The model does not support the structured decision API",
        ),
        (
            "Qwen3ForCausalLM",
            DecisionsConfig {
                logprobs_mode: LogprobsMode::RawLogits,
                ..DecisionsConfig::default()
            },
            "The model does not support the structured decision API",
        ),
    ] {
        let (app, _engine, _) =
            decisions_app(DecisionsBackend::new(architecture), config, respond).await;
        let (status, response) = post(&app, "/v1/systemone", systemone.clone()).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{architecture}");
        assert_eq!(response["error"]["type"], "NotImplementedError");
        assert_eq!(response["error"]["message"], systemone_message);
        let (status, response) = post(&app, "/v1/decisions", decisions.clone()).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{architecture}");
        assert_eq!(
            response["error"]["message"],
            "The model does not support the Decisions API"
        );
    }
}
