// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `/v1/decisions` (the OpenAI Decisions API) and `/v1/systemone` (structured
//! decisions): each question is answered by reading the probability of every
//! option's label as the reply's first token, or from a diffusion canvas.
//!
//! Mirrors `sonar/entrypoints/openai/decisions` and
//! `sonar/entrypoints/generate/structured_decisions`.

mod protocol;
mod py;
mod questions;
mod strategy;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value, json};
use tracing::info;

use self::protocol::{
    DecisionQuestion, DecisionQuestionKind, SystemOneRequest, parse_decisions_request,
    parse_systemone_request,
};
use self::questions::{
    DecisionOption, LETTERS, Question, QuestionType, argmax, build_question, expected_level,
};
pub(crate) use self::strategy::Availability;
use self::strategy::{MAX_OPTIONS, MAX_QUESTIONS, QuestionRead, ReadContext, Reader};
use crate::error::ApiError;
use crate::lora::LoraModelResolution;
use crate::state::AppState;
use crate::utils::{resolve_request_context, unix_timestamp};

/// An invalid decision request. Answered with a 400.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DecisionError(String);

impl DecisionError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl From<DecisionError> for ApiError {
    fn from(error: DecisionError) -> Self {
        ApiError::invalid_request(error.0, None)
    }
}

const UNTRUSTED_TEMPLATE: &str = "Chat template is passed with request, but \
     --trust-request-chat-template is not set. Refused request with untrusted chat template.";

/// The reader for the served model, or the 501 that explains why there is
/// none. `endpoint` names the API like Python's 501 does.
async fn reader<'s>(state: &'s AppState, endpoint: &str) -> Result<&'s Reader, ApiError> {
    let availability = state
        .decisions
        .get_or_init(|| Availability::resolve(state, state.decisions_config))
        .await;
    match availability {
        Availability::Ready(reader) => Ok(reader),
        // The Python frontend serves `/v1/decisions` only with generative reads
        // too.
        Availability::PythonOnly { message } if endpoint == SYSTEMONE => {
            Err(ApiError::not_implemented(message.clone()))
        }
        _ => Err(ApiError::not_implemented(format!(
            "The model does not support the {endpoint}"
        ))),
    }
}

const SYSTEMONE: &str = "structured decision API";
const DECISIONS: &str = "Decisions API";

/// Resolve the requested model, which must be served, and its LoRA adapter.
async fn resolve_model(
    state: &AppState,
    model: Option<&str>,
) -> Result<LoraModelResolution, ApiError> {
    let model = model.filter(|model| !model.is_empty());
    let resolution = state.resolve_model_with_loras(model).await;
    resolution.model_names.check(model)?;
    Ok(resolution)
}

/// Python admits joint reads only on models that read a canvas, and reads a
/// single question on its own.
fn admit_joint(reader: &Reader, joint: bool, questions: usize) -> Result<bool, ApiError> {
    if joint && !reader.joint_reads() {
        return Err(DecisionError::new(
            "joint: this model reads each question on its own; joint reads need DiffusionGemma",
        )
        .into());
    }
    Ok(joint && questions > 1)
}

fn into_response(result: Result<Value, ApiError>) -> Response {
    match result {
        Ok(body) => Json(body).into_response(),
        Err(error) => error.into_response(),
    }
}

/// `POST /v1/systemone`.
pub async fn systemone(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    into_response(create_structured_decision(&state, &headers, body).await)
}

async fn create_structured_decision(
    state: &AppState,
    headers: &HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Value, ApiError> {
    let Json(body) = body?;
    let request = parse_systemone_request(&body)?;
    let reader = reader(state, SYSTEMONE).await?;
    let lora = resolve_model(state, request.model.as_deref()).await?;
    let config = state.decisions_config;
    if !config.trust_request_chat_template
        && request
            .chat_template_kwargs
            .as_ref()
            .and_then(|kwargs| kwargs.get("chat_template"))
            .is_some_and(|template| !template.is_null())
    {
        return Err(ApiError::invalid_request(UNTRUSTED_TEMPLATE, None));
    }
    let base_id = resolve_request_context(headers, request.request_id.as_deref()).request_id;
    let request_id = format!("decision-{base_id}");

    let questions = parse_questions(&request)?;
    let images = request.images.clone().unwrap_or_default();
    reader.check_media(
        !images.is_empty(),
        request.audio.as_ref().is_some_and(|audio| !audio.is_empty()),
        request.videos.as_ref().is_some_and(|videos| !videos.is_empty()),
    )?;
    let joint = admit_joint(reader, request.joint, questions.len())?;
    let priority = i32::try_from(request.priority).map_err(|_| {
        ApiError::invalid_request(
            format!(
                "priority {} is out of range for this frontend, which takes 32-bit priorities",
                request.priority
            ),
            Some("priority"),
        )
    })?;
    if state.api_server_options.enable_log_requests {
        info!(
            request_id,
            questions = questions.len(),
            joint,
            "Received request"
        );
    }
    let ctx = ReadContext {
        request_id: request_id.clone(),
        chat_template_kwargs: request.chat_template_kwargs.clone(),
        lora_request: lora.lora_request.clone(),
        priority,
        cache_salt: request.cache_salt.clone(),
    };
    let state_text = reader.render_state(&request.state);
    let instructions = request.instructions.as_deref();
    let reads = if joint {
        reader
            .read_joint(state, &ctx, &questions, instructions, &state_text, &images)
            .await?
    } else {
        reader.read(state, &ctx, &questions, instructions, &state_text, &images).await?
    };

    let mut answers = Map::new();
    let mut diagnostics = Map::new();
    let mut canvases: Vec<Value> = Vec::new();
    for (question, read) in questions.iter().zip(&reads) {
        answers.insert(
            question.id.clone(),
            question.kind.answer(question, &read.probs, read.label_mass),
        );
        diagnostics.insert(
            question.id.clone(),
            json!({"label_mass": read.label_mass, "argmax_is_label": read.argmax_is_label}),
        );
        if let Some(slot) = read.slot {
            if slot.canvas == canvases.len() {
                canvases.push(json!({"questions": [], "width": slot.width, "slots": []}));
            }
            let canvas = canvases.last_mut().expect("a canvas was just added");
            canvas["questions"].as_array_mut().expect("a list").push(json!(question.id));
            canvas["slots"].as_array_mut().expect("a list").push(json!(slot.position));
        }
    }
    Ok(json!({
        "id": request_id,
        "object": "structured_decision",
        "created": unix_timestamp(),
        "model": lora.response_model(),
        "answers": answers,
        "usage": {
            "input_tokens": reads.iter().map(|read| read.input_tokens).sum::<usize>(),
            "output_tokens": reads.iter().map(|read| read.output_tokens).sum::<usize>(),
        },
        "diagnostics": diagnostics,
        "joint": joint.then(|| json!({"canvases": canvases})),
    }))
}

fn parse_questions(request: &SystemOneRequest) -> Result<Vec<Question>, ApiError> {
    if request.questions.is_empty() {
        return Err(DecisionError::new("questions: needs at least one question").into());
    }
    if request.questions.len() > MAX_QUESTIONS {
        return Err(DecisionError::new(format!(
            "questions: at most {MAX_QUESTIONS} for this model"
        ))
        .into());
    }
    request
        .questions
        .iter()
        .map(|(qid, spec)| {
            if !spec.extra.is_empty() {
                let mut extra: Vec<&String> = spec.extra.iter().collect();
                extra.sort();
                let extra: Vec<String> = extra.into_iter().map(|key| py::repr_str(key)).collect();
                return Err(DecisionError::new(format!(
                    "question {}: unknown field(s) [{}]",
                    py::repr_str(qid),
                    extra.join(", ")
                ))
                .into());
            }
            build_question(
                qid,
                &spec.type_name,
                &spec.instructions,
                &spec.criteria,
                MAX_OPTIONS,
            )
            .map_err(ApiError::from)
        })
        .collect()
}

/// `POST /v1/decisions`.
pub async fn decisions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    into_response(create_decisions(&state, &headers, body).await)
}

async fn create_decisions(
    state: &AppState,
    headers: &HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Value, ApiError> {
    let Json(body) = body?;
    let request = parse_decisions_request(&body)?;
    let reader = reader(state, DECISIONS).await?;
    let lora = resolve_model(state, Some(&request.model)).await?;
    let request_id = format!(
        "decision-{}",
        resolve_request_context(headers, None).request_id
    );
    if request.questions.len() > MAX_QUESTIONS {
        return Err(DecisionError::new(format!(
            "questions: at most {MAX_QUESTIONS} for this model"
        ))
        .into());
    }
    let questions: Vec<Question> = request
        .questions
        .iter()
        .enumerate()
        .map(|(i, question)| read_question(i, question))
        .collect::<Result<_, _>>()?;
    let joint = admit_joint(reader, request.joint, questions.len())?;
    if state.api_server_options.enable_log_requests {
        info!(
            request_id,
            questions = questions.len(),
            joint,
            "Received request"
        );
    }
    let ctx = ReadContext {
        request_id,
        chat_template_kwargs: None,
        lora_request: lora.lora_request.clone(),
        priority: 0,
        cache_salt: request.cache_salt.clone(),
    };
    let reads = if joint {
        reader
            .read_joint(state, &ctx, &questions, None, &request.input_text, &[])
            .await?
    } else {
        reader.read(state, &ctx, &questions, None, &request.input_text, &[]).await?
    };

    let answers: Vec<Value> = request
        .questions
        .iter()
        .zip(&reads)
        .map(|(question, read)| decision_answer(question, read))
        .collect();
    let input_tokens: usize = reads.iter().map(|read| read.input_tokens).sum();
    let output_tokens: usize = reads.iter().map(|read| read.output_tokens).sum();
    Ok(json!({
        "model": lora.response_model(),
        "answers": answers,
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {
                "cached_tokens": reads.iter().map(|read| read.cached_tokens).sum::<usize>(),
                "cache_write_tokens": reads.iter().map(|read| read.cache_write_tokens).sum::<usize>(),
            },
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        },
    }))
}

/// A Decisions API question as a labeled choice read.
fn read_question(index: usize, question: &DecisionQuestion) -> Result<Question, ApiError> {
    let option = |name: String, description: &str| DecisionOption {
        name,
        description: Some(description.to_string()),
    };
    let options: Vec<DecisionOption> = match &question.kind {
        DecisionQuestionKind::Predicate => vec![
            DecisionOption {
                name: "false".to_string(),
                description: None,
            },
            DecisionOption {
                name: "true".to_string(),
                description: None,
            },
        ],
        DecisionQuestionKind::Choice(choices) => choices
            .iter()
            .map(|choice| {
                option(
                    py::json_dumps(&choice.value.to_json(), None),
                    &choice.description,
                )
            })
            .collect(),
        DecisionQuestionKind::Score(levels) => levels
            .iter()
            .map(|level| option(level.label.clone(), &level.description))
            .collect(),
    };
    let limit = MAX_OPTIONS.min(LETTERS.len());
    if options.len() > limit {
        return Err(DecisionError::new(format!(
            "This model supports at most {limit} choices per question"
        ))
        .into());
    }
    Ok(Question::labeled(
        index.to_string(),
        QuestionType::Choice,
        question.instructions.clone(),
        options,
    ))
}

/// A Decisions API answer from a question's label probabilities.
fn decision_answer(question: &DecisionQuestion, read: &QuestionRead) -> Value {
    let probs = &read.probs;
    let name = json!(question.name);
    match &question.kind {
        DecisionQuestionKind::Predicate => {
            json!({"type": "predicate", "name": name, "probability": probs[1]})
        }
        DecisionQuestionKind::Choice(choices) => {
            let top = argmax(probs);
            let probabilities: Vec<Value> = choices
                .iter()
                .zip(probs)
                .map(|(choice, p)| json!({"value": choice.value.to_json(), "probability": p}))
                .collect();
            json!({
                "type": "choice",
                "name": name,
                "choice": choices[top].value.to_json(),
                "probabilities": probabilities,
                "confidence": probs[top] * read.label_mass,
            })
        }
        DecisionQuestionKind::Score(levels) => {
            let top = argmax(probs);
            let probabilities: Vec<Value> = levels
                .iter()
                .zip(probs)
                .enumerate()
                .map(|(i, (level, p))| json!({"value": i, "label": level.label, "probability": p}))
                .collect();
            json!({
                "type": "score",
                "name": name,
                "score": expected_level(probs),
                "probabilities": probabilities,
                "confidence": probs[top] * read.label_mass,
            })
        }
    }
}
