// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Request validation for `/v1/decisions` and `/v1/systemone`.
//!
//! Mirrors the pydantic models in
//! `sonar/entrypoints/openai/decisions/protocol.py` and
//! `sonar/entrypoints/generate/structured_decisions/protocol.py`. Invalid
//! bodies are a 400 that lists every error like FastAPI's validation handler.

use serde_json::{Map, Value};
use sonar_text::is_valid_cache_salt;

use super::py;
use crate::error::ApiError;

/// `ShortText` in the Python protocol.
const SHORT_TEXT: usize = 1_048_576;
/// `InputText` in the Python protocol.
const INPUT_TEXT: usize = 10_485_760;
const MAX_INPUT_MESSAGES: usize = 131_072;
const MAX_INPUT_PARTS: usize = 16_384;
const MAX_QUESTIONS: usize = 200;
const MAX_CHOICES: usize = 255;
const MAX_LEVELS: usize = 10;
const MAX_SAFETY_IDENTIFIER: usize = 128;

#[derive(Debug, Clone)]
enum Loc {
    Key(String),
    Index(usize),
}

impl Loc {
    fn repr(&self) -> String {
        match self {
            Self::Key(key) => py::repr_str(key),
            Self::Index(index) => index.to_string(),
        }
    }
}

impl From<&str> for Loc {
    fn from(key: &str) -> Self {
        Self::Key(key.to_string())
    }
}

impl From<usize> for Loc {
    fn from(index: usize) -> Self {
        Self::Index(index)
    }
}

#[derive(Debug)]
struct FieldError {
    kind: &'static str,
    loc: Vec<Loc>,
    msg: String,
    input: Value,
}

/// The validation errors of one request body, in pydantic's order.
#[derive(Debug, Default)]
struct Errors(Vec<FieldError>);

impl Errors {
    fn push(&mut self, kind: &'static str, loc: &[Loc], msg: impl Into<String>, input: &Value) {
        self.0.push(FieldError {
            kind,
            loc: loc.to_vec(),
            msg: msg.into(),
            input: input.clone(),
        });
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    /// FastAPI's validation handler message: every error on its own line.
    fn into_api_error(self) -> ApiError {
        let count = self.0.len();
        let label = if count == 1 { "error" } else { "errors" };
        let mut message = format!("{count} validation {label}:\n");
        for error in self.0 {
            let mut loc: Vec<String> = vec![py::repr_str("body")];
            loc.extend(error.loc.iter().map(Loc::repr));
            let loc = if loc.len() == 1 {
                format!("({},)", loc[0])
            } else {
                format!("({})", loc.join(", "))
            };
            message.push_str(&format!(
                "  {{'type': {}, 'loc': {loc}, 'msg': {}, 'input': {}}}\n",
                py::repr_str(error.kind),
                py::repr_str(&error.msg),
                py::repr(&error.input),
            ));
        }
        ApiError::json_parse_error(message.trim_end().to_string())
    }

    fn finish<T>(self, value: T) -> Result<T, ApiError> {
        if self.0.is_empty() {
            Ok(value)
        } else {
            Err(self.into_api_error())
        }
    }
}

fn at(loc: &[Loc], key: impl Into<Loc>) -> Vec<Loc> {
    let mut loc = loc.to_vec();
    loc.push(key.into());
    loc
}

fn items(n: usize) -> &'static str {
    if n == 1 { "item" } else { "items" }
}

fn strict_str(errors: &mut Errors, loc: &[Loc], value: &Value, max_len: usize) -> Option<String> {
    let Value::String(s) = value else {
        errors.push("string_type", loc, "Input should be a valid string", value);
        return None;
    };
    if s.chars().count() > max_len {
        errors.push(
            "string_too_long",
            loc,
            format!("String should have at most {max_len} characters"),
            value,
        );
        return None;
    }
    Some(s.clone())
}

fn optional_str(
    errors: &mut Errors,
    loc: &[Loc],
    value: Option<&Value>,
    max_len: usize,
) -> Option<String> {
    match value {
        None | Some(Value::Null) => None,
        Some(value) => strict_str(errors, loc, value, max_len),
    }
}

fn strict_bool(errors: &mut Errors, loc: &[Loc], value: Option<&Value>) -> bool {
    match value {
        None => false,
        Some(Value::Bool(value)) => *value,
        Some(value) => {
            errors.push("bool_type", loc, "Input should be a valid boolean", value);
            false
        }
    }
}

fn literal(errors: &mut Errors, loc: &[Loc], value: &Value, expected: &str) {
    if value.as_str() != Some(expected) {
        errors.push(
            "literal_error",
            loc,
            format!("Input should be {}", py::repr_str(expected)),
            value,
        );
    }
}

/// Check a list's length; pydantic checks it before the items.
fn list<'a>(
    errors: &mut Errors,
    loc: &[Loc],
    value: &'a Value,
    min_len: usize,
    max_len: usize,
) -> Option<&'a [Value]> {
    let Value::Array(values) = value else {
        errors.push("list_type", loc, "Input should be a valid list", value);
        return None;
    };
    if values.len() < min_len {
        errors.push(
            "too_short",
            loc,
            format!(
                "List should have at least {min_len} {} after validation, not {}",
                items(min_len),
                values.len()
            ),
            value,
        );
        return None;
    }
    if values.len() > max_len {
        errors.push(
            "too_long",
            loc,
            format!(
                "List should have at most {max_len} {} after validation, not {}",
                items(max_len),
                values.len()
            ),
            value,
        );
        return None;
    }
    Some(values)
}

fn required<'a>(
    errors: &mut Errors,
    loc: &[Loc],
    map: &'a Map<String, Value>,
    key: &str,
    parent: &Value,
) -> Option<&'a Value> {
    let value = map.get(key);
    if value.is_none() {
        errors.push("missing", &at(loc, key), "Field required", parent);
    }
    value
}

fn forbid_extra(errors: &mut Errors, loc: &[Loc], map: &Map<String, Value>, fields: &[&str]) {
    for (key, value) in map {
        if !fields.contains(&key.as_str()) {
            errors.push(
                "extra_forbidden",
                &at(loc, key.as_str()),
                "Extra inputs are not permitted",
                value,
            );
        }
    }
}

/// Python's `validate_cache_salt`, run before the fields are validated.
fn check_cache_salt(body: &Value) -> Result<(), ApiError> {
    let Some(salt) = body.get("cache_salt").filter(|salt| !salt.is_null()) else {
        return Ok(());
    };
    let message = match salt.as_str() {
        Some(salt) if !salt.is_empty() => {
            if is_valid_cache_salt(salt) {
                return Ok(());
            }
            "Parameter 'cache_salt' must be at most 128 characters and must not contain '@', \
             '/', '\\\\', or NUL."
        }
        _ => "Parameter 'cache_salt' must be a non-empty string if provided.",
    };
    let mut errors = Errors::default();
    errors.push("value_error", &[], format!("Value error, {message}"), body);
    errors.finish(())
}

fn body_object(body: &Value) -> Result<&Map<String, Value>, ApiError> {
    match body {
        Value::Object(map) => Ok(map),
        body => {
            let mut errors = Errors::default();
            errors.push(
                "model_attributes_type",
                &[],
                "Input should be a valid dictionary or object to extract fields from",
                body,
            );
            Err(errors.into_api_error())
        }
    }
}

/// A choice's value: a string or a boolean.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum ChoiceValue {
    Str(String),
    Bool(bool),
}

impl ChoiceValue {
    pub(super) fn to_json(&self) -> Value {
        match self {
            Self::Str(s) => Value::String(s.clone()),
            Self::Bool(b) => Value::Bool(*b),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ChoiceOption {
    pub(super) value: ChoiceValue,
    pub(super) description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ScoreLevel {
    pub(super) label: String,
    pub(super) description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum DecisionQuestionKind {
    Predicate,
    Choice(Vec<ChoiceOption>),
    Score(Vec<ScoreLevel>),
}

/// One question of a `/v1/decisions` request.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DecisionQuestion {
    pub(super) instructions: String,
    pub(super) name: Option<String>,
    pub(super) kind: DecisionQuestionKind,
}

/// A validated `/v1/decisions` request.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DecisionsRequest {
    pub(super) model: String,
    /// The input as the model reads it: messages joined by blank lines, and a
    /// message's parts by newlines.
    pub(super) input_text: String,
    pub(super) questions: Vec<DecisionQuestion>,
    pub(super) joint: bool,
    pub(super) cache_salt: Option<String>,
}

const DECISIONS_FIELDS: &[&str] = &[
    "model",
    "input",
    "questions",
    "safety_identifier",
    "joint",
    "cache_salt",
];

pub(super) fn parse_decisions_request(body: &Value) -> Result<DecisionsRequest, ApiError> {
    check_cache_salt(body)?;
    let map = body_object(body)?;
    let mut errors = Errors::default();
    let model = required(&mut errors, &[], map, "model", body)
        .and_then(|value| strict_str(&mut errors, &["model".into()], value, SHORT_TEXT));
    let input_text = required(&mut errors, &[], map, "input", body)
        .and_then(|value| parse_input(&mut errors, value));
    let questions = required(&mut errors, &[], map, "questions", body)
        .and_then(|value| parse_decision_questions(&mut errors, value));
    optional_str(
        &mut errors,
        &["safety_identifier".into()],
        map.get("safety_identifier"),
        MAX_SAFETY_IDENTIFIER,
    );
    let joint = strict_bool(&mut errors, &["joint".into()], map.get("joint"));
    let cache_salt = optional_str(
        &mut errors,
        &["cache_salt".into()],
        map.get("cache_salt"),
        1024,
    );
    forbid_extra(&mut errors, &[], map, DECISIONS_FIELDS);
    if errors.len() > 0 {
        return Err(errors.into_api_error());
    }
    Ok(DecisionsRequest {
        model: model.expect("validated"),
        input_text: input_text.expect("validated"),
        questions: questions.expect("validated"),
        joint,
        cache_salt,
    })
}

/// `input: InputText | list[DecisionInputMessage]`, as the text the model
/// reads.
fn parse_input(errors: &mut Errors, value: &Value) -> Option<String> {
    let loc: [Loc; 1] = ["input".into()];
    let mut text_errors = Errors::default();
    if let Some(text) = strict_str(&mut text_errors, &at(&loc, "str"), value, INPUT_TEXT) {
        return Some(text);
    }
    let mut list_errors = Errors::default();
    let list_loc = at(&loc, "list[DecisionInputMessage]");
    let messages =
        list(&mut list_errors, &list_loc, value, 0, MAX_INPUT_MESSAGES).map(|messages| {
            messages
                .iter()
                .enumerate()
                .map(|(i, message)| {
                    parse_input_message(&mut list_errors, &at(&list_loc, i), message)
                })
                .collect::<Vec<_>>()
        });
    if list_errors.len() == 0 {
        let messages: Vec<String> = messages?.into_iter().map(Option::unwrap).collect();
        return Some(messages.join("\n\n"));
    }
    errors.0.extend(text_errors.0);
    errors.0.extend(list_errors.0);
    None
}

fn parse_input_message(errors: &mut Errors, loc: &[Loc], value: &Value) -> Option<String> {
    let Value::Object(map) = value else {
        errors.push(
            "model_type",
            loc,
            "Input should be a valid dictionary or instance of DecisionInputMessage",
            value,
        );
        return None;
    };
    if let Some(Value::Array(parts)) = map.get("content")
        && parts
            .iter()
            .any(|part| part.get("type").and_then(Value::as_str) == Some("input_image"))
    {
        errors.push(
            "value_error",
            loc,
            "Value error, Decisions supports text input only; images are not supported",
            value,
        );
        return None;
    }
    let before = errors.len();
    if let Some(role) = required(errors, loc, map, "role", value) {
        literal(errors, &at(loc, "role"), role, "user");
    }
    let content = required(errors, loc, map, "content", value)
        .and_then(|content| parse_message_content(errors, &at(loc, "content"), content));
    if let Some(kind) = map.get("type") {
        literal(errors, &at(loc, "type"), kind, "message");
    }
    forbid_extra(errors, loc, map, &["role", "content", "type"]);
    if errors.len() > before {
        return None;
    }
    content
}

/// `content: InputText | list[DecisionInputText]`.
fn parse_message_content(errors: &mut Errors, loc: &[Loc], value: &Value) -> Option<String> {
    let mut text_errors = Errors::default();
    if let Some(text) = strict_str(&mut text_errors, &at(loc, "str"), value, INPUT_TEXT) {
        return Some(text);
    }
    let mut list_errors = Errors::default();
    let list_loc = at(loc, "list[DecisionInputText]");
    let parts = list(&mut list_errors, &list_loc, value, 0, MAX_INPUT_PARTS).map(|parts| {
        parts
            .iter()
            .enumerate()
            .map(|(i, part)| parse_input_text(&mut list_errors, &at(&list_loc, i), part))
            .collect::<Vec<_>>()
    });
    if list_errors.len() == 0 {
        let parts: Vec<String> = parts?.into_iter().map(Option::unwrap).collect();
        return Some(parts.join("\n"));
    }
    errors.0.extend(text_errors.0);
    errors.0.extend(list_errors.0);
    None
}

fn parse_input_text(errors: &mut Errors, loc: &[Loc], value: &Value) -> Option<String> {
    let Value::Object(map) = value else {
        errors.push(
            "model_type",
            loc,
            "Input should be a valid dictionary or instance of DecisionInputText",
            value,
        );
        return None;
    };
    let before = errors.len();
    if let Some(kind) = required(errors, loc, map, "type", value) {
        literal(errors, &at(loc, "type"), kind, "input_text");
    }
    let text = required(errors, loc, map, "text", value)
        .and_then(|text| strict_str(errors, &at(loc, "text"), text, INPUT_TEXT));
    forbid_extra(errors, loc, map, &["type", "text"]);
    if errors.len() > before {
        return None;
    }
    text
}

fn parse_decision_questions(errors: &mut Errors, value: &Value) -> Option<Vec<DecisionQuestion>> {
    let loc: [Loc; 1] = ["questions".into()];
    let values = list(errors, &loc, value, 1, MAX_QUESTIONS)?;
    let before = errors.len();
    let questions: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, question)| parse_decision_question(errors, &at(&loc, i), question))
        .collect();
    if errors.len() > before {
        return None;
    }
    questions.into_iter().collect()
}

const QUESTION_TAGS: [&str; 3] = ["predicate", "choice", "score"];

fn parse_decision_question(
    errors: &mut Errors,
    loc: &[Loc],
    value: &Value,
) -> Option<DecisionQuestion> {
    let Value::Object(map) = value else {
        errors.push(
            "model_attributes_type",
            loc,
            "Input should be a valid dictionary or object to extract fields from",
            value,
        );
        return None;
    };
    let Some(tag) = map.get("type") else {
        errors.push(
            "union_tag_not_found",
            loc,
            "Unable to extract tag using discriminator 'type'",
            value,
        );
        return None;
    };
    let Some(&tag) = QUESTION_TAGS.iter().find(|&&expected| tag.as_str() == Some(expected)) else {
        let expected: Vec<_> = QUESTION_TAGS.iter().map(|tag| py::repr_str(tag)).collect();
        errors.push(
            "union_tag_invalid",
            loc,
            format!(
                "Input tag {} found using 'type' does not match any of the expected tags: {}",
                py::repr_str(&py::str(tag)),
                expected.join(", ")
            ),
            value,
        );
        return None;
    };
    let loc = at(loc, tag);
    let before = errors.len();
    let instructions =
        required(errors, &loc, map, "instructions", value).and_then(|instructions| {
            strict_str(errors, &at(&loc, "instructions"), instructions, SHORT_TEXT)
        });
    let name = optional_str(errors, &at(&loc, "name"), map.get("name"), SHORT_TEXT);
    let mut fields = vec!["instructions", "name", "type"];
    let kind = match tag {
        "predicate" => Some(DecisionQuestionKind::Predicate),
        "choice" => {
            fields.push("choices");
            required(errors, &loc, map, "choices", value)
                .and_then(|choices| parse_choices(errors, &at(&loc, "choices"), choices))
                .map(DecisionQuestionKind::Choice)
        }
        _ => {
            fields.push("levels");
            required(errors, &loc, map, "levels", value)
                .and_then(|levels| parse_levels(errors, &at(&loc, "levels"), levels))
                .map(DecisionQuestionKind::Score)
        }
    };
    forbid_extra(errors, &loc, map, &fields);
    if errors.len() > before {
        return None;
    }
    if map.get("name").is_some_and(Value::is_null) {
        errors.push(
            "value_error",
            &loc,
            "Value error, name must be a string when provided",
            value,
        );
        return None;
    }
    let kind = kind?;
    if let DecisionQuestionKind::Choice(choices) = &kind {
        let distinct = choices
            .iter()
            .enumerate()
            .all(|(i, choice)| choices[..i].iter().all(|other| other.value != choice.value));
        if !distinct {
            errors.push(
                "value_error",
                &loc,
                "Value error, choice values must be distinct",
                value,
            );
            return None;
        }
    }
    Some(DecisionQuestion {
        instructions: instructions?,
        name,
        kind,
    })
}

fn parse_choices(errors: &mut Errors, loc: &[Loc], value: &Value) -> Option<Vec<ChoiceOption>> {
    let values = list(errors, loc, value, 2, MAX_CHOICES)?;
    let before = errors.len();
    let choices: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, choice)| {
            let loc = at(loc, i);
            let Value::Object(map) = choice else {
                errors.push(
                    "model_type",
                    &loc,
                    "Input should be a valid dictionary or instance of ChoiceOption",
                    choice,
                );
                return None;
            };
            let value =
                required(errors, &loc, map, "value", choice).and_then(|value| match value {
                    Value::String(s) => Some(ChoiceValue::Str(s.clone())),
                    Value::Bool(b) => Some(ChoiceValue::Bool(*b)),
                    value => {
                        let value_loc = at(&loc, "value");
                        errors.push(
                            "string_type",
                            &at(&value_loc, "str"),
                            "Input should be a valid string",
                            value,
                        );
                        errors.push(
                            "bool_type",
                            &at(&value_loc, "bool"),
                            "Input should be a valid boolean",
                            value,
                        );
                        None
                    }
                });
            let description =
                optional_text(errors, &at(&loc, "description"), map.get("description"));
            forbid_extra(errors, &loc, map, &["value", "description"]);
            Some(ChoiceOption {
                value: value?,
                description: description?,
            })
        })
        .collect();
    if errors.len() > before {
        return None;
    }
    choices.into_iter().collect()
}

fn parse_levels(errors: &mut Errors, loc: &[Loc], value: &Value) -> Option<Vec<ScoreLevel>> {
    let values = list(errors, loc, value, 2, MAX_LEVELS)?;
    let before = errors.len();
    let levels: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, level)| {
            let loc = at(loc, i);
            let Value::Object(map) = level else {
                errors.push(
                    "model_type",
                    &loc,
                    "Input should be a valid dictionary or instance of ScoreLevel",
                    level,
                );
                return None;
            };
            let label = required(errors, &loc, map, "label", level)
                .and_then(|label| strict_str(errors, &at(&loc, "label"), label, SHORT_TEXT));
            let description =
                optional_text(errors, &at(&loc, "description"), map.get("description"));
            forbid_extra(errors, &loc, map, &["label", "description"]);
            Some(ScoreLevel {
                label: label?,
                description: description?,
            })
        })
        .collect();
    if errors.len() > before {
        return None;
    }
    levels.into_iter().collect()
}

/// A `ShortText` field that defaults to an empty string.
fn optional_text(errors: &mut Errors, loc: &[Loc], value: Option<&Value>) -> Option<String> {
    match value {
        None => Some(String::new()),
        Some(value) => strict_str(errors, loc, value, SHORT_TEXT),
    }
}

/// One question of a `/v1/systemone` request, before its type parses it.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct QuestionSpec {
    pub(super) type_name: String,
    pub(super) instructions: Value,
    pub(super) criteria: Value,
    /// The names of fields the spec does not know, which reject the request.
    pub(super) extra: Vec<String>,
}

/// A validated `/v1/systemone` request.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SystemOneRequest {
    pub(super) model: Option<String>,
    pub(super) state: Value,
    /// Question id to question, in request order.
    pub(super) questions: Vec<(String, QuestionSpec)>,
    pub(super) instructions: Option<String>,
    pub(super) images: Option<Vec<String>>,
    pub(super) audio: Option<Vec<String>>,
    pub(super) videos: Option<Vec<String>>,
    pub(super) chat_template_kwargs: Option<Map<String, Value>>,
    pub(super) joint: bool,
    pub(super) cache_salt: Option<String>,
    pub(super) priority: i64,
    pub(super) request_id: Option<String>,
}

const SYSTEMONE_FIELDS: &[&str] = &[
    "model",
    "state",
    "questions",
    "instructions",
    "images",
    "audio",
    "videos",
    "chat_template_kwargs",
    "joint",
    "cache_salt",
    "priority",
    "request_id",
];

/// A lax `str` field: only strings, without a length bound.
fn lax_str(errors: &mut Errors, loc: &[Loc], value: Option<&Value>) -> Option<String> {
    optional_str(errors, loc, value, usize::MAX)
}

fn str_list(errors: &mut Errors, key: &str, value: Option<&Value>) -> Option<Vec<String>> {
    let value = value.filter(|value| !value.is_null())?;
    let loc: [Loc; 1] = [key.into()];
    let values = list(errors, &loc, value, 0, usize::MAX)?;
    let before = errors.len();
    let strings: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, value)| strict_str(errors, &at(&loc, i), value, usize::MAX))
        .collect();
    if errors.len() > before {
        return None;
    }
    strings.into_iter().collect()
}

/// A lax `int` in the `int64` range: integers, booleans, integral floats and
/// integer strings.
fn lax_i64(errors: &mut Errors, loc: &[Loc], value: &Value) -> Option<i64> {
    match value {
        Value::Bool(b) => Some(i64::from(*b)),
        Value::Number(number) => {
            if let Some(n) = number.as_i64() {
                Some(n)
            } else if number.is_u64() {
                errors.push(
                    "less_than_equal",
                    loc,
                    format!("Input should be less than or equal to {}", i64::MAX),
                    value,
                );
                None
            } else {
                let f = number.as_f64().unwrap_or(f64::NAN);
                if f.fract() != 0.0 || !f.is_finite() {
                    errors.push(
                        "int_from_float",
                        loc,
                        "Input should be a valid integer, got a number with a fractional part",
                        value,
                    );
                    return None;
                }
                if f < i64::MIN as f64 || f >= i64::MAX as f64 {
                    errors.push(
                        if f < 0.0 {
                            "greater_than_equal"
                        } else {
                            "less_than_equal"
                        },
                        loc,
                        if f < 0.0 {
                            format!("Input should be greater than or equal to {}", i64::MIN)
                        } else {
                            format!("Input should be less than or equal to {}", i64::MAX)
                        },
                        value,
                    );
                    return None;
                }
                Some(f as i64)
            }
        }
        Value::String(s) => match s.trim().parse::<i64>() {
            Ok(n) => Some(n),
            Err(_) => {
                errors.push(
                    "int_parsing",
                    loc,
                    "Input should be a valid integer, unable to parse string as an integer",
                    value,
                );
                None
            }
        },
        value => {
            errors.push("int_type", loc, "Input should be a valid integer", value);
            None
        }
    }
}

pub(super) fn parse_systemone_request(body: &Value) -> Result<SystemOneRequest, ApiError> {
    check_cache_salt(body)?;
    let map = body_object(body)?;
    let mut errors = Errors::default();
    let model = lax_str(&mut errors, &["model".into()], map.get("model"));
    let state = required(&mut errors, &[], map, "state", body).cloned();
    let questions = required(&mut errors, &[], map, "questions", body)
        .and_then(|questions| parse_question_specs(&mut errors, questions));
    let instructions = lax_str(
        &mut errors,
        &["instructions".into()],
        map.get("instructions"),
    );
    let images = str_list(&mut errors, "images", map.get("images"));
    let audio = str_list(&mut errors, "audio", map.get("audio"));
    let videos = str_list(&mut errors, "videos", map.get("videos"));
    let chat_template_kwargs = match map.get("chat_template_kwargs") {
        None | Some(Value::Null) => None,
        Some(Value::Object(kwargs)) => Some(kwargs.clone()),
        Some(value) => {
            errors.push(
                "dict_type",
                &["chat_template_kwargs".into()],
                "Input should be a valid dictionary",
                value,
            );
            None
        }
    };
    let joint = strict_bool(&mut errors, &["joint".into()], map.get("joint"));
    let cache_salt = lax_str(&mut errors, &["cache_salt".into()], map.get("cache_salt"));
    let priority = match map.get("priority") {
        None => Some(0),
        Some(value) => lax_i64(&mut errors, &["priority".into()], value),
    };
    let request_id = match map.get("request_id") {
        None => None,
        Some(value) => strict_str(&mut errors, &["request_id".into()], value, usize::MAX),
    };
    forbid_extra(&mut errors, &[], map, SYSTEMONE_FIELDS);
    if errors.len() > 0 {
        return Err(errors.into_api_error());
    }
    Ok(SystemOneRequest {
        model,
        state: state.expect("validated"),
        questions: questions.expect("validated"),
        instructions,
        images,
        audio,
        videos,
        chat_template_kwargs,
        joint,
        cache_salt,
        priority: priority.expect("validated"),
        request_id,
    })
}

fn parse_question_specs(errors: &mut Errors, value: &Value) -> Option<Vec<(String, QuestionSpec)>> {
    let loc: [Loc; 1] = ["questions".into()];
    let Value::Object(map) = value else {
        errors.push(
            "dict_type",
            &loc,
            "Input should be a valid dictionary",
            value,
        );
        return None;
    };
    let before = errors.len();
    let specs: Vec<_> = map
        .iter()
        .map(|(qid, spec)| {
            let loc = at(&loc, qid.as_str());
            let Value::Object(fields) = spec else {
                errors.push(
                    "model_type",
                    &loc,
                    "Input should be a valid dictionary or instance of QuestionSpec",
                    spec,
                );
                return None;
            };
            let type_name = required(errors, &loc, fields, "type", spec)
                .and_then(|kind| strict_str(errors, &at(&loc, "type"), kind, usize::MAX));
            let extra = fields
                .keys()
                .filter(|key| !["type", "instructions", "criteria"].contains(&key.as_str()))
                .cloned()
                .collect();
            Some((
                qid.clone(),
                QuestionSpec {
                    type_name: type_name?,
                    instructions: fields
                        .get("instructions")
                        .cloned()
                        .unwrap_or_else(|| Value::String(String::new())),
                    criteria: fields.get("criteria").cloned().unwrap_or(Value::Null),
                    extra,
                },
            ))
        })
        .collect();
    if errors.len() > before {
        return None;
    }
    specs.into_iter().collect()
}
