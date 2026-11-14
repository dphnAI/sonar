// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Read strategies: how a model is asked for label probabilities.
//!
//! Mirrors `sonar/entrypoints/openai/decisions/strategies.py` for the
//! strategies that generate: `NextTokenStrategy`, `LiquidStrategy` and
//! `DiffusionGemmaCanvasStrategy`. The strategies that score options with a
//! pooling head live in `pooling.rs`.

use std::collections::{BTreeSet, HashMap};

use futures::future::join_all;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use sonar_chat::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, GenerationPromptMode,
    ResolvedToolContext,
};
use sonar_engine_core_client::protocol::lora::LoraRequest;
use sonar_llm::{CollectedGenerateOutput, FinishReason, GenerateOutputStreamExt as _};
use sonar_text::output::TextDecodeOptions;
use sonar_text::{Prompt, SamplingParams, TextRequest};
use tracing::warn;

use super::pooling::PoolingReader;
use super::questions::{LETTERS, Question, QuestionType, label_softmax};
use super::{DecisionError, py};
use crate::config::DecisionsConfig;
use crate::error::{ApiError, chat_submit_error, engine_error, server_error, text_submit_error};
use crate::state::AppState;

/// The most questions one request may ask of a supported model.
pub(super) const MAX_QUESTIONS: usize = 64;
/// The most options one question may have: a choice labels them A to Z.
pub(super) const MAX_OPTIONS: usize = LETTERS.len();
/// Fewer shared tokens are not worth an extra round trip before the reads.
const MIN_WARM_PREFIX_TOKENS: usize = 128;
/// Python's `MAX_LOGPROB_TOKEN_IDS`.
const MAX_LOGPROB_TOKEN_IDS: usize = 128;

pub(super) const FRONTEND_FALLBACK: &str =
    "Set SONAR_USE_RUST_FRONTEND=0 to use the Python frontend.";

/// How the Python frontend reads a served architecture that generates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PythonStrategy {
    family: Family,
    /// Whether the model class takes images (`SupportsMultiModal`).
    multimodal: bool,
    /// Whether the model class keeps recurrent state (`IsHybrid`).
    hybrid: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    NextToken,
    Liquid,
    DiffusionGemma,
}

/// Python's `READ_STRATEGIES`, for the architectures that generate.
fn python_strategy(architecture: &str) -> Option<PythonStrategy> {
    use Family::*;
    let generate = |family, multimodal, hybrid| PythonStrategy {
        family,
        multimodal,
        hybrid,
    };
    Some(match architecture {
        "Gemma4ForCausalLM" | "Qwen3ForCausalLM" => generate(NextToken, false, false),
        "Gemma4ForConditionalGeneration" | "Gemma4UnifiedForConditionalGeneration" => {
            generate(NextToken, true, false)
        }
        "Qwen3_5ForConditionalGeneration" | "Qwen3_5MoeForConditionalGeneration" => {
            generate(NextToken, true, true)
        }
        "Lfm2VlForConditionalGeneration" => generate(Liquid, true, true),
        "DiffusionGemmaForBlockDiffusion" => generate(DiffusionGemma, true, false),
        _ => return None,
    })
}

/// Whether the served model answers decisions, as resolved once per server.
pub(crate) enum Availability {
    Ready(Box<Strategy>),
    /// Neither frontend serves decisions for this model.
    Unsupported,
}

/// How the served model reads decisions.
pub(crate) enum Strategy {
    /// By generating labels. Serves `/v1/systemone` and `/v1/decisions`.
    Generate(Reader),
    /// By pooling, which Python serves on `/v1/systemone` only.
    Pooling(PoolingReader),
}

impl Availability {
    pub(super) async fn resolve(state: &AppState, config: DecisionsConfig) -> Self {
        let hf_config = state.chat.text().hf_config();
        let Some(architecture) = hf_config
            .get("architectures")
            .and_then(Value::as_array)
            .and_then(|architectures| architectures.first())
            .and_then(Value::as_str)
        else {
            warn!("decisions are unavailable: the model config names no architecture");
            return Self::Unsupported;
        };
        if let Some(reader) = PoolingReader::new(state, architecture, hf_config) {
            return match reader {
                Ok(reader) => Self::Ready(Box::new(Strategy::Pooling(reader))),
                Err(reason) => {
                    warn!(architecture, %reason, "decisions are unavailable");
                    Self::Unsupported
                }
            };
        }
        let Some(strategy) = python_strategy(architecture) else {
            warn!(
                architecture,
                "decisions are unavailable for this architecture"
            );
            return Self::Unsupported;
        };
        if !config.logprobs_mode.returns_logprobs() {
            warn!(
                architecture,
                logprobs_mode = %config.logprobs_mode,
                "decisions need --logprobs-mode raw_logprobs or processed_logprobs"
            );
            return Self::Unsupported;
        }
        match Reader::new(state, config, strategy).await {
            Ok(reader) => Self::Ready(Box::new(Strategy::Generate(reader))),
            Err(reason) => {
                warn!(architecture, %reason, "decisions are unavailable");
                Self::Unsupported
            }
        }
    }
}

impl Strategy {
    /// The most options a question may have, and whether each is scored by
    /// its label's token.
    pub(super) fn max_options(&self) -> (usize, bool) {
        match self {
            Self::Generate(_) => (MAX_OPTIONS, true),
            Self::Pooling(reader) => reader.max_options(),
        }
    }

    /// Whether `read_joint` answers several questions from one canvas.
    pub(super) fn joint_reads(&self) -> bool {
        match self {
            Self::Generate(reader) => reader.joint_reads(),
            Self::Pooling(_) => false,
        }
    }

    /// The request's state as the model reads it.
    pub(super) fn render_state(&self, state: &Value) -> String {
        match self {
            Self::Generate(reader) => reader.render_state(state),
            Self::Pooling(reader) => reader.render_state(state),
        }
    }

    pub(super) fn check_media(
        &self,
        images: bool,
        audio: bool,
        videos: bool,
    ) -> Result<(), ApiError> {
        match self {
            Self::Generate(reader) => reader.check_media(images, audio, videos),
            Self::Pooling(reader) => reader.check_media(images, audio, videos),
        }
    }

    /// One answer per question, in order; jointly only if `joint_reads`.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn read(
        &self,
        state: &AppState,
        ctx: &ReadContext,
        questions: &[Question],
        instructions: Option<&str>,
        state_text: &str,
        images: &[String],
        joint: bool,
    ) -> Result<Vec<QuestionRead>, ApiError> {
        match self {
            Self::Generate(reader) if joint => {
                reader.read_joint(state, ctx, questions, instructions, state_text, images).await
            }
            Self::Generate(reader) => {
                reader.read(state, ctx, questions, instructions, state_text, images).await
            }
            Self::Pooling(reader) => {
                reader.read(state, ctx, questions, instructions, state_text, images).await
            }
        }
    }
}

/// Whether a reader takes the request's images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImageInput {
    /// Like Python, the model takes no images: a 400.
    NotTaken,
    /// The Python frontend reads them, the Rust frontend cannot: a 501.
    PythonOnly,
    Taken,
}

/// The canvas of DiffusionGemma reads.
struct Canvas {
    thought: Vec<u32>,
    end: u32,
    pad: u32,
    served: usize,
    vocab_size: u32,
    max_model_len: usize,
    /// Present when every answer line tokenizes as a prefix and one label.
    joint: Option<JointLines>,
}

struct JointLines {
    newline: u32,
    /// `(number, question type)` to the answer line's ids before the label.
    line_prefix: HashMap<(usize, QuestionType), Vec<u32>>,
    /// A label's id at the end of an answer line.
    slot_label_ids: HashMap<&'static str, u32>,
}

const THOUGHT: &str = "<|channel>thought\n<channel|>";
const END_OF_TURN: &str = "<turn|>";
const CANVAS_STEP: usize = 16;
const JOINT_REPLY: &str =
    "Reply with one line per question, in order, formatted as \"number: answer\".";

/// Where a joint read found an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct JointSlot {
    pub(super) canvas: usize,
    pub(super) width: usize,
    pub(super) position: usize,
}

/// Each question's slot on a canvas, with the tokens of each of its labels.
type Layout = Vec<(usize, Vec<Vec<u32>>)>;

/// One question's read.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct QuestionRead {
    pub(super) probs: Vec<f64>,
    pub(super) label_mass: f64,
    pub(super) argmax_is_label: bool,
    pub(super) input_tokens: usize,
    pub(super) output_tokens: usize,
    pub(super) cached_tokens: usize,
    pub(super) cache_write_tokens: usize,
    /// Where a joint read found this answer. Usage sits on each canvas's
    /// first read.
    pub(super) slot: Option<JointSlot>,
}

/// The request-wide options of one decision's reads.
pub(super) struct ReadContext {
    pub(super) request_id: String,
    pub(super) chat_template_kwargs: Option<Map<String, Value>>,
    pub(super) lora_request: Option<LoraRequest>,
    pub(super) priority: i32,
    pub(super) cache_salt: Option<String>,
}

/// A prompt rendered for one read.
struct Rendered {
    request: TextRequest,
    prompt_ids: Vec<u32>,
}

/// One engine read and the label tokens it is scored on.
struct LabelRead {
    request: TextRequest,
    /// `(position, label ids)` to read, in order.
    slots: Vec<(usize, Vec<u32>)>,
}

/// Reads label probabilities from a generative model.
pub(crate) struct Reader {
    family: Family,
    /// The prompt's ids after its last added token, which every rendered read
    /// prompt ends with.
    tail: Vec<u32>,
    /// The reply's first token for each label of every question type.
    label_ids: HashMap<&'static str, u32>,
    images: ImageInput,
    /// Prefill the shared state once before the reads.
    warm_prefix: bool,
    server_chat_template: bool,
    canvas: Option<Canvas>,
}

impl Reader {
    async fn new(
        state: &AppState,
        config: DecisionsConfig,
        PythonStrategy {
            family,
            multimodal,
            hybrid,
        }: PythonStrategy,
    ) -> Result<Self, String> {
        let text = state.chat.text();
        if !text.has_tokenizer() {
            return Err("the server runs without a tokenizer".to_string());
        }
        let tokenizer = text.tokenizer();
        let probe = generation_prompt(state).await?;
        let tail = prompt_tail(tokenizer.as_ref(), &probe);
        let rust_images = state.chat.supports_multimodal();
        let python_images = match family {
            Family::DiffusionGemma => true,
            _ => multimodal && !config.language_model_only,
        };
        let images = match (python_images, family, rust_images) {
            (false, _, _) => ImageInput::NotTaken,
            // Liquid shrinks pictures before the processor; the Rust frontend
            // does not.
            (true, Family::Liquid, _) | (true, _, false) => ImageInput::PythonOnly,
            (true, _, true) => ImageInput::Taken,
        };
        let prefix_caching = state.engine_core_client().ready_responses().iter().all(|ready| {
            ready
                .cache_config_info
                .as_ref()
                .and_then(|info| info.get("enable_prefix_caching"))
                .is_some_and(|value| value == "True")
        });
        let (label_ids, canvas) = match family {
            Family::DiffusionGemma => {
                let canvas = Canvas::new(state, config)?;
                (
                    type_label_ids(tokenizer.as_ref(), &canvas.thought)?,
                    Some(canvas),
                )
            }
            _ => (type_label_ids(tokenizer.as_ref(), &probe)?, None),
        };
        let mut reader = Self {
            family,
            tail,
            label_ids,
            images,
            warm_prefix: hybrid && prefix_caching,
            server_chat_template: config.server_chat_template,
            canvas,
        };
        if let Some(canvas) = &mut reader.canvas {
            canvas.joint = JointLines::new(tokenizer.as_ref(), &reader.label_ids);
        }
        Ok(reader)
    }

    /// Whether `read_joint` answers several questions from one canvas.
    pub(super) fn joint_reads(&self) -> bool {
        self.canvas.as_ref().is_some_and(|canvas| canvas.joint.is_some())
    }

    /// The request's state as the model reads it.
    pub(super) fn render_state(&self, state: &Value) -> String {
        match (self.family, state) {
            (Family::Liquid, Value::Null) => String::new(),
            (Family::Liquid, Value::String(s)) => s.clone(),
            (Family::Liquid, state) => py::json_dumps(state, Some(2)),
            (_, state) => state_text(state),
        }
    }

    /// Check the request's media like Python's `_media`, which only names the
    /// modalities the model does not take; the Rust frontend also refuses
    /// images it cannot process.
    pub(super) fn check_media(
        &self,
        images: bool,
        audio: bool,
        videos: bool,
    ) -> Result<(), ApiError> {
        let takes_images = self.images != ImageInput::NotTaken;
        let unsupported: Vec<&str> = [
            ("audio", audio),
            ("image", images && !takes_images),
            ("video", videos),
        ]
        .into_iter()
        .filter_map(|(name, present)| present.then_some(name))
        .collect();
        if !unsupported.is_empty() {
            return Err(DecisionError::new(format!(
                "this model does not take {} inputs",
                unsupported.join(", ")
            ))
            .into());
        }
        if images && self.images == ImageInput::PythonOnly {
            return Err(ApiError::not_implemented(format!(
                "Image inputs to structured decisions on this model are not supported by the \
                 Rust frontend yet. {FRONTEND_FALLBACK}"
            )));
        }
        Ok(())
    }

    /// One read per question, in order.
    pub(super) async fn read(
        &self,
        state: &AppState,
        ctx: &ReadContext,
        questions: &[Question],
        instructions: Option<&str>,
        state_text: &str,
        images: &[String],
    ) -> Result<Vec<QuestionRead>, ApiError> {
        let chat_options = self.chat_options(ctx)?;
        let contents: Vec<String> =
            questions.iter().map(|question| self.content(state_text, question)).collect();
        let rendered =
            self.prompts(state, ctx, &chat_options, &contents, instructions, images).await?;

        let mut reads = Vec::with_capacity(questions.len());
        for (i, (question, rendered)) in questions.iter().zip(rendered).enumerate() {
            let groups = self.label_groups(state, question)?;
            let label_ids: Vec<u32> = groups.iter().flatten().copied().collect();
            let (prompt_ids, params) = match &self.canvas {
                Some(canvas) => (
                    canvas.read_input(&rendered.prompt_ids, CANVAS_STEP)?,
                    canvas.single_params(label_ids.clone(), &rendered.prompt_ids),
                ),
                None => (
                    rendered.prompt_ids.clone(),
                    next_token_params(label_ids.clone()),
                ),
            };
            reads.push((
                groups,
                LabelRead {
                    request: engine_request(
                        rendered.request,
                        format!("{}-{i}", ctx.request_id),
                        prompt_ids,
                        params,
                    ),
                    slots: vec![(0, label_ids)],
                },
            ));
        }

        if self.warm_prefix {
            let prompts: Vec<&[u32]> =
                reads.iter().map(|(_, read)| prompt_ids(&read.request)).collect();
            warm_shared_prefix(state, ctx, &prompts).await?;
        }
        let (groups, label_reads): (Vec<_>, Vec<_>) = reads.into_iter().unzip();
        let outputs = label_reads_of(state, label_reads).await?;
        Ok(groups
            .iter()
            .zip(outputs)
            .map(|(groups, (output, logprobs))| question_read(&output, groups, &logprobs, 0, true))
            .collect())
    }

    /// Like `read`, with questions sharing canvases. Only if `joint_reads`.
    pub(super) async fn read_joint(
        &self,
        state: &AppState,
        ctx: &ReadContext,
        questions: &[Question],
        instructions: Option<&str>,
        state_text: &str,
        images: &[String],
    ) -> Result<Vec<QuestionRead>, ApiError> {
        let canvas = self.canvas.as_ref().expect("joint reads need a canvas");
        let lines = canvas.joint.as_ref().expect("joint reads need answer lines");
        let chat_options = self.chat_options(ctx)?;
        let chunks = canvas.joint_chunks(lines, questions);
        let contents: Vec<String> = chunks
            .iter()
            .map(|(start, chunk)| joint_content(state_text, chunk, *start))
            .collect();
        let rendered =
            self.prompts(state, ctx, &chat_options, &contents, instructions, images).await?;

        let mut layouts = Vec::with_capacity(chunks.len());
        let mut widths = Vec::with_capacity(chunks.len());
        let mut label_reads = Vec::with_capacity(chunks.len());
        for (i, ((start, chunk), rendered)) in chunks.iter().zip(rendered).enumerate() {
            let (seed, layout) =
                canvas.joint_canvas(lines, &self.label_ids, chunk, &rendered.prompt_ids, *start);
            let slots: Vec<(usize, Vec<u32>)> = layout
                .iter()
                .map(|(position, groups)| (*position, groups.iter().flatten().copied().collect()))
                .collect();
            let label_ids: Vec<u32> = slots
                .iter()
                .flat_map(|(_, ids)| ids.iter().copied())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let max_tokens = layout.last().expect("a chunk has a question").0 + 1;
            let prompt_ids = canvas.read_input(&rendered.prompt_ids, seed.len())?;
            widths.push(seed.len());
            let params = read_params(seed, max_tokens, label_ids);
            label_reads.push(LabelRead {
                request: engine_request(
                    rendered.request,
                    format!("{}-{i}", ctx.request_id),
                    prompt_ids,
                    params,
                ),
                slots,
            });
            layouts.push(layout);
        }

        let outputs = label_reads_of(state, label_reads).await?;
        let mut reads = Vec::with_capacity(questions.len());
        for (index, (layout, (output, logprobs))) in layouts.iter().zip(outputs).enumerate() {
            let mut remaining = logprobs.into_iter();
            for (n, (position, groups)) in layout.iter().enumerate() {
                let count = groups.iter().map(Vec::len).sum();
                let logprobs: Vec<f64> = remaining.by_ref().take(count).collect();
                let mut read = question_read(&output, groups, &logprobs, *position, n == 0);
                read.slot = Some(JointSlot {
                    canvas: index,
                    width: widths[index],
                    position: *position,
                });
                reads.push(read);
            }
        }
        Ok(reads)
    }

    /// The chat options of every read: thinking off unless the request turns
    /// it on, which a read refuses, and the generation prompt.
    fn chat_options(&self, ctx: &ReadContext) -> Result<ChatOptions, ApiError> {
        let kwargs = ctx.chat_template_kwargs.clone().unwrap_or_default();
        if kwargs.get("enable_thinking").is_some_and(py::truthy) {
            return Err(DecisionError::new(
                "a read is the reply's first token, so thinking must be off",
            )
            .into());
        }
        // Python's `merge_kwargs` leaves out request values of None and "auto".
        let mut template_kwargs = HashMap::from([("enable_thinking".to_string(), json!(false))]);
        let mut chat_template = None;
        for (key, value) in kwargs {
            if value.is_null() || value.as_str() == Some("auto") {
                continue;
            }
            match key.as_str() {
                "add_generation_prompt" | "continue_final_message" => {}
                // Python passes the server's --chat-template over a request's.
                "chat_template" if self.server_chat_template => {}
                "chat_template" => match value {
                    Value::String(template) => chat_template = Some(template),
                    _ => {
                        return Err(ApiError::invalid_request(
                            "chat_template must be a string",
                            Some("chat_template"),
                        ));
                    }
                },
                _ => {
                    template_kwargs.insert(key, value);
                }
            }
        }
        Ok(ChatOptions {
            generation_prompt_mode: GenerationPromptMode::StartNewAssistant,
            chat_template,
            reasoning_effort: None,
            response_format: None,
            template_kwargs,
        })
    }

    /// The user turn of one read: the state, then the question.
    fn content(&self, state: &str, question: &Question) -> String {
        match self.family {
            Family::Liquid => liquid_content(state, question),
            _ => format!("{state}\n\n{}", question.kind.prompt(question)),
        }
    }

    /// The tokens that count for each label. A label scores its best one.
    fn label_groups(
        &self,
        state: &AppState,
        question: &Question,
    ) -> Result<Vec<Vec<u32>>, ApiError> {
        if self.family != Family::Liquid {
            return Ok(question.labels.iter().map(|label| vec![self.label_ids[label]]).collect());
        }
        liquid_label_groups(state.chat.text().tokenizer().as_ref(), question)
    }
}

/// The user turn of a Liquid read, in the wording of the `prompt.py` the
/// checkpoint ships.
pub(super) fn liquid_content(state: &str, question: &Question) -> String {
    let q = question;
    let ask = match q.kind {
        QuestionType::Choice => {
            let lines: Vec<String> = liquid_codes(q)
                .iter()
                .zip(&q.options)
                .map(|(code, option)| {
                    let text = match option.description.as_deref() {
                        Some(description) if !description.is_empty() => description.to_string(),
                        _ => option.name.replace('_', " "),
                    };
                    format!("{code} {text}")
                })
                .collect();
            format!(
                "{}\n\nOptions:\n{}\n\nReply with the option code only.",
                q.instructions,
                lines.join("\n")
            )
        }
        QuestionType::Noul => {
            let description = |i: usize| q.options[i].description.clone().unwrap_or_default();
            let (yes, no) = (description(0), description(1));
            let sides = if yes.is_empty() && no.is_empty() {
                String::new()
            } else {
                format!(
                    "\nYes: {}\nNo: {}",
                    py_none_or(&q.options[0].description),
                    py_none_or(&q.options[1].description)
                )
            };
            format!("{}{sides}\n\nReply with yes or no only.", q.instructions)
        }
        QuestionType::Score => {
            let legend: Vec<String> = q
                .labels
                .iter()
                .zip(&q.options)
                .map(|(label, option)| format!("{label} {}", option.name))
                .collect();
            format!(
                "{}\n\n{}\n\nReply with a single digit 0-{} only.",
                q.instructions,
                legend.join("\n"),
                q.options.len() - 1
            )
        }
    };
    // A null state leaves the images, or the question alone, as the prompt.
    if state.is_empty() {
        ask
    } else {
        format!("{state}\n\n\nQUESTION:\n{ask}")
    }
}

/// The tokens that count for each label of a Liquid read.
fn liquid_label_groups(
    tokenizer: &dyn sonar_text::tokenizer::Tokenizer,
    question: &Question,
) -> Result<Vec<Vec<u32>>, ApiError> {
    let single_tokens = |texts: &[String]| -> Result<Vec<u32>, ApiError> {
        let mut ids = Vec::new();
        for text in texts {
            let encoded = tokenizer
                .encode(text, false)
                .map_err(|error| server_error!("failed to tokenize a label: {error}"))?;
            if let [id] = encoded[..]
                && !ids.contains(&id)
            {
                ids.push(id);
            }
        }
        Ok(ids)
    };
    let forms: Vec<Vec<String>> = match question.kind {
        QuestionType::Noul => [["yes", "Yes", "YES"], ["no", "No", "NO"]]
            .iter()
            .map(|forms| forms.iter().map(|form| form.to_string()).collect())
            .collect(),
        QuestionType::Choice => liquid_codes(question)
            .into_iter()
            .map(|code| vec![code.clone(), format!(" {code}")])
            .collect(),
        QuestionType::Score => {
            question.labels.iter().map(|label| vec![label.to_string()]).collect()
        }
    };
    let mut groups = Vec::with_capacity(forms.len());
    let mut seen = BTreeSet::new();
    for forms in forms {
        let label = &forms[0];
        let ids = single_tokens(&forms)?;
        if ids.is_empty()
            || single_tokens(std::slice::from_ref(label))? != ids[..1]
            || ids.iter().any(|id| seen.contains(id))
        {
            return Err(DecisionError::new(format!(
                "question {}: label {} is not one distinct token for this model",
                py::repr_str(&question.id),
                py::repr_str(label)
            ))
            .into());
        }
        seen.extend(ids.iter().copied());
        groups.push(ids);
    }
    Ok(groups)
}

impl Reader {
    /// Each content rendered as a chat prompt, with the prompt's ids.
    async fn prompts(
        &self,
        state: &AppState,
        ctx: &ReadContext,
        chat_options: &ChatOptions,
        contents: &[String],
        instructions: Option<&str>,
        images: &[String],
    ) -> Result<Vec<Rendered>, ApiError> {
        let mut rendered = Vec::with_capacity(contents.len());
        for (i, text) in contents.iter().enumerate() {
            let content = if images.is_empty() {
                ChatContent::Text(text.clone())
            } else {
                let mut parts: Vec<ChatContentPart> = images
                    .iter()
                    .map(|url| ChatContentPart::ImageUrl {
                        image_url: url.clone(),
                        detail: None,
                        uuid: None,
                    })
                    .collect();
                parts.push(ChatContentPart::text(text.clone()));
                ChatContent::Parts(parts)
            };
            let mut messages = Vec::with_capacity(2);
            if let Some(instructions) = instructions.filter(|instructions| !instructions.is_empty())
            {
                messages.push(ChatMessage::System {
                    content: ChatContent::Text(instructions.to_string()),
                });
            }
            messages.push(ChatMessage::User { content });
            let request = render(
                state,
                ctx,
                chat_options.clone(),
                format!("{}-{i}", ctx.request_id),
                messages,
            )
            .await?;
            let prompt_ids = prompt_ids(&request).to_vec();
            if !prompt_ids.ends_with(&self.tail) {
                return Err(DecisionError::new(
                    "these chat options end the prompt differently, so the labels' tokens are unknown",
                )
                .into());
            }
            rendered.push(Rendered {
                request,
                prompt_ids,
            });
        }
        Ok(rendered)
    }
}

/// Python's `str(None)` for a missing noul description.
fn py_none_or(description: &Option<String>) -> String {
    description.clone().unwrap_or_else(|| "None".to_string())
}

/// A Liquid choice's codes are A to Z, or the option names when those are
/// already single letters.
fn liquid_codes(question: &Question) -> Vec<String> {
    let names: Vec<&str> = question.options.iter().map(|option| option.name.trim()).collect();
    if names.iter().all(|name| {
        let mut chars = name.chars();
        chars.next().is_some_and(char::is_alphabetic) && chars.next().is_none()
    }) {
        return names.into_iter().map(str::to_string).collect();
    }
    question.labels.iter().map(|label| label.to_string()).collect()
}

/// The request's state as the base strategy writes it.
pub(super) fn state_text(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        state => py::json_dumps(state, None),
    }
}

/// A chat prompt's ids up to the generation prompt, with thinking off.
async fn generation_prompt(state: &AppState) -> Result<Vec<u32>, String> {
    let options = ChatOptions {
        template_kwargs: HashMap::from([("enable_thinking".to_string(), json!(false))]),
        ..ChatOptions::default()
    };
    let ctx = ReadContext {
        request_id: "decisions-probe".to_string(),
        chat_template_kwargs: None,
        lora_request: None,
        priority: 0,
        cache_salt: None,
    };
    let messages = vec![ChatMessage::User {
        content: ChatContent::Text("x".to_string()),
    }];
    let request = render(state, &ctx, options, ctx.request_id.clone(), messages)
        .await
        .map_err(|error| error.to_error_response().error.message)?;
    Ok(prompt_ids(&request).to_vec())
}

/// Render and tokenize one read prompt.
async fn render(
    state: &AppState,
    ctx: &ReadContext,
    chat_options: ChatOptions,
    request_id: String,
    messages: Vec<ChatMessage>,
) -> Result<TextRequest, ApiError> {
    let request = ChatRequest {
        request_id,
        messages,
        sampling_params: SamplingParams::default(),
        chat_options,
        tool_context: ResolvedToolContext::default(),
        decode_options: TextDecodeOptions {
            detokenize: false,
            ..TextDecodeOptions::default()
        },
        intermediate: false,
        prompt_truncation: None,
        priority: ctx.priority,
        documents: None,
        cache_salt: ctx.cache_salt.clone(),
        add_special_tokens: false,
        data_parallel_rank: None,
        session_id: None,
        lora_request: ctx.lora_request.clone(),
        return_prompt_text: false,
    };
    let mut request = state
        .chat
        .request_processor()
        .prepare_for_tokenization(request)
        .await
        .map_err(|error| chat_submit_error("failed to render a decision prompt", error))?;
    state
        .chat
        .text()
        .request_processor()
        .tokenize_in_place(&mut request)
        .map_err(|error| text_submit_error("failed to tokenize a decision prompt", error))?;
    Ok(request)
}

fn prompt_ids(request: &TextRequest) -> &[u32] {
    match &request.prompt {
        Prompt::TokenIds(ids) => ids,
        Prompt::Text(_) => unreachable!("decision prompts are tokenized when rendered"),
    }
}

/// The prompt's ids after its last added token.
fn prompt_tail(tokenizer: &dyn sonar_text::tokenizer::Tokenizer, prompt_ids: &[u32]) -> Vec<u32> {
    let added: BTreeSet<u32> = tokenizer.added_vocab().iter().map(|(_, id)| *id).collect();
    let start = prompt_ids.iter().rposition(|id| added.contains(id)).map_or(0, |i| i + 1);
    prompt_ids[start..].to_vec()
}

/// The token each label adds after the prompt's tail as the first token of
/// the reply.
fn reply_label_ids(
    tokenizer: &dyn sonar_text::tokenizer::Tokenizer,
    prompt_ids: &[u32],
    labels: &[&'static str],
) -> Result<Vec<u32>, String> {
    let tail = prompt_tail(tokenizer, prompt_ids);
    let text = tokenizer.decode(&tail, false).map_err(|error| error.to_string())?;
    let mut ids = Vec::with_capacity(labels.len());
    for label in labels {
        let extended = tokenizer
            .encode(&format!("{text}{label}"), false)
            .map_err(|error| error.to_string())?;
        match extended.split_last() {
            Some((&last, rest)) if rest == tail && !ids.contains(&last) => ids.push(last),
            _ => {
                return Err(format!(
                    "label {} is not one distinct token after this model's chat prompt",
                    py::repr_str(label)
                ));
            }
        }
    }
    Ok(ids)
}

/// The reply's first token for each label of every question type.
fn type_label_ids(
    tokenizer: &dyn sonar_text::tokenizer::Tokenizer,
    prompt_ids: &[u32],
) -> Result<HashMap<&'static str, u32>, String> {
    let mut label_ids = HashMap::new();
    for kind in QuestionType::ALL {
        let ids = reply_label_ids(tokenizer, prompt_ids, kind.label_set())?;
        label_ids.extend(kind.label_set().iter().copied().zip(ids));
    }
    Ok(label_ids)
}

/// Neutral sampling, so the logprobs are the model's whatever the
/// generation config says.
fn neutral_params(
    temperature: f32,
    max_tokens: u32,
    logprob_token_ids: Vec<u32>,
) -> SamplingParams {
    SamplingParams {
        temperature: Some(temperature),
        top_p: Some(1.0),
        top_k: Some(0),
        min_p: Some(0.0),
        repetition_penalty: Some(1.0),
        presence_penalty: Some(0.0),
        frequency_penalty: Some(0.0),
        max_tokens: Some(max_tokens),
        logprob_token_ids: Some(logprob_token_ids),
        ..SamplingParams::default()
    }
}

/// A next-token read: the label tokens' logprobs at the reply's first token.
fn next_token_params(label_ids: Vec<u32>) -> SamplingParams {
    neutral_params(0.0, 1, label_ids)
}

/// A read-only canvas read: one denoising step over `canvas`.
fn read_params(canvas: Vec<u32>, max_tokens: usize, label_ids: Vec<u32>) -> SamplingParams {
    let width = canvas.len();
    let mut params = neutral_params(1.0, max_tokens as u32, label_ids);
    params.sonar_xargs = Some(HashMap::from([
        ("diffusion_seed_canvas".to_string(), json!(canvas)),
        ("diffusion_canvas_length".to_string(), json!(width)),
        ("diffusion_max_steps".to_string(), json!(1)),
        ("diffusion_read_only".to_string(), json!(true)),
    ]));
    params
}

/// The engine request of one read from its rendered prompt.
fn engine_request(
    mut request: TextRequest,
    request_id: String,
    prompt_ids: Vec<u32>,
    sampling_params: SamplingParams,
) -> TextRequest {
    request.request_id = request_id;
    request.prompt = Prompt::TokenIds(prompt_ids);
    request.sampling_params = sampling_params;
    request
}

/// The number of leading token ids that every prompt starts with.
pub(super) fn common_prefix_len(prompts: &[&[u32]]) -> usize {
    let Some(shortest) = prompts.iter().min_by_key(|prompt| prompt.len()) else {
        return 0;
    };
    shortest
        .iter()
        .enumerate()
        .position(|(i, id)| prompts.iter().any(|prompt| prompt[i] != *id))
        .unwrap_or(shortest.len())
}

/// Prefill the tokens every question starts with, before the reads.
///
/// The reads share the state only through the prefix cache. Reads that start
/// together cannot reuse each other's recurrent state, so on hybrid models
/// each of them would prefill the state again.
async fn warm_shared_prefix(
    state: &AppState,
    ctx: &ReadContext,
    prompts: &[&[u32]],
) -> Result<(), ApiError> {
    if prompts.len() < 2 {
        return Ok(());
    }
    let shared = common_prefix_len(prompts);
    if shared < MIN_WARM_PREFIX_TOKENS {
        return Ok(());
    }
    let request = TextRequest {
        request_id: format!("{}-prefix", ctx.request_id),
        prompt: Prompt::TokenIds(prompts[0][..shared].to_vec()),
        sampling_params: SamplingParams {
            temperature: Some(0.0),
            max_tokens: Some(1),
            ..SamplingParams::default()
        },
        decode_options: TextDecodeOptions {
            detokenize: false,
            ..TextDecodeOptions::default()
        },
        intermediate: false,
        priority: ctx.priority,
        cache_salt: ctx.cache_salt.clone(),
        lora_request: ctx.lora_request.clone(),
        ..TextRequest::for_test()
    };
    let stream = state
        .chat
        .text()
        .generate_raw(request)
        .await
        .map_err(|error| text_submit_error("failed to prefill the decision state", error))?;
    stream
        .collect_output()
        .await
        .map_err(|error| server_error!("failed to prefill the decision state: {error}"))?;
    Ok(())
}

/// Runs the reads at once and returns each one's output with the logprob of
/// each label token, slot after slot. Fails naming the item when a read has no
/// output or lacks a label's logprob.
async fn label_reads_of(
    state: &AppState,
    reads: Vec<LabelRead>,
) -> Result<Vec<(CollectedGenerateOutput, Vec<f64>)>, ApiError> {
    let text = state.chat.text();
    let (requests, slots): (Vec<_>, Vec<_>) =
        reads.into_iter().map(|read| (read.request, read.slots)).unzip();
    let results = join_all(requests.into_iter().map(|request| async move {
        let stream = text
            .generate_raw(request)
            .await
            .map_err(|error| text_submit_error("failed to submit a decision read", error))?;
        stream
            .collect_output()
            .await
            .map_err(|error| server_error!("failed to read a decision: {error}"))
    }))
    .await;

    let mut reads = Vec::with_capacity(results.len());
    for (i, (result, slots)) in results.into_iter().zip(slots).enumerate() {
        let output = result?;
        match &output.finish_reason {
            FinishReason::Error(Some(error)) => return Err(engine_error(Some(error))),
            FinishReason::Error(None) => {
                return Err(server_error!("Generation error for item {i}"));
            }
            _ => {}
        }
        let Some(positions) = output.logprobs.as_ref().filter(|logprobs| !logprobs.is_empty())
        else {
            return Err(server_error!(
                "No logprobs available for item {i}. This might indicate an issue with logprobs \
                 configuration."
            ));
        };
        let mut label_logprobs = Vec::new();
        for (position, label_ids) in slots {
            let entries = positions
                .positions
                .get(position)
                .map(|position| &position.entries[..])
                .unwrap_or_default();
            let logprob = |id: u32| {
                entries
                    .iter()
                    .find(|entry| entry.token_id == id)
                    .map(|entry| f64::from(entry.logprob))
            };
            let missing: Vec<u32> =
                label_ids.iter().copied().filter(|&id| logprob(id).is_none()).collect();
            if !missing.is_empty() {
                return Err(server_error!(
                    "Token IDs {missing:?} not found in logprobs for item {i}. This might \
                     indicate the tokens are outside the model's vocabulary."
                ));
            }
            label_logprobs.extend(label_ids.iter().map(|&id| logprob(id).expect("checked")));
        }
        reads.push((output, label_logprobs));
    }
    Ok(reads)
}

/// The answer at `position`. Only a read with `usage` counts tokens.
fn question_read(
    output: &CollectedGenerateOutput,
    groups: &[Vec<u32>],
    logprobs: &[f64],
    position: usize,
    usage: bool,
) -> QuestionRead {
    let mut remaining = logprobs.iter().copied();
    let best: Vec<f64> = groups
        .iter()
        .map(|group| remaining.by_ref().take(group.len()).fold(f64::NEG_INFINITY, f64::max))
        .collect();
    let mut read = QuestionRead {
        probs: label_softmax(&best),
        label_mass: py::sum(logprobs.iter().map(|lp| lp.exp())),
        argmax_is_label: output
            .token_ids
            .get(position)
            .is_some_and(|token| groups.iter().any(|group| group.contains(token))),
        input_tokens: 0,
        output_tokens: 0,
        cached_tokens: 0,
        cache_write_tokens: 0,
        slot: None,
    };
    if usage {
        read.input_tokens = output.prompt_token_ids.len();
        read.output_tokens = output.token_ids.len();
        read.cached_tokens = output.usage.prompt_cache_usage.cached_token_count;
        read.cache_write_tokens = output.usage.prompt_cache_usage.cache_creation_token_count;
    }
    read
}

/// The user turn of a joint read: the state, then the numbered questions.
pub(super) fn joint_content(state: &str, questions: &[Question], start: usize) -> String {
    let asks: Vec<String> = questions
        .iter()
        .enumerate()
        .map(|(k, question)| joint_question(start + k, question))
        .collect();
    format!("{state}\n\n{}\n\n{JOINT_REPLY}", asks.join("\n\n"))
}

/// A numbered question. The heading shows even without instructions, so each
/// answer line's number matches a question.
fn joint_question(number: usize, question: &Question) -> String {
    let mut head = format!("Question {number}:");
    if !question.instructions.is_empty() {
        head = format!("{head} {}", question.instructions);
    }
    format!("{head}\n{}", question.kind.body(question))
}

impl Canvas {
    fn new(state: &AppState, config: DecisionsConfig) -> Result<Self, String> {
        let text = state.chat.text();
        let tokenizer = text.tokenizer();
        let encode = |s: &str| tokenizer.encode(s, false).map_err(|error| error.to_string());
        let thought = encode(THOUGHT)?;
        let end = match encode(END_OF_TURN)?[..] {
            [end] => Some(end),
            _ => None,
        };
        let (Some(end), Some(pad)) = (end, text.pad_token_id()) else {
            return Err(format!(
                "Structured decisions need the model's {} and pad tokens",
                py::repr_str(END_OF_TURN)
            ));
        };
        let served = config
            .diffusion_canvas_length
            .map(|length| length as usize)
            .or_else(|| {
                text.hf_config()
                    .get("canvas_length")
                    .and_then(Value::as_u64)
                    .map(|length| length as usize)
            })
            .ok_or_else(|| "the model config sets no canvas_length".to_string())?;
        if CANVAS_STEP > served {
            return Err(format!(
                "Structured decisions need a canvas of {CANVAS_STEP}, and the served canvas is {served}"
            ));
        }
        Ok(Self {
            thought,
            end,
            pad,
            served,
            vocab_size: u32::try_from(state.chat.model_vocab_size())
                .map_err(|error| error.to_string())?,
            max_model_len: state.llm().info().max_model_len as usize,
            joint: None,
        })
    }

    /// The prompt of a canvas read: the thought block goes outside of the
    /// canvas, after the rendered prompt.
    fn read_input(&self, prompt_ids: &[u32], width: usize) -> Result<Vec<u32>, ApiError> {
        let extra = self.thought.len() + width;
        if prompt_ids.len() + extra > self.max_model_len {
            return Err(DecisionError::new(format!(
                "the prompt has {} tokens and a canvas read adds {extra}, more than max_model_len={}",
                prompt_ids.len(),
                self.max_model_len
            ))
            .into());
        }
        // Media placeholders come before the thought, so their ranges hold.
        Ok([prompt_ids, &self.thought].concat())
    }

    /// The params of a one-question read. The noise at the label slot is drawn
    /// from the prompt, so a repeated read gets the same canvas.
    fn single_params(&self, label_ids: Vec<u32>, prompt_ids: &[u32]) -> SamplingParams {
        let mut random = prompt_random(prompt_ids);
        let mut canvas = vec![random.randrange(self.vocab_size), self.end];
        canvas.resize(CANVAS_STEP, self.pad);
        read_params(canvas, 2, label_ids)
    }

    /// The fewest even runs that fit the canvas, each with its first number.
    fn joint_chunks<'q>(
        &self,
        lines: &JointLines,
        questions: &'q [Question],
    ) -> Vec<(usize, &'q [Question])> {
        let rows: Vec<usize> = questions
            .iter()
            .enumerate()
            .map(|(k, question)| lines.line_prefix[&(k + 1, question.kind)].len() + 2)
            .collect();
        for count in 1..questions.len() {
            let (size, extra) = (questions.len() / count, questions.len() % count);
            let bounds: Vec<usize> = (0..=count).map(|i| i * size + i.min(extra)).collect();
            // The lines and the end of turn must leave a pad row, or label mass
            // is lost.
            if bounds
                .windows(2)
                .all(|span| 1 + rows[span[0]..span[1]].iter().sum::<usize>() < self.served)
            {
                return bounds
                    .windows(2)
                    .map(|span| (span[0] + 1, &questions[span[0]..span[1]]))
                    .collect();
            }
        }
        (0..questions.len()).map(|k| (k + 1, &questions[k..=k])).collect()
    }

    /// The answer lines with noise in the slots, and each label's tokens.
    fn joint_canvas(
        &self,
        lines: &JointLines,
        label_ids: &HashMap<&'static str, u32>,
        questions: &[Question],
        prompt_ids: &[u32],
        start: usize,
    ) -> (Vec<u32>, Layout) {
        let mut random = prompt_random(prompt_ids);
        let mut canvas = Vec::new();
        let mut layout = Vec::with_capacity(questions.len());
        for (k, question) in questions.iter().enumerate() {
            canvas.extend_from_slice(&lines.line_prefix[&(start + k, question.kind)]);
            // A label also counts without its leading space, as in a
            // per-question read.
            let groups = question
                .labels
                .iter()
                .map(|label| {
                    let (slot, reply) = (lines.slot_label_ids[label], label_ids[label]);
                    if slot == reply {
                        vec![slot]
                    } else {
                        vec![slot, reply]
                    }
                })
                .collect();
            layout.push((canvas.len(), groups));
            canvas.push(random.randrange(self.vocab_size));
            canvas.push(lines.newline);
        }
        canvas.push(self.end);
        let width = (canvas.len() + 1).div_ceil(CANVAS_STEP) * CANVAS_STEP;
        canvas.resize(width.min(self.served).max(canvas.len()), self.pad);
        (canvas, layout)
    }
}

/// Python's `random.Random(sha256(json.dumps(prompt_ids)).digest())`.
fn prompt_random(prompt_ids: &[u32]) -> py::PyRandom {
    let ids: Vec<String> = prompt_ids.iter().map(u32::to_string).collect();
    py::PyRandom::from_bytes(&Sha256::digest(format!("[{}]", ids.join(", ")).as_bytes()))
}

impl JointLines {
    /// The answer lines, when every line "k: label" ends in one label token.
    fn new(
        tokenizer: &dyn sonar_text::tokenizer::Tokenizer,
        label_ids: &HashMap<&'static str, u32>,
    ) -> Option<Self> {
        let encode = |s: &str| tokenizer.encode(s, false).ok();
        let [newline] = encode("\n")?[..] else {
            warn!("Joint decision reads are off: a newline is not one token");
            return None;
        };
        let mut line_prefix = HashMap::new();
        let mut slot_label_ids: HashMap<&'static str, u32> = HashMap::new();
        for k in 1..=MAX_QUESTIONS {
            for kind in QuestionType::ALL {
                let labels = kind.label_set();
                let mut prefix = encode(&format!("{k}: {}", labels[0]))?;
                prefix.pop();
                for &label in labels {
                    let line = encode(&format!("{k}: {label}"))?;
                    let after = encode(&format!("{k}: {label}\n{k}: {label}"))?;
                    let last = *line.last()?;
                    let slot_id = *slot_label_ids.entry(label).or_insert(last);
                    if after != [&line[..], &[newline], &line[..]].concat()
                        || line[..line.len() - 1] != prefix[..]
                        || slot_id != last
                    {
                        warn!(
                            line = %format!("{k}: {label}"),
                            "Joint decision reads are off: an answer line does not tokenize as \
                             a prefix and one label token"
                        );
                        return None;
                    }
                }
                line_prefix.insert((k, kind), prefix);
            }
        }
        // A canvas asks for the ids of every label on it, however many
        // questions it holds, so the union of all labels' ids bounds a request.
        let ids: BTreeSet<u32> =
            slot_label_ids.values().chain(label_ids.values()).copied().collect();
        if ids.len() > MAX_LOGPROB_TOKEN_IDS {
            warn!(
                ids = ids.len(),
                max = MAX_LOGPROB_TOKEN_IDS,
                "Joint decision reads are off: the labels have more token ids than a request can \
                 ask logprobs for"
            );
            return None;
        }
        Some(Self {
            newline,
            line_prefix,
            slot_label_ids,
        })
    }
}
