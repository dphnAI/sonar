// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Question types: a request's criteria become options, the question is
//! written for the model with a label per option, and the answer is built from
//! the label probabilities.
//!
//! Mirrors `sonar/entrypoints/openai/decisions/question_types.py`.

use serde_json::{Map, Value, json};

use super::DecisionError;
use super::py;

/// A choice labels its options A to Z in order.
pub(super) const LETTERS: [&str; 26] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z",
];
const YES_NO: [&str; 2] = ["yes", "no"];
const DIGITS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

/// The registered question types, in registration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum QuestionType {
    /// Pick one option. The criteria map each option name to a description,
    /// or to null.
    Choice,
    /// Yes or no. The criteria may describe what true and false mean.
    Noul,
    /// Rate the state on an ordered scale. The criteria list the levels in
    /// order.
    Score,
}

impl QuestionType {
    pub(super) const ALL: [Self; 3] = [Self::Choice, Self::Noul, Self::Score];

    fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Noul => "noul",
            Self::Score => "score",
        }
    }

    /// The labels of a question's options, in option order.
    pub(super) fn label_set(self) -> &'static [&'static str] {
        match self {
            Self::Choice => &LETTERS,
            Self::Noul => &YES_NO,
            // 0-indexed like the answer, so a level's label is its score.
            Self::Score => &DIGITS,
        }
    }

    fn reply_instruction(self) -> &'static str {
        match self {
            Self::Choice => "Answer with the letter of one option only.",
            Self::Noul => "Answer with yes or no only.",
            Self::Score => "Answer with the number of one level only.",
        }
    }

    fn reply_line(self, label: &str, option: &DecisionOption) -> String {
        let description = option.description.as_deref().filter(|d| !d.is_empty());
        match (self, description) {
            // A side without a description still shows its label. Without the
            // label in the prompt, models answer "Yes" or "No" instead.
            (Self::Noul, Some(description)) => format!("{label}: {description}"),
            (Self::Noul, None) => label.to_string(),
            (_, Some(description)) => format!("{label}: {} - {description}", option.name),
            (_, None) => format!("{label}: {}", option.name),
        }
    }

    /// The options and the reply instruction, after the question's heading.
    pub(super) fn body(self, question: &Question) -> String {
        let mut lines: Vec<String> = question
            .labels
            .iter()
            .zip(&question.options)
            .map(|(label, option)| self.reply_line(label, option))
            .collect();
        lines.push(self.reply_instruction().to_string());
        lines.join("\n")
    }

    /// The question as the model reads it, after the state.
    pub(super) fn prompt(self, question: &Question) -> String {
        let body = self.body(question);
        if question.instructions.is_empty() {
            return body;
        }
        format!("Question: {}\n{body}", question.instructions)
    }

    fn parse_options(
        self,
        qid: &str,
        criteria: &Value,
    ) -> Result<Vec<DecisionOption>, DecisionError> {
        let qid = py::repr_str(qid);
        match self {
            Self::Choice => match criteria {
                Value::Object(map) if !map.is_empty() => Ok(map
                    .iter()
                    .map(|(name, description)| DecisionOption {
                        name: name.clone(),
                        description: (!description.is_null()).then(|| py::str(description)),
                    })
                    .collect()),
                _ => Err(DecisionError::new(format!(
                    "question {qid}: choice criteria must map option names to a description or null"
                ))),
            },
            Self::Noul => {
                let side = |key: &str| match criteria {
                    Value::Object(map) => map.get(key).filter(|v| !v.is_null()).map(py::str),
                    _ => None,
                };
                let valid = match criteria {
                    Value::Null => true,
                    Value::Object(map) => map.keys().all(|key| key == "true" || key == "false"),
                    _ => false,
                };
                if !valid {
                    return Err(DecisionError::new(format!(
                        "question {qid}: noul criteria must be an object with true and false"
                    )));
                }
                Ok(vec![
                    DecisionOption {
                        name: "yes".to_string(),
                        description: side("true"),
                    },
                    DecisionOption {
                        name: "no".to_string(),
                        description: side("false"),
                    },
                ])
            }
            Self::Score => match criteria {
                Value::Array(levels) if levels.len() >= 2 => Ok(levels
                    .iter()
                    .map(|level| DecisionOption {
                        name: py::str(level),
                        description: None,
                    })
                    .collect()),
                _ => Err(DecisionError::new(format!(
                    "question {qid}: score criteria must be an ordered list of levels"
                ))),
            },
        }
    }

    /// The answer for `question`. `probs[i]` is the probability of
    /// `question.labels[i]` among the labels, and the list sums to 1.
    /// `label_mass` is the labels' total probability over the vocabulary.
    pub(super) fn answer(self, question: &Question, probs: &[f64], label_mass: f64) -> Value {
        let top = argmax(probs);
        let mut answer = Map::new();
        answer.insert("type".to_string(), json!(self.name()));
        match self {
            Self::Choice => {
                answer.insert("choice".to_string(), json!(question.options[top].name));
                let probabilities: Map<String, Value> = question
                    .options
                    .iter()
                    .zip(probs)
                    .map(|(option, &p)| (option.name.clone(), json!(p)))
                    .collect();
                answer.insert("probabilities".to_string(), Value::Object(probabilities));
                // The chosen label's probability over the whole vocabulary, so a
                // read that mostly wanted a non-label token reports low
                // confidence.
                answer.insert("confidence".to_string(), json!(probs[top] * label_mass));
            }
            Self::Noul => {
                answer.insert("noul".to_string(), json!(probs[0]));
                answer.insert(
                    "probabilities".to_string(),
                    json!({"yes": probs[0], "no": probs[1]}),
                );
                answer.insert("confidence".to_string(), json!(probs[top] * label_mass));
            }
            Self::Score => {
                answer.insert("score".to_string(), json!(expected_level(probs)));
                let legend: Map<String, Value> = question
                    .options
                    .iter()
                    .enumerate()
                    .map(|(i, option)| (i.to_string(), json!(option.name)))
                    .collect();
                answer.insert("legend".to_string(), Value::Object(legend));
                let probabilities: Map<String, Value> =
                    probs.iter().enumerate().map(|(i, &p)| (i.to_string(), json!(p))).collect();
                answer.insert("probabilities".to_string(), Value::Object(probabilities));
                answer.insert("confidence".to_string(), json!(probs[top] * label_mass));
            }
        }
        Value::Object(answer)
    }
}

/// One option of a question.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DecisionOption {
    pub(super) name: String,
    pub(super) description: Option<String>,
}

/// One question as a read sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Question {
    pub(super) id: String,
    pub(super) kind: QuestionType,
    pub(super) instructions: String,
    pub(super) options: Vec<DecisionOption>,
    /// The label of each option, in option order.
    pub(super) labels: &'static [&'static str],
}

impl Question {
    /// A question labeled with the first labels of its type.
    pub(super) fn labeled(
        id: String,
        kind: QuestionType,
        instructions: String,
        options: Vec<DecisionOption>,
    ) -> Self {
        let labels = &kind.label_set()[..options.len()];
        Self {
            id,
            kind,
            instructions,
            options,
            labels,
        }
    }
}

/// Build one question of the structured decisions API from its request spec.
pub(super) fn build_question(
    qid: &str,
    type_name: &str,
    instructions: &Value,
    criteria: &Value,
    max_options: usize,
) -> Result<Question, DecisionError> {
    if qid.is_empty() {
        return Err(DecisionError::new("question ids must be non-empty"));
    }
    let kind = QuestionType::parse(type_name).ok_or_else(|| {
        let supported: Vec<_> =
            QuestionType::ALL.iter().map(|kind| py::repr_str(kind.name())).collect();
        DecisionError::new(format!(
            "unknown question type {}; supported: [{}]",
            py::repr_str(type_name),
            supported.join(", ")
        ))
    })?;
    let options = kind.parse_options(qid, criteria)?;
    let qid_repr = py::repr_str(qid);
    if options.is_empty() {
        return Err(DecisionError::new(format!(
            "question {qid_repr}: needs at least one option"
        )));
    }
    let mut names: Vec<&str> = options.iter().map(|option| option.name.as_str()).collect();
    names.sort_unstable();
    if names.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(DecisionError::new(format!(
            "question {qid_repr}: duplicate option names"
        )));
    }
    let limit = max_options.min(kind.label_set().len());
    if options.len() > limit {
        return Err(DecisionError::new(format!(
            "question {qid_repr}: at most {limit} options for this model"
        )));
    }
    let instructions = match instructions {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        value => py::str(value),
    };
    Ok(Question::labeled(
        qid.to_string(),
        kind,
        instructions,
        options,
    ))
}

/// The index of the first largest value.
pub(super) fn argmax(values: &[f64]) -> usize {
    let mut top = 0;
    for (i, &value) in values.iter().enumerate().skip(1) {
        if value > values[top] {
            top = i;
        }
    }
    top
}

/// The expected level of a score, 0-indexed.
pub(super) fn expected_level(probs: &[f64]) -> f64 {
    py::sum(probs.iter().enumerate().map(|(i, &p)| i as f64 * p))
}

/// Softmax of the label logprobs.
pub(super) fn label_softmax(logprobs: &[f64]) -> Vec<f64> {
    let top = logprobs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logprobs.iter().map(|&lp| (lp - top).exp()).collect();
    let total = py::sum(weights.iter().copied());
    weights.into_iter().map(|w| w / total).collect()
}
