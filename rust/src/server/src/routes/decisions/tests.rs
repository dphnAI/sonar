// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The Python contract tests of question types, request validation and
//! prompts, against the Rust port.

use serde_json::{Value, json};

use super::protocol::{parse_decisions_request, parse_systemone_request};
use super::questions::{Question, build_question, label_softmax};
use super::strategy::{QuestionRead, common_prefix_len, joint_content, liquid_content, state_text};
use super::{decision_answer, read_question};

fn choice(criteria: Value, max_options: usize) -> Question {
    build_question(
        "bucket",
        "choice",
        &json!("Which team?"),
        &criteria,
        max_options,
        true,
    )
    .unwrap()
}

fn default_choice() -> Question {
    choice(
        json!({"billing": "money", "outage": null, "other": null}),
        128,
    )
}

fn noul(instructions: &str, criteria: Value) -> Question {
    build_question("urgent", "noul", &json!(instructions), &criteria, 128, true).unwrap()
}

fn score(criteria: Value) -> Question {
    build_question("tone", "score", &json!("How angry?"), &criteria, 128, true).unwrap()
}

fn build_error(qid: &str, kind: &str, criteria: Value, max_options: usize) -> String {
    let error = build_question(qid, kind, &json!(""), &criteria, max_options, true).unwrap_err();
    crate::error::ApiError::from(error).to_error_response().error.message
}

#[test]
fn choice_labels_follow_option_order() {
    let q = default_choice();
    assert_eq!(q.labels, ["A", "B", "C"]);
    let names: Vec<_> = q.options.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(names, ["billing", "outage", "other"]);
    let wide = choice(
        Value::Object((0..26).map(|i| (i.to_string(), Value::Null)).collect()),
        128,
    );
    assert_eq!(wide.labels.last(), Some(&"Z"));
}

#[test]
fn prompts_match_python() {
    let q = default_choice();
    assert_eq!(
        q.kind.prompt(&q),
        "Question: Which team?\nA: billing - money\nB: outage\nC: other\nAnswer with the letter of one option only."
    );
    let q = noul("Reply within the hour?", Value::Null);
    assert_eq!(
        q.kind.prompt(&q),
        "Question: Reply within the hour?\nyes\nno\nAnswer with yes or no only."
    );
    let q = noul(
        "Reply within the hour?",
        json!({"true": "same day", "false": "later"}),
    );
    assert_eq!(
        q.kind.prompt(&q),
        "Question: Reply within the hour?\nyes: same day\nno: later\nAnswer with yes or no only."
    );
    let q = score(json!(["calm", "annoyed", "furious"]));
    assert_eq!(
        q.kind.prompt(&q),
        "Question: How angry?\n0: calm\n1: annoyed\n2: furious\nAnswer with the number of one level only."
    );
    // Instructions are kept as given, trailing newlines included, and a
    // question without instructions is its body alone.
    let q = noul("Reply within the hour?\n\n", Value::Null);
    assert_eq!(
        q.kind.prompt(&q),
        "Question: Reply within the hour?\n\n\nyes\nno\nAnswer with yes or no only."
    );
    let bare = Question {
        instructions: String::new(),
        ..q
    };
    assert_eq!(
        bare.kind.prompt(&bare),
        "yes\nno\nAnswer with yes or no only."
    );
}

#[test]
fn non_string_instructions_and_criteria_read_like_python_str() {
    let q = build_question(
        "q",
        "score",
        &json!(5),
        &json!([1, 2.0, true, null]),
        128,
        true,
    )
    .unwrap();
    assert_eq!(q.instructions, "5");
    let names: Vec<_> = q.options.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(names, ["1", "2.0", "True", "None"]);
    let q = build_question(
        "q",
        "choice",
        &Value::Null,
        &json!({"a": {"x": [1]}, "b": 3}),
        128,
        true,
    )
    .unwrap();
    assert_eq!(q.instructions, "");
    assert_eq!(q.options[0].description.as_deref(), Some("{'x': [1]}"));
    assert_eq!(q.options[1].description.as_deref(), Some("3"));
}

#[test]
fn answers_match_python() {
    let q = default_choice();
    assert_eq!(
        q.kind.answer(&q, &[0.2, 0.7, 0.1], 0.5),
        json!({
            "type": "choice",
            "choice": "outage",
            "probabilities": {"billing": 0.2, "outage": 0.7, "other": 0.1},
            "confidence": 0.35,
        })
    );
    let q = noul("x", Value::Null);
    assert_eq!(
        q.kind.answer(&q, &[0.8, 0.2], 0.5),
        json!({"type": "noul", "noul": 0.8, "probabilities": {"yes": 0.8, "no": 0.2}, "confidence": 0.4})
    );
    let q = score(json!(["calm", "annoyed", "furious"]));
    assert_eq!(
        q.kind.answer(&q, &[0.5, 0.25, 0.25], 0.4),
        json!({
            "type": "score",
            "score": 0.75,
            "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
            "probabilities": {"0": 0.5, "1": 0.25, "2": 0.25},
            "confidence": 0.2,
        })
    );
    // Key order is the option order, as clients see it.
    let answer = default_choice().kind.answer(&default_choice(), &[0.2, 0.7, 0.1], 0.5);
    assert_eq!(
        answer.to_string(),
        r#"{"type":"choice","choice":"outage","probabilities":{"billing":0.2,"outage":0.7,"other":0.1},"confidence":0.35}"#
    );
    let one = choice(json!({"only": null}), 128);
    assert_eq!(one.kind.answer(&one, &[1.0], 0.4)["confidence"], 0.4);
}

#[test]
fn build_question_rejects_like_python() {
    for (qid, kind, criteria, expected) in [
        (
            "",
            "choice",
            json!({"x": null, "y": null}),
            "question ids must be non-empty",
        ),
        (
            "q",
            "nope",
            json!({"x": null}),
            "unknown question type 'nope'; supported: ['choice', 'noul', 'score']",
        ),
        (
            "q",
            "choice",
            json!(["x", "y"]),
            "question 'q': choice criteria must map option names to a description or null",
        ),
        (
            "q",
            "noul",
            json!(["yes", "no"]),
            "question 'q': noul criteria must be an object with true and false",
        ),
        (
            "q",
            "noul",
            json!({"maybe": "same day"}),
            "question 'q': noul criteria must be an object with true and false",
        ),
        (
            "q",
            "score",
            json!({"0": "calm"}),
            "question 'q': score criteria must be an ordered list of levels",
        ),
        (
            "q",
            "score",
            json!(["calm"]),
            "question 'q': score criteria must be an ordered list of levels",
        ),
        (
            "q",
            "score",
            json!(["a", "a"]),
            "question 'q': duplicate option names",
        ),
        (
            "q",
            "score",
            json!([1, "1"]),
            "question 'q': duplicate option names",
        ),
    ] {
        assert_eq!(build_error(qid, kind, criteria, 128), expected);
    }
    assert_eq!(
        build_error("q", "choice", json!({"x": null, "y": null, "z": null}), 2),
        "question 'q': at most 2 options for this model"
    );
    let wide = Value::Object((0..27).map(|i| (i.to_string(), Value::Null)).collect());
    assert_eq!(
        build_error("q", "choice", wide, 128),
        "question 'q': at most 26 options for this model"
    );
    let levels = Value::Array((0..11).map(|i| json!(i.to_string())).collect());
    assert_eq!(
        build_error("q", "score", levels, 128),
        "question 'q': at most 10 options for this model"
    );
}

#[test]
fn label_softmax_normalizes() {
    let probs = label_softmax(&[-1.0, -1.0]);
    assert_eq!(probs, [0.5, 0.5]);
    let probs = label_softmax(&[0.0, f64::ln(3.0)]);
    assert!((probs[1] - 0.75).abs() < 1e-12);
}

fn decisions_body(overrides: Value) -> Value {
    let mut body = json!({
        "model": "test-model",
        "input": "The package is broken.",
        "questions": [{"type": "predicate", "instructions": "Is it damaged?"}],
    });
    for (key, value) in overrides.as_object().unwrap() {
        body[key] = value.clone();
    }
    body
}

fn decisions_error(overrides: Value) -> (String, String) {
    let response = parse_decisions_request(&decisions_body(overrides))
        .unwrap_err()
        .to_error_response();
    (response.error.error_type, response.error.message)
}

#[test]
fn decisions_request_rejects_like_pydantic() {
    for (overrides, expected) in [
        (
            json!({"input": [{"role": "assistant", "content": "x"}]}),
            "Input should be 'user'",
        ),
        (
            json!({"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}]}),
            "text input only",
        ),
        (
            json!({"input": {"text": "x"}}),
            "Input should be a valid list",
        ),
        (json!({"questions": []}), "at least 1 item"),
        (
            json!({"questions": [{"type": "predicate", "instructions": "x", "name": null}]}),
            "name must be a string",
        ),
        (
            json!({"questions": [{"type": "predicate", "instructions": 1}]}),
            "valid string",
        ),
        (
            json!({"questions": [{"type": "predicate", "instructions": "x", "criteria": {}}]}),
            "Extra inputs",
        ),
        (
            json!({"safety_identifier": "x".repeat(129)}),
            "at most 128 characters",
        ),
        (json!({"stream": true}), "Extra inputs"),
        (json!({"joint": "true"}), "valid boolean"),
        (
            json!({"questions": [{"type": "nope"}]}),
            "does not match any of the expected tags",
        ),
        (
            json!({"cache_salt": "a/b"}),
            "Parameter 'cache_salt' must be at most 128 characters",
        ),
    ] {
        let (error_type, message) = decisions_error(overrides.clone());
        assert_eq!(error_type, "Bad Request", "{overrides}");
        assert!(message.contains(expected), "{overrides}: {message}");
    }
    let (_, message) = decisions_error(json!({"input": [{"role": "assistant", "content": "x"}]}));
    assert_eq!(
        message,
        "2 validation errors:\n  \
         {'type': 'string_type', 'loc': ('body', 'input', 'str'), 'msg': 'Input should be a valid string', 'input': [{'role': 'assistant', 'content': 'x'}]}\n  \
         {'type': 'literal_error', 'loc': ('body', 'input', 'list[DecisionInputMessage]', 0, 'role'), 'msg': \"Input should be 'user'\", 'input': 'assistant'}"
    );
}

#[test]
fn decisions_input_reads_messages_in_order() {
    let request = parse_decisions_request(&decisions_body(json!({"input": [
        {"role": "user", "content": "first"},
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "second"},
            {"type": "input_text", "text": "third"},
        ]},
    ]})))
    .unwrap();
    assert_eq!(request.input_text, "first\n\nsecond\nthird");
    for input in [
        json!(""),
        json!([]),
        json!([{"role": "user", "content": []}]),
    ] {
        let request = parse_decisions_request(&decisions_body(json!({"input": input}))).unwrap();
        assert_eq!(request.input_text, "");
    }
}

#[test]
fn decisions_choice_and_score_limits() {
    let choices = |values: Value| json!({"questions": [{"type": "choice", "instructions": "x", "choices": values}]});
    for values in [
        json!([true, true]),
        json!(["a", "a"]),
        json!([1, "a"]),
        json!([null, "a"]),
    ] {
        let values: Vec<Value> =
            values.as_array().unwrap().iter().map(|v| json!({"value": v})).collect();
        assert!(parse_decisions_request(&decisions_body(choices(json!(values)))).is_err());
    }
    for (count, valid) in [(1, false), (2, true), (255, true), (256, false)] {
        let values: Vec<Value> = (0..count).map(|i| json!({"value": i.to_string()})).collect();
        assert_eq!(
            parse_decisions_request(&decisions_body(choices(json!(values)))).is_ok(),
            valid,
            "{count}"
        );
    }
    for (count, valid) in [(1, false), (2, true), (10, true), (11, false)] {
        let levels: Vec<Value> = (0..count).map(|i| json!({"label": i.to_string()})).collect();
        let body = decisions_body(
            json!({"questions": [{"type": "score", "instructions": "x", "levels": levels}]}),
        );
        assert_eq!(parse_decisions_request(&body).is_ok(), valid, "{count}");
    }
    let questions = vec![json!({"type": "predicate", "instructions": "x"}); 200];
    let body = decisions_body(json!({"questions": questions, "safety_identifier": null}));
    assert_eq!(parse_decisions_request(&body).unwrap().questions.len(), 200);
    let questions = vec![json!({"type": "predicate", "instructions": "x"}); 201];
    let (_, message) = decisions_error(json!({"questions": questions}));
    assert!(message.contains("at most 200 items"), "{message}");
}

fn read(probs: &[f64], label_mass: f64) -> QuestionRead {
    QuestionRead {
        probs: probs.to_vec(),
        label_mass,
        argmax_is_label: true,
        input_tokens: 0,
        output_tokens: 0,
        cached_tokens: 0,
        cache_write_tokens: 0,
        slot: None,
    }
}

#[test]
fn decisions_answers_match_python() {
    let request = parse_decisions_request(&decisions_body(json!({}))).unwrap();
    let question = &request.questions[0];
    let names: Vec<_> = read_question(0, question)
        .unwrap()
        .options
        .into_iter()
        .map(|o| o.name)
        .collect();
    assert_eq!(names, ["false", "true"]);
    assert_eq!(
        decision_answer(question, &read(&[0.1, 0.9], 0.5)),
        json!({"type": "predicate", "name": null, "probability": 0.9})
    );

    let request = parse_decisions_request(&decisions_body(json!({"questions": [{
        "type": "choice",
        "name": "decision",
        "instructions": "Select a value.",
        "choices": [{"value": true}, {"value": "true"}, {"value": false}],
    }]})))
    .unwrap();
    let question = &request.questions[0];
    let names: Vec<_> = read_question(0, question)
        .unwrap()
        .options
        .into_iter()
        .map(|o| o.name)
        .collect();
    assert_eq!(names, ["true", "\"true\"", "false"]);
    let answer = decision_answer(question, &read(&[0.7, 0.2, 0.1], 0.5));
    assert_eq!(answer["name"], "decision");
    assert_eq!(answer["choice"], true);
    assert_eq!(
        answer["probabilities"],
        json!([
            {"value": true, "probability": 0.7},
            {"value": "true", "probability": 0.2},
            {"value": false, "probability": 0.1},
        ])
    );
    assert!((answer["confidence"].as_f64().unwrap() - 0.35).abs() < 1e-12);

    let request = parse_decisions_request(&decisions_body(json!({"questions": [{
        "type": "score",
        "instructions": "Rate severity.",
        "levels": [{"label": "damage", "description": "cosmetic"}, {"label": "damage", "description": "unusable"}],
    }]})))
    .unwrap();
    let question = &request.questions[0];
    assert_eq!(read_question(0, question).unwrap().labels, ["A", "B"]);
    let answer = decision_answer(question, &read(&[0.1, 0.9], 0.8));
    assert!((answer["score"].as_f64().unwrap() - 0.9).abs() < 1e-12);
    assert_eq!(
        answer["probabilities"],
        json!([
            {"value": 0, "label": "damage", "probability": 0.1},
            {"value": 1, "label": "damage", "probability": 0.9},
        ])
    );

    let choices: Vec<Value> = (0..27).map(|i| json!({"value": i.to_string()})).collect();
    let request = parse_decisions_request(&decisions_body(
        json!({"questions": [{"type": "choice", "instructions": "x", "choices": choices}]}),
    ))
    .unwrap();
    let error = read_question(0, &request.questions[0]).unwrap_err().to_error_response();
    assert_eq!(
        error.error.message,
        "This model supports at most 26 choices per question"
    );
}

fn systemone_error(body: Value) -> String {
    parse_systemone_request(&body).unwrap_err().to_error_response().error.message
}

#[test]
fn systemone_request_rejects_like_pydantic() {
    let body =
        json!({"state": "x", "questions": {"q": {"type": "choice", "criteria": {"yes": null}}}});
    let request = parse_systemone_request(&body).unwrap();
    assert_eq!(request.priority, 0);
    assert_eq!(request.questions[0].1.instructions, json!(""));

    let with = |key: &str, value: Value| {
        let mut body = body.clone();
        body[key] = value;
        body
    };
    assert!(
        systemone_error(with("samples", json!(4)))
            .contains("'loc': ('body', 'samples'), 'msg': 'Extra inputs are not permitted'")
    );
    assert!(
        systemone_error(with("joint", json!("true"))).contains("Input should be a valid boolean")
    );
    assert!(systemone_error(with("priority", json!(1.5))).contains("fractional part"));
    assert_eq!(
        parse_systemone_request(&with("priority", json!("7"))).unwrap().priority,
        7
    );
    assert!(systemone_error(with("images", json!([1]))).contains("('body', 'images', 0)"));
    assert!(
        systemone_error(with("cache_salt", json!("")))
            .contains("must be a non-empty string if provided")
    );
    assert!(systemone_error(json!({"questions": {"a": 1}})).starts_with(
        "2 validation errors:\n  {'type': 'missing', 'loc': ('body', 'state'), 'msg': 'Field required'"
    ));
    assert!(systemone_error(json!(["x"])).contains("'loc': ('body',)"));
    // Unknown question fields are kept for the question check, not rejected
    // here.
    let request = parse_systemone_request(&with(
        "questions",
        json!({"q": {"type": "noul", "options": 1, "extra": 2}}),
    ))
    .unwrap();
    assert_eq!(request.questions[0].1.extra, ["options", "extra"]);
}

#[test]
fn joint_prompt_numbers_each_question() {
    let ask = |qid, kind, instructions: Value, criteria| {
        build_question(qid, kind, &instructions, &criteria, 128, true).unwrap()
    };
    let questions = [
        ask(
            "team",
            "choice",
            json!("Which team?"),
            json!({"billing": "pay", "ship": null}),
        ),
        ask(
            "other",
            "choice",
            json!(""),
            json!({"billing": "pay", "ship": null}),
        ),
        ask(
            "urgent",
            "noul",
            json!("Reply within the hour?\n\n"),
            json!({"true": "a refund"}),
        ),
        ask("tone", "score", Value::Null, json!(["low", "high"])),
    ];
    assert_eq!(
        joint_content("S", &questions, 3),
        "S\n\n\
         Question 3: Which team?\nA: billing - pay\nB: ship\nAnswer with the letter of one option only.\n\n\
         Question 4:\nA: billing - pay\nB: ship\nAnswer with the letter of one option only.\n\n\
         Question 5: Reply within the hour?\n\n\nyes: a refund\nno\nAnswer with yes or no only.\n\n\
         Question 6:\n0: low\n1: high\nAnswer with the number of one level only.\n\n\
         Reply with one line per question, in order, formatted as \"number: answer\"."
    );
}

#[test]
fn liquid_prompt_matches_the_checkpoint() {
    for (kind, criteria, ask) in [
        (
            "choice",
            json!({"billing": "Charges", "app_fault": null}),
            "Which?\n\nOptions:\nA Charges\nB app fault\n\nReply with the option code only.",
        ),
        (
            "choice",
            json!({"a": "Berlin", "b": "Paris"}),
            "Which?\n\nOptions:\na Berlin\nb Paris\n\nReply with the option code only.",
        ),
        (
            "noul",
            json!({"true": "asks to cancel", "false": "anything else"}),
            "Which?\nYes: asks to cancel\nNo: anything else\n\nReply with yes or no only.",
        ),
        (
            "score",
            json!(["low", "high"]),
            "Which?\n\n0 low\n1 high\n\nReply with a single digit 0-1 only.",
        ),
    ] {
        let q = build_question("q", kind, &json!("Which?"), &criteria, 26, true).unwrap();
        assert_eq!(liquid_content("", &q), ask);
        assert_eq!(
            liquid_content("Hi.", &q),
            format!("Hi.\n\n\nQUESTION:\n{ask}")
        );
    }
    let q = build_question(
        "q",
        "noul",
        &json!("Which?"),
        &json!({"true": "yes please"}),
        26,
        true,
    )
    .unwrap();
    assert_eq!(
        liquid_content("", &q),
        "Which?\nYes: yes please\nNo: None\n\nReply with yes or no only."
    );
}

#[test]
fn state_text_keeps_unicode() {
    assert_eq!(
        state_text(&json!({"message": "Français 日本語"})),
        r#"{"message": "Français 日本語"}"#
    );
    assert_eq!(state_text(&json!("plain")), "plain");
    assert_eq!(state_text(&Value::Null), "null");
}

#[test]
fn common_prefix_len_matches_python() {
    assert_eq!(
        common_prefix_len(&[&[1, 2, 3, 4], &[1, 2, 3, 9, 9], &[1, 2, 3]]),
        3
    );
    assert_eq!(common_prefix_len(&[&[1, 2], &[3, 4]]), 0);
    assert_eq!(common_prefix_len(&[&[1, 2, 3]]), 3);
}
