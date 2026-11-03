// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use aphrodite_engine_core_client::protocol::sampling::EngineCoreSamplingParams;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum SamplingParamsError {
    #[error("{parameter} must be a finite number, got {value}")]
    NotFinite { parameter: &'static str, value: f32 },
    #[error("{parameter} must be in {expected}, got {value}")]
    OutOfRange {
        parameter: &'static str,
        value: f32,
        expected: &'static str,
    },
    #[error(
        "routed_experts_prompt_start must be between 0 and the prompt length \
         ({prompt_len}), inclusive, got {value}"
    )]
    RoutedExpertsPromptStartOutOfRange { value: u32, prompt_len: u32 },
}

fn validate_frequency_penalty(value: f32) -> Result<(), SamplingParamsError> {
    validate_closed_range("frequency_penalty", value, -2.0, 2.0, "[-2, 2]")
}

fn validate_presence_penalty(value: f32) -> Result<(), SamplingParamsError> {
    validate_closed_range("presence_penalty", value, -2.0, 2.0, "[-2, 2]")
}

fn validate_temperature(value: f32) -> Result<(), SamplingParamsError> {
    validate_finite("temperature", value)?;
    validate_closed_range("temperature", value, 0.0, 2.0, "[0, 2]")
}

fn validate_top_p(value: f32) -> Result<(), SamplingParamsError> {
    if value > 0.0 && value <= 1.0 {
        return Ok(());
    }
    Err(SamplingParamsError::OutOfRange {
        parameter: "top_p",
        value,
        expected: "(0, 1]",
    })
}

fn validate_min_p(value: f32) -> Result<(), SamplingParamsError> {
    validate_closed_range("min_p", value, 0.0, 1.0, "[0, 1]")
}

fn validate_repetition_penalty(value: f32) -> Result<(), SamplingParamsError> {
    validate_finite("repetition_penalty", value)?;
    if value > 0.0 {
        return Ok(());
    }
    Err(SamplingParamsError::OutOfRange {
        parameter: "repetition_penalty",
        value,
        expected: "(0, inf)",
    })
}

pub(crate) fn validate_resolved_sampling_params(
    params: &EngineCoreSamplingParams,
) -> Result<(), SamplingParamsError> {
    validate_temperature(params.temperature)?;
    validate_top_p(params.top_p)?;
    validate_min_p(params.min_p)?;
    validate_frequency_penalty(params.frequency_penalty)?;
    validate_presence_penalty(params.presence_penalty)?;
    validate_repetition_penalty(params.repetition_penalty)?;

    let extra = &params.aphrodite_sampling;
    for (parameter, value) in [
        ("dynatemp_min", extra.dynatemp_min),
        ("dynatemp_max", extra.dynatemp_max),
        ("top_a", extra.top_a),
        ("eta_cutoff", extra.eta_cutoff),
        ("epsilon_cutoff", extra.epsilon_cutoff),
        ("xtc_threshold", extra.xtc_threshold),
        ("nsigma", extra.nsigma),
        ("mirostat_tau", extra.mirostat_tau),
        ("mirostat_eta", extra.mirostat_eta),
        ("dry_multiplier", extra.dry_multiplier),
    ] {
        validate_finite(parameter, value)?;
        validate_closed_range(parameter, value, 0.0, f32::MAX, "[0, inf)")?;
    }
    for (parameter, value, min) in [
        ("dynatemp_exponent", extra.dynatemp_exponent, 0.0),
        ("dry_base", extra.dry_base, 1.0),
    ] {
        validate_finite(parameter, value)?;
        if value <= min {
            return Err(SamplingParamsError::OutOfRange {
                parameter,
                value,
                expected: if min == 0.0 { "(0, inf)" } else { "(1, inf)" },
            });
        }
    }
    for (parameter, value) in [
        ("typical_p", extra.typical_p),
        ("xtc_probability", extra.xtc_probability),
    ] {
        validate_closed_range(parameter, value, 0.0, 1.0, "[0, 1]")?;
    }
    if !(extra.tfs > 0.0 && extra.tfs <= 1.0) {
        return Err(SamplingParamsError::OutOfRange {
            parameter: "tfs",
            value: extra.tfs,
            expected: "(0, 1]",
        });
    }
    if !matches!(extra.mirostat_mode, 0 | 2) {
        return Err(SamplingParamsError::OutOfRange {
            parameter: "mirostat_mode",
            value: extra.mirostat_mode as f32,
            expected: "{0, 2}",
        });
    }
    Ok(())
}

fn validate_finite(parameter: &'static str, value: f32) -> Result<(), SamplingParamsError> {
    if value.is_finite() {
        return Ok(());
    }
    Err(SamplingParamsError::NotFinite { parameter, value })
}

fn validate_closed_range(
    parameter: &'static str,
    value: f32,
    min: f32,
    max: f32,
    expected: &'static str,
) -> Result<(), SamplingParamsError> {
    if value >= min && value <= max {
        return Ok(());
    }
    Err(SamplingParamsError::OutOfRange {
        parameter,
        value,
        expected,
    })
}
