// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;
use thiserror_ext::{AsReport as _, Macro};

use crate::RendererSelection;

type BoxedError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Error, Macro)]
#[thiserror_ext(macro(path = "crate::error"))]
pub enum Error {
    #[error("chat request must contain at least one message")]
    EmptyMessages,
    #[error("cannot continue the final message when the last message is not from the assistant")]
    ContinueFinalAssistantWithoutFinalAssistant,
    #[error("chat template is required but none was configured")]
    MissingChatTemplate,
    #[error("chat template error: {0}")]
    ChatTemplate(String),
    #[error("{message}")]
    ChatTemplateThrown { message: String },
    #[error("{0}")]
    InvalidReasoningEffort(String),
    #[error("{message}")]
    InvalidReasoningControl { message: String },
    #[error("chat role `{role}` is not supported by this chat renderer")]
    UnsupportedChatRole { role: String },
    #[error(
        "Chat template is passed with request, but --trust-request-chat-template is not set. \
         Refused request with untrusted chat template."
    )]
    UntrustedRequestChatTemplate,
    #[error("chat_template is not supported by this chat renderer")]
    UnsupportedRequestChatTemplate,
    #[error(
        "--chat-template is not supported by the `{renderer}` renderer; use --tokenizer-mode hf to render a Jinja chat template"
    )]
    UnsupportedChatTemplateOption { renderer: RendererSelection },
    #[error("multimodal input is not supported by this chat renderer")]
    UnsupportedMultimodalRenderer,
    #[error("unsupported multimodal content: {0}")]
    UnsupportedMultimodalContent(&'static str),
    #[error("`{0}` content parts are not supported by this chat renderer")]
    UnsupportedContentPart(&'static str),
    #[error("`{modality}` input is not supported by this model")]
    UnsupportedModality { modality: String },
    #[error("At most {limit} {modality}(s) may be provided in one prompt.")]
    MmLimitExceeded { modality: String, limit: usize },
    #[error("invalid inline multimodal features: {message}")]
    InvalidPreprocessedMultimodal { message: String },
    /// A media URL or payload the request got wrong, or one the server's
    /// media policy refuses (Python raises `ValueError`: 400).
    #[error("{0}")]
    InvalidMedia(String),
    /// Media the origin answers with a client error, or a download over the
    /// size cap (Python's `_wrap_media_fetch_error`: 422).
    #[error("{0}")]
    UnprocessableMedia(String),
    #[error("multimodal preprocessing error: {0}")]
    Multimodal(#[message] String),
    #[error("{kind} parsing is not available for model `{model_id}`")]
    ParserUnavailableForModel {
        kind: &'static str,
        model_id: String,
    },
    #[error("{kind} parsing is disabled by frontend configuration")]
    ParserDisabled { kind: &'static str },
    #[error(
        "unified parsing requires the tool and reasoning selections to resolve to the same parser; resolved tool={tool}, reasoning={reasoning}"
    )]
    IncompatibleParserSelections { tool: String, reasoning: String },
    #[error(
        "{kind} parser `{name}` is not registered{}",
        available_parser_hint(.available_names)
    )]
    ParserUnavailableByName {
        kind: &'static str,
        name: String,
        available_names: Vec<String>,
    },
    #[error("failed to initialize {kind} parser `{name}`")]
    ParserInitialization {
        kind: &'static str,
        name: String,
        #[source]
        error: BoxedError,
    },
    #[error("failed to initialize request output parser")]
    OutputParserInitialization {
        #[source]
        error: BoxedError,
    },
    #[error(
        "gpt_oss uses native Harmony output parsing; generic {kind} parser override `{selection}` is not supported"
    )]
    HarmonyParserOverrideUnsupported {
        kind: &'static str,
        selection: String,
    },
    #[error("harmony output parsing failed")]
    HarmonyOutputParsing {
        #[source]
        error: BoxedError,
    },
    #[error("chat request stream `{request_id}` closed before terminal output")]
    StreamClosedBeforeTerminalOutput { request_id: String },
    #[error("tool call stream state is inconsistent: {message}")]
    ToolCallStreamInvariant { message: String },
    #[error("duplicate tool name `{name}`")]
    DuplicateToolName { name: String },
    #[error("tool_choice requires at least one available tool")]
    ToolChoiceRequiresTools,
    #[error("tool_choice function `{name}` was not found in the available tools")]
    ToolChoiceFunctionNotFound { name: String },
    #[error("tool_reference `{name}` was not found in the available tools")]
    ToolReferenceNotFound { name: String },
    #[error("at least one tool must have defer_loading=false; all tools cannot be deferred")]
    AllToolsDeferred,
    #[error("failed to build output grammar")]
    OutputGrammar {
        #[source]
        error: BoxedError,
    },
    #[error(transparent)]
    Text(#[from] sonar_text::Error),
    #[error(transparent)]
    Tokenizer(#[from] sonar_tokenizer::TokenizerError),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Whether this error represents invalid user request parameters.
    pub fn is_request_validation_error(&self) -> bool {
        match self {
            Self::ChatTemplateThrown { .. }
            | Self::InvalidReasoningEffort(_)
            | Self::InvalidReasoningControl { .. }
            | Self::DuplicateToolName { .. }
            | Self::ToolChoiceRequiresTools
            | Self::ToolChoiceFunctionNotFound { .. }
            | Self::ToolReferenceNotFound { .. }
            | Self::AllToolsDeferred
            | Self::UnsupportedChatRole { .. }
            | Self::UntrustedRequestChatTemplate
            | Self::UnsupportedRequestChatTemplate
            | Self::Tokenizer(sonar_tokenizer::TokenizerError::Unavailable) => true,
            Self::Text(error) => error.is_request_validation_error(),
            Self::UnsupportedMultimodalRenderer
            | Self::UnsupportedMultimodalContent(_)
            | Self::UnsupportedContentPart(_)
            | Self::UnsupportedModality { .. }
            | Self::InvalidPreprocessedMultimodal { .. }
            | Self::InvalidMedia(_)
            | Self::UnprocessableMedia(_)
            | Self::MmLimitExceeded { .. } => true,

            _ => false,
        }
    }

    /// The request parameter a validation error refers to, reported as the
    /// `param` of the OpenAI error object.
    pub fn param(&self) -> Option<&'static str> {
        match self {
            Self::Text(error) => error.param(),
            Self::UnsupportedRequestChatTemplate => Some("chat_template"),
            Self::UnprocessableMedia(_) => Some("image_url"),
            _ => None,
        }
    }

    /// Whether the request names media that cannot be processed, answered
    /// with HTTP 422 like Python's `SonarUnprocessableEntityError`.
    pub fn is_unprocessable_entity(&self) -> bool {
        matches!(self, Self::UnprocessableMedia(_))
    }

    /// Whether the request could not be taken now, so retrying later may
    /// succeed.
    pub fn is_unavailable(&self) -> bool {
        matches!(self, Self::Text(error) if error.is_unavailable())
    }
}

/// Answer media failures the way Python's media connector does. Refusals of
/// the request's own URL or payload are 400s, origin 4xx answers (but 408 and
/// 429) and downloads over the cap are 422s, and everything else (connection
/// failures, timeouts, 5xx answers, audio and video decoding) stays a server
/// error.
impl From<llm_multimodal::MediaConnectorError> for Error {
    fn from(error: llm_multimodal::MediaConnectorError) -> Self {
        use llm_multimodal::MediaConnectorError as E;

        match &error {
            E::InvalidUrl(_)
            | E::UnsupportedScheme(_)
            | E::DisallowedDomain(_)
            | E::DisallowedAddress { .. }
            | E::DisallowedRedirect { .. }
            | E::TooManyRedirects { .. }
            | E::DisallowedLocalPath(_)
            | E::DataUrl(_)
            | E::Base64Decode(_) => Self::InvalidMedia(error.to_report_string()),
            // Python's image loader turns an undecodable image into a
            // `ValueError`; decoder limits are not part of that.
            E::Image(image) if !matches!(image, image::ImageError::Limits(_)) => {
                Self::InvalidMedia(error.to_report_string())
            }
            E::Http(http)
                if http.status().is_some_and(|status| {
                    status.is_client_error() && !matches!(status.as_u16(), 408 | 429)
                }) =>
            {
                let status = http.status().map_or(0, |status| status.as_u16());
                Self::UnprocessableMedia(format!(
                    "Failed to fetch media from URL: HTTP {status} error"
                ))
            }
            E::PayloadTooLarge { .. } => Self::UnprocessableMedia(format!(
                "Failed to fetch media from URL: {}",
                error.to_report_string()
            )),
            _ => Self::Multimodal(error.to_report_string()),
        }
    }
}

impl From<llm_multimodal::MultiModalError> for Error {
    fn from(error: llm_multimodal::MultiModalError) -> Self {
        match error {
            llm_multimodal::MultiModalError::Media(error) => error.into(),
            error => Self::Multimodal(error.to_report_string()),
        }
    }
}

impl From<llm_multimodal::TransformError> for Error {
    fn from(error: llm_multimodal::TransformError) -> Self {
        Self::Multimodal(error.to_report_string())
    }
}

impl From<llm_multimodal::registry::ModelRegistryError> for Error {
    fn from(error: llm_multimodal::registry::ModelRegistryError) -> Self {
        Self::Multimodal(error.to_report_string())
    }
}

/// Format the available-parser suffix used in user-facing error messages.
fn available_parser_hint(available_names: &[String]) -> String {
    if available_names.is_empty() {
        String::new()
    } else {
        format!(" (choose from: {})", available_names.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use std::net::IpAddr;
    use std::time::Duration;

    use llm_multimodal::{MediaConnectorError, MultiModalError};

    use super::*;

    /// The status class each media failure is answered with: 400, 422, or a
    /// server error.
    fn class(error: MediaConnectorError) -> &'static str {
        match Error::from(MultiModalError::Media(error)) {
            error @ Error::InvalidMedia(_) => {
                assert!(error.is_request_validation_error());
                "400"
            }
            error @ Error::UnprocessableMedia(_) => {
                assert!(error.is_unprocessable_entity());
                assert_eq!(error.param(), Some("image_url"));
                "422"
            }
            error @ Error::Multimodal(_) => {
                assert!(!error.is_request_validation_error());
                "500"
            }
            error => panic!("unexpected error {error:?}"),
        }
    }

    #[test]
    fn media_refusals_are_client_errors_and_fetch_failures_are_not() {
        let refused = [
            MediaConnectorError::DisallowedDomain("evil.example".to_string()),
            MediaConnectorError::DisallowedAddress {
                host: "media.example".to_string(),
                address: IpAddr::from([10, 0, 0, 5]),
                class: "private",
            },
            MediaConnectorError::DisallowedRedirect {
                url: "http://media.example/a.png".to_string(),
                target: "http://10.0.0.5/a.png".to_string(),
                reason: "private".to_string(),
            },
            MediaConnectorError::TooManyRedirects {
                url: "http://media.example/a.png".to_string(),
                limit: 10,
            },
            MediaConnectorError::UnsupportedScheme("ftp".to_string()),
            MediaConnectorError::InvalidUrl("not a url".to_string()),
        ];
        for error in refused {
            assert_eq!(class(error), "400");
        }
        assert_eq!(
            class(MediaConnectorError::PayloadTooLarge {
                media: "image",
                limit: 1,
            }),
            "422"
        );
        assert_eq!(
            class(MediaConnectorError::Timeout {
                url: "http://media.example/a.png".to_string(),
                budget: Duration::from_secs(1),
            }),
            "500"
        );
        assert_eq!(
            class(MediaConnectorError::UnresolvableHost {
                host: "media.example".to_string(),
                source: std::io::Error::other("no such host"),
            }),
            "500"
        );
    }
}
