use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use aphrodite_engine_core_client::protocol::stats::PrefillStats;
use aphrodite_engine_core_client::protocol::{EngineCoreEvent, EngineCoreEventType, EngineCoreOutput};
use aphrodite_metrics::{
    EngineLabels, FinishedReasonLabels, METRICS, PromptTokenSourceLabels, RequestMetrics,
};
use tracing::info;

use crate::FinishReason;

fn metrics() -> &'static RequestMetrics {
    &METRICS.request
}

/// Env var mirroring Python's `APHRODITE_REQUEST_LEVEL_METRICS`. When truthy (the
/// default), the frontend logs one line per finished request and suppresses the
/// periodic aggregated "Avg prompt tput" log. When falsy, the frontend falls back
/// to the periodic aggregated log.
const REQUEST_LEVEL_METRICS_ENV: &str = "APHRODITE_REQUEST_LEVEL_METRICS";

/// Returns true when per-request metrics logging is enabled.
///
/// Matches Python's `envs.APHRODITE_REQUEST_LEVEL_METRICS` parsing: the env var is
/// interpreted as an integer and any non-zero value enables the feature. Unset or
/// unparseable values fall back to the Python default of `1` (enabled).
pub(crate) fn request_level_metrics_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var(REQUEST_LEVEL_METRICS_ENV) {
        Ok(raw) => raw.trim().parse::<i64>().map(|v| v != 0).unwrap_or(true),
        Err(_) => true,
    })
}

const PROMPT_TOKEN_SOURCE_LOCAL_COMPUTE: &str = "local_compute";
const PROMPT_TOKEN_SOURCE_LOCAL_CACHE_HIT: &str = "local_cache_hit";
const PROMPT_TOKEN_SOURCE_EXTERNAL_KV_TRANSFER: &str = "external_kv_transfer";

/// Request-scoped metrics state tracked across streamed engine-core updates.
///
/// This is the Rust-side counterpart of the Python frontend's request-lifecycle bookkeeping,
/// centered on `RequestStateStats` and the per-output/per-finished update flow.
#[derive(Debug, Clone)]
pub(crate) struct RequestMetricsTracker {
    model_name: String,
    arrival_time: f64,
    prompt_len: u32,
    max_tokens_param: Option<u32>,
    n_param: u32,
    is_prefilling: bool,
    queued_ts: f64,
    scheduled_ts: f64,
    first_token_ts: f64,
    last_token_ts: f64,
    first_token_latency: f64,
    num_generation_tokens: u32,
    latest_num_cached_tokens: u32,
    last_seen_engine_index: u32,
    log_request_level: bool,
    logged_finished: bool,
}

impl RequestMetricsTracker {
    /// Create the per-request tracker from the normalized `llm`-layer request context.
    pub(crate) fn new(
        model_name: String,
        arrival_time: f64,
        prompt_len: u32,
        max_tokens_param: Option<u32>,
        n_param: u32,
        log_request_level: bool,
    ) -> Self {
        Self {
            model_name,
            arrival_time,
            prompt_len,
            max_tokens_param,
            n_param,
            is_prefilling: true,
            queued_ts: 0.0,
            scheduled_ts: 0.0,
            first_token_ts: 0.0,
            last_token_ts: 0.0,
            first_token_latency: 0.0,
            num_generation_tokens: 0,
            latest_num_cached_tokens: 0,
            last_seen_engine_index: 0,
            log_request_level,
            logged_finished: false,
        }
    }

    /// Update request-lifecycle state from one engine-core output item.
    pub(crate) fn observe_output(
        &mut self,
        engine_index: u32,
        batch_timestamp: f64,
        received_at: f64,
        output: &EngineCoreOutput,
    ) {
        self.last_seen_engine_index = engine_index;
        if let Some(prefill_stats) = &output.prefill_stats {
            self.latest_num_cached_tokens = prefill_stats.num_cached_tokens;
        }
        self.num_generation_tokens += output.new_token_ids.len() as u32;
        metrics()
            .generation_tokens
            .get_or_create(&engine_labels(&self.model_name, engine_index))
            .inc_by(output.new_token_ids.len() as u64);

        if let Some(events) = &output.events {
            self.observe_events(engine_index, events);
        }

        if self.is_prefilling {
            if let Some(prefill_stats) = &output.prefill_stats {
                record_prompt_tokens(&self.model_name, engine_index, prefill_stats);
            }
            self.first_token_latency = received_at - self.arrival_time;
            observe_time_to_first_token_seconds(
                &self.model_name,
                engine_index,
                self.first_token_latency,
            );
            self.first_token_ts = batch_timestamp;
            self.is_prefilling = false;
        } else if self.last_token_ts > 0.0 {
            observe_inter_token_latency_seconds(
                &self.model_name,
                engine_index,
                batch_timestamp - self.last_token_ts,
            );
        }

        self.last_token_ts = batch_timestamp;
    }

    /// Emit the terminal request metrics once a finished output has been observed.
    pub(crate) fn record_finished(&mut self, received_at: f64, finish_reason: FinishReason) {
        // Guard against double-recording: `record_finished` is called both from the normal
        // stream termination path and from the `GenerateOutputStream::Drop` abort path.
        if self.logged_finished {
            return;
        }
        self.logged_finished = true;

        let labels = engine_labels(&self.model_name, self.last_seen_engine_index);
        let prefill_kv_computed_tokens = self
            .prompt_len
            .saturating_sub(self.latest_num_cached_tokens);
        let e2e_latency_seconds = received_at - self.arrival_time;
        let queue_time_seconds = diff_or_zero(self.scheduled_ts, self.queued_ts);
        let prefill_time_seconds = diff_or_zero(self.first_token_ts, self.scheduled_ts);
        let decode_time_seconds = diff_or_zero(self.last_token_ts, self.first_token_ts);
        let inference_time_seconds = diff_or_zero(self.last_token_ts, self.scheduled_ts);
        let time_per_output_token_seconds = if self.num_generation_tokens > 1 {
            diff_or_zero(self.last_token_ts, self.first_token_ts)
                / (self.num_generation_tokens - 1) as f64
        } else {
            0.0
        };

        if self.log_request_level {
            self.log_finished_request(
                e2e_latency_seconds,
                prefill_time_seconds,
                decode_time_seconds,
            );
        }

        record_request_success(&self.model_name, self.last_seen_engine_index, finish_reason);
        metrics()
            .request_prompt_tokens
            .get_or_create(&labels)
            .observe(self.prompt_len as f64);
        metrics()
            .request_generation_tokens
            .get_or_create(&labels)
            .observe(self.num_generation_tokens as f64);
        metrics()
            .request_max_num_generation_tokens
            .get_or_create(&labels)
            .observe(self.num_generation_tokens as f64);
        if let Some(max_tokens_param) = self.max_tokens_param {
            metrics()
                .request_params_max_tokens
                .get_or_create(&labels)
                .observe(max_tokens_param as f64);
        }
        metrics()
            .request_params_n
            .get_or_create(&labels)
            .observe(self.n_param as f64);
        metrics()
            .request_prefill_kv_computed_tokens
            .get_or_create(&labels)
            .observe(prefill_kv_computed_tokens as f64);
        metrics()
            .e2e_request_latency_seconds
            .get_or_create(&labels)
            .observe(e2e_latency_seconds);
        metrics()
            .request_queue_time_seconds
            .get_or_create(&labels)
            .observe(queue_time_seconds);
        metrics()
            .request_prefill_time_seconds
            .get_or_create(&labels)
            .observe(prefill_time_seconds);
        metrics()
            .request_decode_time_seconds
            .get_or_create(&labels)
            .observe(decode_time_seconds);
        metrics()
            .request_inference_time_seconds
            .get_or_create(&labels)
            .observe(inference_time_seconds);
        metrics()
            .request_time_per_output_token_seconds
            .get_or_create(&labels)
            .observe(time_per_output_token_seconds);
    }

    /// Emit a human-readable summary for one finished request, matching the Python
    /// frontend's `_log_finished_requests` output format. This runs only when
    /// `APHRODITE_REQUEST_LEVEL_METRICS` is enabled, and replaces the periodic aggregated
    /// log line in the same way `LoggingStatLogger.log()` no-ops in Python.
    fn log_finished_request(
        &self,
        e2e_latency_seconds: f64,
        prefill_time_seconds: f64,
        decode_time_seconds: f64,
    ) {
        let prefill_throughput = if prefill_time_seconds > 0.0 {
            self.prompt_len as f64 / prefill_time_seconds
        } else {
            0.0
        };
        let decode_throughput = if decode_time_seconds > 0.0 && self.num_generation_tokens > 0 {
            self.num_generation_tokens as f64 / decode_time_seconds
        } else {
            0.0
        };

        let cache_hit_info = if self.latest_num_cached_tokens > 0 {
            let cache_hit_rate = if self.prompt_len > 0 {
                self.latest_num_cached_tokens as f64 / self.prompt_len as f64 * 100.0
            } else {
                0.0
            };
            format!(
                ", Cache hits: {} tokens ({cache_hit_rate:.1}%)",
                self.latest_num_cached_tokens,
            )
        } else {
            String::new()
        };

        info!(
            "Request completed - E2E time: {e2e_latency_seconds:.2}s, \
             TTFT: {prefill_time_seconds:.2}s, \
             Prefill: {} tokens ({prefill_throughput:.1} tokens/s), \
             Decode: {} tokens ({decode_throughput:.1} tokens/s)\
             {cache_hit_info}",
            self.prompt_len, self.num_generation_tokens,
        );
    }

    fn observe_events(&mut self, engine_index: u32, events: &[EngineCoreEvent]) {
        for event in events {
            match event.r#type {
                EngineCoreEventType::Queued => {
                    self.queued_ts = event.timestamp;
                }
                EngineCoreEventType::Scheduled => {
                    if self.scheduled_ts == 0.0 {
                        self.scheduled_ts = event.timestamp;
                    }
                }
                EngineCoreEventType::Preempted => {
                    metrics()
                        .num_preemptions
                        .get_or_create(&engine_labels(&self.model_name, engine_index))
                        .inc();
                }
            }
        }
    }
}

fn engine_labels(model_name: &str, engine: u32) -> EngineLabels {
    EngineLabels {
        model_name: model_name.to_string(),
        engine,
    }
}

fn observe_time_to_first_token_seconds(model_name: &str, engine: u32, seconds: f64) {
    metrics()
        .time_to_first_token_seconds
        .get_or_create(&engine_labels(model_name, engine))
        .observe(seconds);
}

fn observe_inter_token_latency_seconds(model_name: &str, engine: u32, seconds: f64) {
    metrics()
        .inter_token_latency_seconds
        .get_or_create(&engine_labels(model_name, engine))
        .observe(seconds);
}

fn record_request_success(model_name: &str, engine: u32, finish_reason: FinishReason) {
    metrics()
        .request_success
        .get_or_create(&FinishedReasonLabels {
            model_name: model_name.to_string(),
            engine,
            finished_reason: finish_reason.as_str(),
        })
        .inc();
}

fn prompt_token_source_labels(
    model_name: &str,
    engine: u32,
    source: &'static str,
) -> PromptTokenSourceLabels {
    PromptTokenSourceLabels {
        model_name: model_name.to_string(),
        engine,
        source,
    }
}

fn record_prompt_tokens(model_name: &str, engine: u32, prefill_stats: &PrefillStats) {
    let computed = prefill_stats.num_computed_tokens as u64;
    let local_cache_hit = prefill_stats.num_local_cached_tokens as u64;
    let external_kv_transfer = prefill_stats.num_external_cached_tokens as u64;

    metrics()
        .prompt_tokens
        .get_or_create(&engine_labels(model_name, engine))
        .inc_by(prefill_stats.num_prompt_tokens as u64);
    metrics()
        .prompt_tokens_by_source
        .get_or_create(&prompt_token_source_labels(
            model_name,
            engine,
            PROMPT_TOKEN_SOURCE_LOCAL_COMPUTE,
        ))
        .inc_by(computed);
    metrics()
        .prompt_tokens_by_source
        .get_or_create(&prompt_token_source_labels(
            model_name,
            engine,
            PROMPT_TOKEN_SOURCE_LOCAL_CACHE_HIT,
        ))
        .inc_by(local_cache_hit);
    metrics()
        .prompt_tokens_by_source
        .get_or_create(&prompt_token_source_labels(
            model_name,
            engine,
            PROMPT_TOKEN_SOURCE_EXTERNAL_KV_TRANSFER,
        ))
        .inc_by(external_kv_transfer);
    metrics()
        .prompt_tokens_cached
        .get_or_create(&engine_labels(model_name, engine))
        .inc_by(prefill_stats.num_cached_tokens as u64);
}

fn diff_or_zero(end: f64, start: f64) -> f64 {
    if end > 0.0 && start > 0.0 && end >= start {
        end - start
    } else {
        0.0
    }
}

/// Return the current wall-clock time in seconds since the Unix epoch.
///
/// This is used for frontend-side latency measurements such as TTFT and E2E, matching the Python
/// frontend's use of wall-clock request arrival/iteration timestamps rather than engine-core's
/// monotonic scheduler timestamps.
pub(crate) fn current_unix_timestamp_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before unix epoch")
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use aphrodite_engine_core_client::protocol::stats::PrefillStats;
    use aphrodite_engine_core_client::protocol::{EngineCoreEvent, EngineCoreEventType};

    use super::{RequestMetricsTracker, diff_or_zero};

    #[test]
    fn tracker_updates_timing_state_across_prefill_decode_and_finish() {
        let mut tracker =
            RequestMetricsTracker::new("model".to_string(), 100.0, 64, Some(128), 1, false);

        tracker.observe_output(
            2,
            10.0,
            100.2,
            &aphrodite_engine_core_client::protocol::EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![1],
                finish_reason: None,
                events: Some(vec![
                    EngineCoreEvent {
                        r#type: EngineCoreEventType::Queued,
                        timestamp: 8.0,
                    },
                    EngineCoreEvent {
                        r#type: EngineCoreEventType::Scheduled,
                        timestamp: 9.0,
                    },
                ]),
                prefill_stats: Some(PrefillStats {
                    num_prompt_tokens: 64,
                    num_computed_tokens: 60,
                    num_cached_tokens: 4,
                    num_local_cached_tokens: 4,
                    num_external_cached_tokens: 0,
                }),
                ..Default::default()
            },
        );
        tracker.observe_output(
            2,
            11.5,
            100.4,
            &aphrodite_engine_core_client::protocol::EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![2, 3],
                finish_reason: None,
                events: Some(vec![EngineCoreEvent {
                    r#type: EngineCoreEventType::Preempted,
                    timestamp: 10.5,
                }]),
                ..Default::default()
            },
        );

        assert!(!tracker.is_prefilling);
        assert_eq!(tracker.last_seen_engine_index, 2);
        assert_eq!(tracker.num_generation_tokens, 3);
        assert_eq!(tracker.queued_ts, 8.0);
        assert_eq!(tracker.scheduled_ts, 9.0);
        assert_eq!(tracker.first_token_ts, 10.0);
        assert_eq!(tracker.last_token_ts, 11.5);
        assert!((tracker.first_token_latency - 0.2).abs() < 1e-9);
        assert_eq!(
            diff_or_zero(tracker.last_token_ts, tracker.first_token_ts),
            1.5
        );
    }
}
