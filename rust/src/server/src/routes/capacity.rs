// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde_json::{Value, json};
use sonar_engine_core_client::protocol::handshake::EngineCoreReadyResponse;

use crate::state::AppState;

/// Return startup capacity for connected DP engines, not live availability.
pub async fn capacity(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(capacity_response(
        state.engine_core_client().ready_responses(),
    ))
}

fn capacity_response(mut reports: Vec<&EngineCoreReadyResponse>) -> Value {
    reports.sort_by_key(|report| report.data_parallel_rank);
    let engines: Vec<_> = reports
        .iter()
        .map(|report| {
            json!({
                "dp_rank": report.data_parallel_rank,
                "max_model_len": report.max_model_len,
                "max_num_seqs": report.max_num_seqs,
                "kv_cache_capacity_tokens": report.kv_cache_size_tokens,
                "estimated_concurrency_at_max_model_len": report.kv_cache_max_concurrency,
            })
        })
        .collect();
    json!({"schema_version": 1, "scope": "connected_engines", "engines": engines})
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonar_engine_core_client::mock_engine::default_ready_response;

    #[test]
    fn capacity_preserves_distinct_replica_values_in_rank_order() {
        let mut first = default_ready_response();
        first.data_parallel_rank = 1;
        first.kv_cache_size_tokens = Some(7850);
        first.kv_cache_max_concurrency = Some(7.85);
        let mut second = first.clone();
        second.data_parallel_rank = 3;
        second.kv_cache_size_tokens = Some(9100);
        second.kv_cache_max_concurrency = Some(9.1);
        let result = capacity_response(vec![&second, &first]);
        assert_eq!(result["engines"][0]["dp_rank"], 1);
        assert_eq!(result["engines"][1]["dp_rank"], 3);
        assert_eq!(result["engines"][0]["kv_cache_capacity_tokens"], 7850);
        assert_eq!(result["engines"][1]["kv_cache_capacity_tokens"], 9100);
        assert_eq!(
            result["engines"][0]["estimated_concurrency_at_max_model_len"],
            7.85
        );
        assert_eq!(
            result["engines"][1]["estimated_concurrency_at_max_model_len"],
            9.1
        );
    }
}
