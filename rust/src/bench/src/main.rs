// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use anyhow::Context;
use clap::{Parser, Subcommand};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser)]
#[command(
    name = "sonar-bench",
    about = "Benchmark online serving throughput and offline multimodal preprocessing",
    version = sonar_build_info::VERSION
)]
struct Cli {
    #[command(flatten)]
    args: sonar_bench::BenchServeArgs,

    /// Optional subcommand; absent by default for the online serving benchmark.
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Offline multimodal preprocessing latency benchmark.
    MmProcessor(sonar_bench::MmProcessorArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    sonar_bench::prepare_process();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("Failed to build tokio runtime")?;

    match cli.command {
        Some(Command::MmProcessor(args)) => {
            let (timing_layer, timing_stats) = sonar_chat::mm_timing_layer();
            sonar_tracing::init_tracing_with("Bench", timing_layer);
            runtime.block_on(sonar_bench::run_mm_processor(args, timing_stats))
        }
        None => {
            sonar_tracing::init_tracing("Bench");
            runtime.block_on(sonar_bench::run(cli.args))
        }
    }
}
