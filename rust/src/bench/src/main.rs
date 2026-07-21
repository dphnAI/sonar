// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use anyhow::Context;
use clap::Parser;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser)]
#[command(
    name = "aphrodite-bench",
    about = "Benchmark online serving throughput",
    version
)]
struct Cli {
    #[command(flatten)]
    args: aphrodite_bench::BenchServeArgs,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    aphrodite_bench::prepare_process();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("Failed to build tokio runtime")?;

    runtime.block_on(aphrodite_bench::run(cli.args))
}
