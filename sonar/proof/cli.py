# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private DPHN command interface; intentionally absent from top-level help."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path


def _keygen(args):
    from sonar.proof.verifier import generate_x25519_keypair

    private, public = generate_x25519_keypair()
    path = Path(args.private_key)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(private + "\n")
    print(json.dumps({"public_key": public, "private_key_file": str(path)}))


def _check(args):
    from sonar.proof.bundle import RuntimeWeightVerifier

    key_path = Path(args.private_key)
    if os.name == "posix" and key_path.stat().st_mode & 0o077:
        raise ValueError("Private key must not be readable by group or other users; use chmod 600")
    verifier = RuntimeWeightVerifier(args.manifest, key_path.read_text().strip())
    bundle = json.loads(Path(args.proof_file).read_text())
    tokens = json.loads(
        Path(args.output_token_ids_file).read_text() if args.output_token_ids_file else args.output_token_ids
    )
    valid = verifier.verify(bundle, nonce=args.nonce, request_id=args.request_id, output_token_ids=tokens)
    print(json.dumps({"valid": valid, "expected_worker_ranks": len(verifier.workers)}))
    if not valid:
        raise SystemExit(1)


async def _snapshot(args):
    from sonar.engine.arg_utils import AsyncEngineArgs
    from sonar.proof import PROOF_VERSION
    from sonar.v1.engine.async_llm import AsyncLLM

    directory = Path(args.output_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Snapshot output directory must be new or empty")
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLM.from_engine_args(engine_args)
    try:
        config = engine.sonar_config.dphn_network_config
        if config is None:
            raise ValueError("Runtime snapshots require --dphn-network configuration")
        workers = await engine.dphn_snapshot(str(directory))
        identities = [(worker["dp_rank"], worker["rank"]) for worker in workers]
        if not workers or len(identities) != len(set(identities)):
            raise ValueError("Snapshot export returned missing or duplicate worker ranks")
        manifest = {
            "version": PROOF_VERSION,
            "worker_id": config.worker_id,
            "model_manifest_id": config.model_manifest_id,
            "coverage": "all worker ranks managed by this engine instance",
            "workers": sorted(workers, key=lambda worker: (worker["dp_rank"], worker["rank"])),
        }
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"manifest": str(path), "worker_ranks": len(workers)}))
    finally:
        engine.shutdown()


def main(argv: list[str] | None = None):
    from sonar.utils.argparse_utils import FlexibleArgumentParser

    if argv is None:
        import sys

        argv = sys.argv[1:]
    parser = FlexibleArgumentParser(
        prog="sonar verify", description="DPHN sampled-weight receipts; not proof of inference."
    )
    actions: argparse._SubParsersAction = parser.add_subparsers(dest="action", required=True)
    keygen = actions.add_parser("keygen", help="Create an X25519 verifier key; print only the public key.")
    keygen.add_argument("--private-key", required=True, help="New mode-0600 private-key file; never overwritten.")
    keygen.set_defaults(handler=_keygen)
    check = actions.add_parser("check", help="Check a receipt bundle against approved local CPU snapshots.")
    check.add_argument("--manifest", required=True)
    check.add_argument("--private-key", required=True)
    check.add_argument("--proof-file", required=True, help="JSON x_dphn_proof choice bundle.")
    check.add_argument("--nonce", required=True, help="Expected gateway nonce, not a value copied from the proof.")
    check.add_argument("--request-id", required=True, help="Expected per-choice request ID.")
    tokens = check.add_mutually_exclusive_group(required=True)
    tokens.add_argument("--output-token-ids", help="Expected output token IDs as a JSON list.")
    tokens.add_argument("--output-token-ids-file", help="File containing the expected output token ID list.")
    check.set_defaults(handler=_check)
    snapshot = actions.add_parser("snapshot", help="Export approved runtime shards on trusted model workers.")
    # Engine arguments pull in the serving stack; load them only when needed.
    if argv[:1] == ["snapshot"]:
        from sonar.engine.arg_utils import AsyncEngineArgs

        AsyncEngineArgs.add_cli_args(snapshot)
        snapshot.set_defaults(dphn_network=True)
    snapshot.add_argument(
        "--output-dir", required=True, help="New/empty directory, shared between distributed workers."
    )
    snapshot.set_defaults(handler=lambda args: asyncio.run(_snapshot(args)))
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Verification error: {exc}\n")
