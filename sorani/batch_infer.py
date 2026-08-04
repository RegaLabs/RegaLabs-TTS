#!/usr/bin/env python3
"""Generate a Sorani regression WAV for every row in a TSV pair file."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

import soundfile as sf
import torch


def load_normalizer(path: Path):
    spec = importlib.util.spec_from_file_location("sorani_normalizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load normalizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_kurdish_text


def load_adapted_model(args):
    repo = os.environ.get("COSYVOICE_REPO", ".")
    sys.path.insert(0, repo)
    matcha = Path(repo) / "third_party" / "Matcha-TTS"
    if matcha.exists():
        sys.path.insert(0, str(matcha))

    from cosyvoice.cli.cosyvoice import CosyVoice3
    from cosyvoice.utils.lora import inject_lora, load_lora_state_dict

    if not torch.cuda.is_available():
        raise RuntimeError("Batch inference requires a GPU.")

    cosyvoice = CosyVoice3(args.model_dir, fp16=True)
    llm = cosyvoice.model.llm
    target_count = inject_lora(llm, rank=args.rank, alpha=args.alpha, dropout=0.0)
    adapter = torch.load(args.adapter, map_location="cpu", weights_only=False)
    load_lora_state_dict(llm, adapter)
    llm.to(cosyvoice.model.device).eval()

    if args.flow_checkpoint:
        flow_state = torch.load(
            args.flow_checkpoint, map_location="cpu", weights_only=False
        )
        if isinstance(flow_state, dict):
            for key in ("model", "state_dict", "flow"):
                nested = flow_state.get(key)
                if isinstance(nested, dict):
                    flow_state = nested
                    break
            flow_state = {
                key: value
                for key, value in flow_state.items()
                if isinstance(key, str) and isinstance(value, torch.Tensor)
            }
        missing, unexpected = cosyvoice.model.flow.load_state_dict(
            flow_state, strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                "Flow checkpoint did not match the base flow model: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        cosyvoice.model.flow.to(cosyvoice.model.device).eval()

    print(f"loaded LLM adapter ({target_count} target projections)", flush=True)
    if args.flow_checkpoint:
        print(f"loaded flow checkpoint {args.flow_checkpoint}", flush=True)
    return cosyvoice


def synthesize(cosyvoice, text: str, prompt_text: str, prompt_wav: str, speed: float):
    pieces = []
    with torch.inference_mode():
        for output in cosyvoice.inference_zero_shot(
            text,
            prompt_text,
            prompt_wav,
            stream=False,
            speed=speed,
            text_frontend=False,
        ):
            pieces.append(output["tts_speech"].detach().cpu())
    if not pieces:
        raise RuntimeError("CosyVoice returned no audio.")
    return torch.cat(pieces, dim=1).squeeze(0).numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--flow-checkpoint")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path)
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument(
        "--prompt-instruct",
        default="You are a helpful assistant.<|endofprompt|>",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalize = load_normalizer(args.normalizer) if args.normalizer else lambda text: text
    rows = []
    with args.pairs.open(encoding="utf-8", newline="") as pairs_file:
        for row in csv.DictReader(pairs_file, delimiter="\t"):
            if row.get("id") and row.get("sorani"):
                rows.append(row)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("The pair file contains no usable Sorani rows.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "results.tsv"
    cosyvoice = load_adapted_model(args)
    prompt_text = normalize(args.prompt_text)
    if "<|endofprompt|>" not in prompt_text:
        prompt_text = f"{args.prompt_instruct} {prompt_text}"

    with results_path.open("w", encoding="utf-8", newline="") as results_file:
        writer = csv.DictWriter(
            results_file,
            fieldnames=["id", "english", "raw_sorani", "used_sorani", "wav", "seconds", "status"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            raw_sorani = row["sorani"].strip()
            used_sorani = normalize(raw_sorani)
            out = args.out_dir / f"{row['id']}.wav"
            try:
                speech = synthesize(
                    cosyvoice,
                    used_sorani,
                    prompt_text,
                    args.prompt_wav,
                    args.speed,
                )
                sf.write(out, speech, cosyvoice.sample_rate, subtype="PCM_16")
                status = "ok"
                seconds = f"{len(speech) / cosyvoice.sample_rate:.3f}"
                print(f"[{index}/{len(rows)}] {row['id']} -> {seconds}s", flush=True)
            except Exception as exc:  # keep the rest of the regression set running
                status = f"error:{type(exc).__name__}:{exc}"
                seconds = ""
                print(f"[{index}/{len(rows)}] {row['id']} FAILED: {exc}", flush=True)
            writer.writerow({
                "id": row["id"],
                "english": row["english"],
                "raw_sorani": raw_sorani,
                "used_sorani": used_sorani,
                "wav": str(out) if status == "ok" else "",
                "seconds": seconds,
                "status": status,
            })
            results_file.flush()


if __name__ == "__main__":
    main()
