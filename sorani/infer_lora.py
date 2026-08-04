#!/usr/bin/env python3
"""Run CosyVoice 3 zero-shot inference with Sorani LLM and flow adaptation."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument(
        "--flow-checkpoint",
        help="Optional full flow/acoustic checkpoint to load after the base model.",
    )
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument(
        "--prompt-instruct",
        default="You are a helpful assistant.<|endofprompt|>",
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--normalizer", type=Path)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--speed", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = os.environ.get("COSYVOICE_REPO", ".")
    sys.path.insert(0, repo)
    matcha = Path(repo) / "third_party" / "Matcha-TTS"
    if matcha.exists():
        sys.path.insert(0, str(matcha))

    from cosyvoice.cli.cosyvoice import CosyVoice3
    from cosyvoice.utils.lora import inject_lora, load_lora_state_dict

    if not torch.cuda.is_available():
        raise RuntimeError("The inference test requires a GPU.")

    cosyvoice = CosyVoice3(args.model_dir, fp16=True)
    llm = cosyvoice.model.llm
    target_count = inject_lora(llm, rank=args.rank, alpha=args.alpha, dropout=0.0)
    adapter = torch.load(args.adapter, map_location="cpu", weights_only=False)
    load_lora_state_dict(llm, adapter)
    llm.to(cosyvoice.model.device).eval()
    print(f"loaded adapter with {target_count} target projections", flush=True)

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
        print(f"loaded flow checkpoint {args.flow_checkpoint}", flush=True)

    normalize = load_normalizer(args.normalizer) if args.normalizer else lambda text: text
    prompt_text = normalize(args.prompt_text)
    text = normalize(args.text)
    if "<|endofprompt|>" not in prompt_text:
        prompt_text = f"{args.prompt_instruct} {prompt_text}"

    pieces = []
    with torch.inference_mode():
        for output in cosyvoice.inference_zero_shot(
            text,
            prompt_text,
            args.prompt_wav,
            stream=False,
            speed=args.speed,
            text_frontend=False,
        ):
            pieces.append(output["tts_speech"].detach().cpu())

    if not pieces:
        raise RuntimeError("CosyVoice returned no audio.")
    speech = torch.cat(pieces, dim=1).squeeze(0).numpy()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, speech, cosyvoice.sample_rate, subtype="PCM_16")
    print(f"wrote {out} ({len(speech) / cosyvoice.sample_rate:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
