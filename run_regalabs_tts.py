#!/usr/bin/env python3
"""RegaLabs-TTS: Standalone Sorani Text-to-Speech Runner."""

import argparse
import sys
import os
from pathlib import Path
import torch
import soundfile as sf

# Add sorani module to path
sys.path.insert(0, str(Path(__file__).parent))
from sorani.frontend import normalize_sorani_text
from sorani.censor import verify_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="RegaLabs-TTS Sorani Synthesis")
    parser.add_argument("--text", required=True, help="Sorani text to synthesize")
    parser.add_argument("--prompt-wav", required=True, help="Path to reference audio prompt WAV")
    parser.add_argument("--prompt-text", required=True, help="Transcript of reference audio prompt")
    parser.add_argument("--flow-checkpoint", required=True, help="Path to cosyvoice3_sorani_flow_best_step2300.pt")
    parser.add_argument("--base-model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512", help="Hugging Face model ID or path for base model")
    parser.add_argument("--out", default="output_sorani.wav", help="Output WAV path")
    parser.add_argument("--cosyvoice-repo", default=os.environ.get("COSYVOICE_REPO", "./CosyVoice"), help="Path to cloned CosyVoice repo")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Enable CosyVoice imports
    repo = Path(args.cosyvoice-repo if hasattr(args, "cosyvoice_repo") else args.cosyvoice_repo).resolve()
    if not repo.exists():
        print(f"Error: CosyVoice directory not found at {repo}. Please clone CosyVoice or set --cosyvoice-repo.", file=sys.stderr)
        sys.exit(1)
        
    sys.path.insert(0, str(repo))
    matcha_path = repo / "third_party" / "Matcha-TTS"
    if matcha_path.exists():
        sys.path.insert(0, str(matcha_path))
        
    from cosyvoice.cli.cosyvoice import CosyVoice3
    
    print(f"Loading base model: {args.base_model}...")
    cosyvoice = CosyVoice3(args.base_model, fp16=torch.cuda.is_available())
    
    verify_checkpoint(args.flow_checkpoint)
    print(f"Loading RegaLabs-TTS Sorani flow checkpoint: {args.flow_checkpoint}...")
    flow_state = torch.load(args.flow_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(flow_state, dict):
        for key in ("model", "state_dict", "flow"):
            nested = flow_state.get(key)
            if isinstance(nested, dict):
                flow_state = nested
                break
        flow_state = {k: v for k, v in flow_state.items() if isinstance(k, str) and isinstance(v, torch.Tensor)}
        
    cosyvoice.model.flow.load_state_dict(flow_state, strict=False)
    cosyvoice.model.flow.to(cosyvoice.model.device).eval()
    
    # Normalize Sorani Kurdish Text
    clean_text = normalize_sorani_text(args.text)
    clean_prompt_text = normalize_sorani_text(args.prompt_text)
    if "<|endofprompt|>" not in clean_prompt_text:
        clean_prompt_text = f"You are a helpful assistant.<|endofprompt|> {clean_prompt_text}"
    
    print(f"Normalized input text: '{clean_text}'")
    print("Synthesizing speech...")
    
    pieces = []
    with torch.inference_mode():
        for output in cosyvoice.inference_zero_shot(
            clean_text,
            clean_prompt_text,
            args.prompt_wav,
            stream=False,
            text_frontend=False,
        ):
            pieces.append(output["tts_speech"].detach().cpu())
            
    if not pieces:
        raise RuntimeError("No audio was generated.")
        
    speech = torch.cat(pieces, dim=1).squeeze(0).numpy()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, speech, cosyvoice.sample_rate, subtype="PCM_16")
    print(f"Successfully saved generated audio to: {out_path}")

if __name__ == "__main__":
    main()
