import os
import sys
from pathlib import Path
import torch
import soundfile as sf
import tempfile
import gradio as gr

# Ensure local imports and CosyVoice path are configured
ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

# Ensure CosyVoice repo is accessible
COSYVOICE_REPO = os.environ.get("COSYVOICE_REPO", "./CosyVoice")
if os.path.exists(COSYVOICE_REPO):
    sys.path.insert(0, COSYVOICE_REPO)
    matcha_path = Path(COSYVOICE_REPO) / "third_party" / "Matcha-TTS"
    if matcha_path.exists():
        sys.path.insert(0, str(matcha_path))

# Sorani text normalizer fallback logic
try:
    from sorani.frontend import normalize_sorani_text
except ImportError:
    import unicodedata
    import re

    try:
        from sorani.censor import censor_text, verify_wordlist
    except ImportError:
        from sorani.censor import CensorIntegrityError

        def censor_text(text):
            raise CensorIntegrityError(
                "Sorani censorship module (sorani/censor.py) is missing. "
                "RegaLabs-TTS refuses to synthesize without it."
            )

        def verify_wordlist():
            raise CensorIntegrityError("Sorani censorship module is missing.")

    _CHARACTER_MAP = str.maketrans({"ك": "ک", "ي": "ی", "ى": "ی", "ة": "ە"})
    def normalize_sorani_text(text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text).translate(_CHARACTER_MAP)
        text = re.sub(r"[\u064B-\u065F]", "", text)
        text = censor_text(text)
        verify_wordlist()
        return re.sub(r"\s+", " ", text).strip()

# Global model instance
COSYVOICE_MODEL = None

def get_model():
    global COSYVOICE_MODEL
    if COSYVOICE_MODEL is None:
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice3
        except ImportError:
            raise RuntimeError(
                "CosyVoice runtime not found. Please ensure CosyVoice is cloned: "
                "git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git"
            )
        
        base_model = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
        flow_ckpt = ROOT_DIR / "cosyvoice3_sorani_flow_best_step2300.pt"
        
        print(f"Loading base CosyVoice model: {base_model}...")
        device_fp16 = torch.cuda.is_available()
        cosyvoice = CosyVoice3(base_model, fp16=device_fp16)
        
        if flow_ckpt.exists():
            from sorani.censor import verify_checkpoint
            verify_checkpoint(flow_ckpt)
            print(f"Loading RegaLabs-TTS Sorani flow weights: {flow_ckpt}...")
            flow_state = torch.load(flow_ckpt, map_location="cpu", weights_only=False)
            if isinstance(flow_state, dict):
                for key in ("model", "state_dict", "flow"):
                    nested = flow_state.get(key)
                    if isinstance(nested, dict):
                        flow_state = nested
                        break
                flow_state = {k: v for k, v in flow_state.items() if isinstance(k, str) and isinstance(v, torch.Tensor)}
            
            cosyvoice.model.flow.load_state_dict(flow_state, strict=False)
            cosyvoice.model.flow.to(cosyvoice.model.device).eval()
            print("Successfully loaded RegaLabs-TTS Sorani flow model.")
            
        COSYVOICE_MODEL = cosyvoice
        
    return COSYVOICE_MODEL

def synthesize_sorani(text, prompt_wav, prompt_text, speed, do_normalize):
    if not text or not text.strip():
        raise gr.Error("Please enter Sorani Kurdish text to synthesize.")
    if not prompt_wav:
        raise gr.Error("Please upload or select a reference prompt audio WAV.")
    if not prompt_text or not prompt_text.strip():
        raise gr.Error("Please provide the transcript text for the reference prompt audio.")

    # Text Normalization
    clean_text = normalize_sorani_text(text) if do_normalize else text.strip()
    clean_prompt_text = normalize_sorani_text(prompt_text) if do_normalize else prompt_text.strip()

    model = get_model()

    pieces = []
    with torch.inference_mode():
        for output in model.inference_zero_shot(
            clean_text,
            clean_prompt_text,
            prompt_wav,
            stream=False,
            speed=speed,
            text_frontend=False,
        ):
            pieces.append(output["tts_speech"].detach().cpu())

    if not pieces:
        raise RuntimeError("No audio was returned from the TTS model.")

    speech = torch.cat(pieces, dim=1).squeeze(0).numpy()
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        sf.write(tmp_file.name, speech, model.sample_rate, subtype="PCM_16")
        out_wav_path = tmp_file.name

    return out_wav_path, f"✅ Synthesis complete! Normalized Text: '{clean_text}'"

# Default sample files
sample_wav = str(ROOT_DIR / "samples" / "aran_en021.wav") if (ROOT_DIR / "samples" / "aran_en021.wav").exists() else None

# Custom CSS
custom_css = """
.container { max-width: 900px; margin: auto; }
.header { text-align: center; margin-bottom: 20px; }
.attribution { background-color: #1a1a24; border-radius: 8px; padding: 15px; margin-top: 25px; border-left: 4px solid #6366f1; }
"""

with gr.Blocks(title="RegaLabs-TTS: Sorani Speech Synthesis", css=custom_css) as demo:
    gr.Markdown(
        """
        # 🎙️ RegaLabs-TTS: Central Kurdish (Sorani) Speech Synthesis
        **CosyVoice 3 Zero-Shot TTS & Voice Cloning Adaptation for Sorani Kurdish (سۆرانی)**
        Developed by **[RegaLabs](https://huggingface.co/RegaLabs)**
        """,
        elem_classes=["header"]
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Sorani Text (دەقی سۆرانی)",
                placeholder="سڵاو، بەخێربێن بۆ تاقیکردنەوەی دەنگی RegaLabs-TTS...",
                value="سڵاو، بەخێربێن بۆ تاقیکردنەوەی دەنگی RegaLabs-TTS",
                lines=4,
            )
            
            prompt_audio = gr.Audio(
                label="Reference Voice Audio (دەنگی نموونە)",
                type="filepath",
                value=sample_wav,
            )
            
            prompt_text_input = gr.Textbox(
                label="Reference Voice Transcript (دەقی دەنگی نموونەکە)",
                placeholder="ئەمە دەنگی نموونەیە",
                value="ئەمە دەنگی نموونەیە",
                lines=2,
            )
            
            with gr.Accordion("Advanced Options (ڕێکخستنەکان)", open=False):
                speed_slider = gr.Slider(
                    minimum=0.5, maximum=1.5, value=1.0, step=0.05, label="Speech Speed (خێرایی خوێندنەوە)"
                )
                normalize_check = gr.Checkbox(
                    value=True, label="Enable Sorani Text Normalization (ڕێکخستنی دەق)"
                )
                
            generate_btn = gr.Button("🎤 Synthesize Sorani Speech (دروستکردنی دەنگ)", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Synthesized Sorani Speech (دەنگی دروستکراو)", type="filepath")
            status_output = gr.Textbox(label="Status & Info", interactive=False)

    generate_btn.click(
        fn=synthesize_sorani,
        inputs=[text_input, prompt_audio, prompt_text_input, speed_slider, normalize_check],
        outputs=[audio_output, status_output],
    )
    
    gr.Markdown(
        """
        ---
        ### 📜 License & Mandatory Attribution
        * **Model Checkpoint & Codebase:** **Apache 2.0** by **RegaLabs**. Commercial and non-commercial use is **fully allowed** with mandatory credit to RegaLabs.
        * **Stock Prompt Voices & Audio Samples:** **Non-Commercial Use Only**. Pre-packaged stock voice samples are strictly restricted from commercial use/cloning.
        
        **📌 Mandatory Credit:** Any public use, generated media (videos, podcasts, voiceovers, radio/TV), applications, or derivative works MUST explicitly credit **RegaLabs**:
        > *"Voice synthesized using RegaLabs-TTS by RegaLabs"* (or *"Audio powered by RegaLabs"*).
        """,
        elem_classes=["attribution"]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
