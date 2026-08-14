![RegaLabs-TTS Banner](https://raw.githubusercontent.com/RegaLabs/RegaLabs-TTS/main/assets/banner.jpg)

# RegaLabs-TTS: CosyVoice 3 Sorani Flow Adaptation

**RegaLabs-TTS** is a high-quality Central Kurdish (Sorani / سۆرانی) text-to-speech system developed by **RegaLabs** based on **CosyVoice 3**.

* **GitHub Repository:** [`RegaLabs/RegaLabs-TTS`](https://github.com/RegaLabs/RegaLabs-TTS)
* **Hugging Face Model:** [`RegaLabs/RegaLabs-TTS`](https://huggingface.co/RegaLabs/RegaLabs-TTS)

---

![RegaLabs-TTS Features](https://raw.githubusercontent.com/RegaLabs/RegaLabs-TTS/main/assets/features.jpg)

---

## 🎧 Audio Example

Listen to a generated Sorani audio sample:

<audio controls src="https://raw.githubusercontent.com/RegaLabs/RegaLabs-TTS/main/samples/aran_en021.wav"></audio>

[🔊 Download Sample Audio WAV](https://raw.githubusercontent.com/RegaLabs/RegaLabs-TTS/main/samples/aran_en021.wav)

---

## 📊 Dataset & Voice Cloning Performance

* **Training Dataset:** Trained on **53 hours** of Sorani Kurdish speech data.
  * **Male Dataset:** ~35–40 hours of speech.
  * **Female Dataset:** ~13–18 hours of speech.
* **Cloning Performance:**
  * **Male Voices:** Tested and verified — clones male voices **flawlessly** with high similarity and naturalness.
  * **Female Voices:** Untested / not yet evaluated for female zero-shot cloning.

---

## 💻 Installation & Quick Usage

### 1. Install via `pip`

```bash
pip install git+https://github.com/RegaLabs/RegaLabs-TTS.git
```

### 2. Standalone Synthesis Command

```bash
# Clone base engine
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
export COSYVOICE_REPO=$(pwd)/CosyVoice

# Download the model artifacts from Hugging Face
git clone https://huggingface.co/RegaLabs/RegaLabs-TTS regalabs-tts-weights

# Run RegaLabs-TTS synthesis (adapter required for Sorani speech)
python run_regalabs_tts.py \
  --base-model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --flow-checkpoint regalabs-tts-weights/cosyvoice3_sorani_flow_best_step2300.pt \
  --adapter regalabs-tts-weights/cosyvoice3_sorani_lora_refined_best.pt \
  --prompt-wav regalabs-tts-weights/samples/aran_en021.wav \
  --prompt-text "دەنگێکی لەسەرخۆ، هێمن و پڕ لە بڕوابەخۆبوون." \
  --text "سڵاو لە تەواوی كوردستان" \
  --out output_sorani.wav
```

### 3. Run Live Web UI (Gradio Demo)

```bash
python app.py
```

---

## 🛠️ Repository Contents

- `sorani/frontend.py` — Conservative Central Kurdish (Sorani) text normalization and number-to-words conversion.
- `sorani/censor.py` — Fail-closed Sorani sexual-word filter with integrity checks (see below).
- `sorani/infer_lora.py` — Zero-shot inference runner for loading flow checkpoints and optional LLM adapters.
- `sorani/batch_infer.py` — Batch regression and dataset testing runner.
- `run_regalabs_tts.py` — Standalone execution runner (requires the LLM adapter, downloaded from Hugging Face).
- `app.py` — Gradio Live Web Demo UI.
- `samples/aran_en021.wav` — Sample audio snippet.
- `assets/` — Project banners & graphics.
- `setup.py` / `pyproject.toml` — Python packaging scripts.

---

## 🚫 Content Filtering & Model Integrity

RegaLabs-TTS ships with a **fail-closed Sorani content filter** (`sorani/censor.py`). Sexual obscenities are bleeped (`......`) before synthesis instead of being spoken.

* **Blocked vocabulary:** a curated, dictionary-verified blocklist of Sorani sexual obscenities (the exact terms are defined in `sorani/censor.py`). Inflected forms and spelling variants are caught automatically.
* **Tamper protection:** the word list is SHA-256-signed, and the flow checkpoint must match the official SHA-256 (`033abd6f...`). If either is modified, synthesis refuses to run — the model "breaks itself" rather than speaking uncensored.
* **Re-signing (only for official model updates):** `python sorani/censor.py --rehash` re-signs the word list; `python sorani/censor.py --sign-checkpoint PATH` signs a newly released official checkpoint.
* **Honest limitation:** TTS censorship lives in the text layer (the weights themselves cannot refuse words), so a determined attacker with full code access can patch the checks out. This protects against accidental or naive removal and model swaps, not against deliberate reverse engineering.

---

---

## ⚠️ Responsible Use & No-Liability Disclaimer

RegaLabs-TTS is a **voice-cloning-capable** TTS system. It generates whatever text it is given, in whichever voice it is prompted with. RegaLabs:

* **is NOT responsible** for any content generated with this system, or for how it is used, modified, or redistributed — the user is solely responsible for the legality and consequences of their use;
* **prohibits** cloning the voice of any real person without that person's explicit consent, and prohibits use for impersonation, fraud, deepfakes, scams, defamation, harassment, sexual content involving minors, or any illegal activity (see Section 4 of the `LICENSE`);
* **warns** that all generated audio should be treated as potentially synthetic; never rely on model output as evidence of a real person's words.

**Voice consent rule:** zero-shot voice cloning is only permitted with the speaker's own approval, or for voices you own or are authorized to use.

---

## 📜 License & Mandatory Attribution

* **Model Checkpoint & Codebase:** Licensed under **Apache 2.0** by **RegaLabs**. Commercial and non-commercial use is **fully allowed**, provided mandatory credit for RegaLabs is included and the consent/prohibited-uses conditions in the LICENSE are respected. Base runtime and original model architecture belong to the upstream CosyVoice project.
* **Stock Voices & Audio Samples:** **Non-Commercial Use Only**. Pre-packaged stock prompt voice samples and demo audio files (including samples in `samples/`) are strictly restricted to non-commercial use and prohibited for commercial voice cloning/redistribution.

### 📌 Credit & Attribution Guidelines
Any public use, generated media (videos, podcasts, audiobooks, broadcasts, or AI services), software, or derivative works utilizing **RegaLabs-TTS** MUST explicitly provide visible credit to **RegaLabs**:
* **For Videos & Audio Content:** Include in video description/credits: *"Voice synthesized using RegaLabs-TTS by RegaLabs"* (or *"Audio powered by RegaLabs"*).
* **For Applications & Services:** Include attribution in application credits or about section.
