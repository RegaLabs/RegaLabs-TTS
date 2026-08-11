# RegaLabs-TTS: CosyVoice 3 Sorani Flow Adaptation

**RegaLabs-TTS** is a high-quality Central Kurdish (Sorani / سۆرانی) text-to-speech system developed by **RegaLabs** based on **CosyVoice 3**.

* **GitHub Repository:** [`RegaLabs/RegaLabs-TTS`](https://github.com/RegaLabs/RegaLabs-TTS)
* **Hugging Face Model:** [`RegaLabs/RegaLabs-TTS`](https://huggingface.co/RegaLabs/RegaLabs-TTS)

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

# Run RegaLabs-TTS synthesis
python run_regalabs_tts.py \
  --base-model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --flow-checkpoint /path/to/cosyvoice3_sorani_flow_best_step2300.pt \
  --prompt-wav /path/to/reference.wav \
  --prompt-text "دەقی نموونەی دەنگەکە" \
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
- `sorani/infer_lora.py` — Zero-shot inference runner for loading flow checkpoints and optional LLM adapters.
- `sorani/batch_infer.py` — Batch regression and dataset testing runner.
- `run_regalabs_tts.py` — Standalone execution runner.
- `app.py` — Gradio Live Web Demo UI.
- `setup.py` / `pyproject.toml` — Python packaging scripts.

---

## 📜 License & Attribution Requirement

Licensed under **Apache 2.0** by **RegaLabs**. Base runtime and original model architecture belong to the upstream CosyVoice project.

### 📌 Credit & Attribution Guidelines
Any public use, generated media (videos, podcasts, audiobooks, broadcasts, or AI services), software, or derivative works utilizing **RegaLabs-TTS** MUST explicitly provide visible credit to **RegaLabs**:
* **For Videos & Audio Content:** Include in video description/credits: *"Voice synthesized using RegaLabs-TTS by RegaLabs"* (or *"Audio powered by RegaLabs"*).
* **For Applications & Services:** Include attribution in application credits or about section.
