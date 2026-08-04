# CosyVoice Sorani flow adaptation tools

Private release of the reviewed Sorani frontend and inference helpers for a
CosyVoice 3 acoustic/flow adaptation.

The matching flow checkpoint is stored separately in the private Hugging Face
repository [`requite/cosyvoice-sorani-flow`](https://huggingface.co/requite/cosyvoice-sorani-flow).
This GitHub repository intentionally contains no datasets, manifests,
reference recordings, generated WAVs, evaluation outputs, or model weights.

## Contents

- `sorani/frontend.py` — conservative Central Kurdish text normalization.
- `sorani/infer_lora.py` — CosyVoice 3 inference helper that can load a flow
  checkpoint and an optional matching LLM adapter.
- `sorani/batch_infer.py` — batch regression helper.

The current private release contains the full flow checkpoint but no LLM LoRA
adapter. It is therefore a flow component release, not a standalone Sorani
CosyVoice package. Use it with the matching CosyVoice 3 base model and any
authorized adapter checkpoint.

The upstream CosyVoice project and its license remain the source of truth for
the base model and runtime.
