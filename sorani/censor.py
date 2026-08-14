#!/usr/bin/env python3
"""Fail-closed Sorani (Central Kurdish) sexual-word censoring for RegaLabs-TTS.

Why a text filter and not the checkpoint
----------------------------------------
CosyVoice3 is a neural TTS: it pronounces whatever text tokens it receives
(text_frontend=False routes raw text straight into the model). The flow
checkpoint contains learned weights with no word-refusal mechanism, so
censorship can only live in the text pipeline. This module is that pipeline
stage, and it is deliberately fail-closed:

* Removing or editing the word list raises ``CensorIntegrityError`` and
  synthesis refuses to run (the model "breaks itself" instead of speaking
  uncensored).
* Loading any flow checkpoint other than the official RegaLabs-TTS Sorani
  checkpoint raises ``CensorIntegrityError`` and refuses to run, so swapping
  the model file is detected too.

Caveat (honest limitation): the anchor lives inside this repository, so a
determined attacker with full access to the code can always patch the checks
out. No locally shipped software can do better without an external signer.
What this guarantees is that *accidental or naive* removal, word-list edits,
or checkpoint swaps fail loudly instead of silently producing uncensored
audio.

Word sources (public dictionaries, checked 2026-08-15)
------------------------------------------------------
    کێر  penis  https://ckb.wiktionary.org/wiki/کێر   (ئەندامی نێرینە)
    کیر  penis  https://ku.wiktionary.org/wiki/kîr    (kurdî-erebî: کیر; soranî: kêr, kîr)
    قوز  vulva  https://ku.wiktionary.org/wiki/quz    (etymology: hevreha soranî قوز (quz))
    کوز  vulva  https://ku.wiktionary.org/wiki/quz    (etymology: hevreha soranî کوز (kuz))
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
from pathlib import Path

# --------------------------------------------------------------------------
# Word list
# --------------------------------------------------------------------------
# Every root below is a fully normalized spelling (see _CHARACTER_MAP).
# Arabic kaf/yeh variants (ك، ي، ى، ة) are converted before matching, so
# كێر / يزنی-style spellings are caught automatically.
SORANI_SEX_TERMS: tuple[str, ...] = (
    "کێر",    # penis  (ckb.wiktionary.org/wiki/کێر)
    "کیر",    # penis  (ku.wiktionary.org/wiki/kîr)
    "قوز",    # vulva  (ku.wiktionary.org/wiki/quz, "hevreha soranî قوز")
    "قووز",   # spelling variant of قوز
    "کوز",    # vulva  (ku.wiktionary.org/wiki/quz, "hevreha soranî کوز")
    "کووز",   # spelling variant of کوز
)

# What censored words are replaced with (a bleep: long pause in TTS).
# If CosyVoice pronounces the dots aloud, set this to "" (delete) or a
# neutral syllable.
CENSOR_BLEEP = "......"

# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
# Sorani nominal inflections that may follow a root: definite suffixes
# (ەکە، ەکان، ەکەی، ...), possessive suffixes (م، ت، ن، مان، تان، یان,
# یم، یت، ین، ...), ezafe (ی، ێ), indefinite (ێک، ێکی، ...) and the
# demonstrative (ەوە، ...). Repeated so compounds like کیرەکانم match.
_SUFFIX_RE = (
    r"(?:ەکەی|ەکەم|ەکەت|ەکەمان|ەکەتان|ەکانیان|ەکەکان|"
    r"ەوەمان|ەوەتان|ەوەیان|ەوەم|ەوەت|ەوەن|ەوە|"
    r"یمان|یتان|ییان|یم|یت|ین|"
    r"ەکە|ەکان|ەکەی|ەکانی|ەکانم|ەکانت|"
    r"مان|تان|یان|"
    r"ێکە|ێکان|ێکی|ێک|"
    r"ە|ێ|ی|و|ن|م|ت"
    r")*"
)

_CHARACTER_MAP = str.maketrans({
    "ك": "ک",  # Arabic kaf -> Kurdish kaf
    "ي": "ی",  # Arabic yeh -> Kurdish yeh
    "ى": "ی",  # alef maksura -> Kurdish yeh
    "ة": "ە",  # teh marbuta -> Kurdish ae
})

_WORD_BOUNDARY = r"(?<![\w\u200C]){root}(?:\u200C)?{suffix}(?![\w\u200C])"
_CENSOR_PATTERN = re.compile(
    "|".join(
        _WORD_BOUNDARY.format(root=re.escape(root), suffix=_SUFFIX_RE)
        for root in SORANI_SEX_TERMS
    )
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_CHARACTER_MAP)
    return re.sub(r"[\u064B-\u065F]", "", text)


def censor_text(text: str) -> str:
    """Replace every blocked Sorani word (and its inflections) with a bleep."""
    if not text:
        return text
    return _CENSOR_PATTERN.sub(CENSOR_BLEEP, _normalize(text))


# --------------------------------------------------------------------------
# Integrity (fail-closed)
# --------------------------------------------------------------------------
class CensorIntegrityError(RuntimeError):
    """Raised when the censorship filter or the model checkpoint is tampered."""


def _wordlist_digest() -> str:
    return hashlib.sha256(repr(SORANI_SEX_TERMS).encode("utf-8")).hexdigest()


# SHA-256 of the official RegaLabs-TTS Sorani flow checkpoint
# (cosyvoice3_sorani_flow_best_step2300.pt).
EXPECTED_FLOW_SHA256 = "033abd6fcb88c8069a24ac7215dfaac92b2526ee687a05dc0e327693a1cea75c"

# Digest of SORANI_SEX_TERMS above. Regenerate with: python -m sorani.censor --rehash
_WORDLIST_SHA256 = "25ca7c6b6f724b0f5a68886a6526410e1dda1d42de18cbd3fd927465363a0263"


def verify_wordlist() -> None:
    """Fail closed if the word list was edited after signing."""
    if _wordlist_digest() != _WORDLIST_SHA256:
        raise CensorIntegrityError(
            "The Sorani censorship word list has been modified. RegaLabs-TTS "
            "refuses to synthesize with a tampered filter. Restore "
            "sorani/censor.py from the official repository, or re-sign it "
            "with: python -m sorani.censor --rehash"
        )


def verify_checkpoint(path: str | Path) -> None:
    """Fail closed if the flow checkpoint is not the official RegaLabs file."""
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise CensorIntegrityError(
            f"Flow checkpoint not found: {checkpoint}. RegaLabs-TTS only runs "
            "with the official Sorani checkpoint."
        )
    hasher = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != EXPECTED_FLOW_SHA256:
        raise CensorIntegrityError(
            f"Flow checkpoint {checkpoint} does not match the official "
            "RegaLabs-TTS Sorani checkpoint (SHA-256 mismatch). RegaLabs-TTS "
            "refuses to run with a swapped model. If you intentionally "
            "released a new checkpoint, re-sign it with: "
            "python -m sorani.censor --sign-checkpoint PATH"
        )


# --------------------------------------------------------------------------
# Tooling
# --------------------------------------------------------------------------
def _embed_digest(module_path: Path, digest: str) -> None:
    source = module_path.read_text(encoding="utf-8")
    pattern = re.compile(r'^_WORDLIST_SHA256 = "[a-f0-9]*"$', re.MULTILINE)
    updated, count = pattern.subn(f'_WORDLIST_SHA256 = "{digest}"', source)
    if count != 1:
        raise RuntimeError("Could not locate _WORDLIST_SHA256 in the module.")
    module_path.write_text(updated, encoding="utf-8")


def _sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="RegaLabs-TTS Sorani censoring tools")
    parser.add_argument("--rehash", action="store_true",
                        help="Re-sign the current word list into this file")
    parser.add_argument("--sign-checkpoint", metavar="PATH",
                        help="Embed the SHA-256 of a new official checkpoint")
    parser.add_argument("--hash-checkpoint", metavar="PATH",
                        help="Print the SHA-256 of a checkpoint file")
    parser.add_argument("--self-test", action="store_true", help="Run self-tests")
    args = parser.parse_args()

    if args.hash_checkpoint:
        print(_sha256_of(Path(args.hash_checkpoint)))
        return 0

    if args.sign_checkpoint:
        digest = _sha256_of(Path(args.sign_checkpoint))
        source = Path(__file__).read_text(encoding="utf-8")
        pattern = re.compile(r'^EXPECTED_FLOW_SHA256 = "[a-f0-9]*"$', re.MULTILINE)
        updated, count = pattern.subn(f'EXPECTED_FLOW_SHA256 = "{digest}"', source)
        if count != 1:
            raise RuntimeError("Could not locate EXPECTED_FLOW_SHA256.")
        Path(__file__).write_text(updated, encoding="utf-8")
        print(f"Signed checkpoint {digest}")
        return 0

    if args.rehash:
        _embed_digest(Path(__file__), _wordlist_digest())
        print(f"Re-signed word list ({_wordlist_digest()})")
        return 0

    if args.self_test:
        assert censor_text("کێر باشە") == "...... باشە"
        assert censor_text("کیرەکە") == "......"
        assert censor_text("کیریم و قوز") == "...... و ......"
        assert censor_text("كێر") == "......"  # Arabic kaf -> Kurdish kaf
        assert censor_text("کێرد") == "کێرد"  # knife (kêrd), not a match
        assert censor_text("کۆنە") == "کۆنە"   # old, not a match
        assert censor_text("کیرەکانی") == "......"
        assert censor_text("قوزەکە") == "......"
        assert censor_text("کوزەکانی") == "......"
        verify_wordlist()
        original = SORANI_SEX_TERMS
        try:
            globals()["SORANI_SEX_TERMS"] = original + ("تاقیکردنەوە",)
            try:
                verify_wordlist()
                raise AssertionError("tampered list was not detected")
            except CensorIntegrityError:
                pass
        finally:
            globals()["SORANI_SEX_TERMS"] = original
        print("censor self-test passed")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
