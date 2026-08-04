#!/usr/bin/env python3
"""Conservative Central Kurdish (Sorani) text normalization.

This frontend deliberately does not apply contextual ``ه`` -> ``ھ`` guesses.
Both characters occur in Kurdish text, and changing one without a lexicon can
alter the model's learned pronunciation. User-confirmed spelling is preserved
unless it is an unambiguous Arabic/Persian codepoint variant.
"""

from __future__ import annotations

import re
import unicodedata


_CHARACTER_MAP = str.maketrans({
    "ك": "ک",  # Arabic kaf -> Kurdish kaf
    "ي": "ی",  # Arabic yeh -> Kurdish yeh
    "ى": "ی",  # alef maksura -> Kurdish yeh
    "ة": "ە",  # teh marbuta -> Kurdish ae
})

_DIGIT_MAP = str.maketrans({
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})

_DIGIT_WORDS = {
    "0": "سفر", "1": "یەک", "2": "دوو", "3": "سێ", "4": "چوار",
    "5": "پێنج", "6": "شەش", "7": "حەوت", "8": "هەشت", "9": "نۆ",
}
_UNDER_20 = {
    0: "سفر", 1: "یەک", 2: "دوو", 3: "سێ", 4: "چوار", 5: "پێنج",
    6: "شەش", 7: "حەوت", 8: "هەشت", 9: "نۆ", 10: "دە", 11: "یازدە",
    12: "دوازدە", 13: "سێزدە", 14: "چواردە", 15: "پازدە", 16: "شازدە",
    17: "حەڤدە", 18: "هەژدە", 19: "نۆزدە",
}
_TENS = {20: "بیست", 30: "سی", 40: "چل", 50: "پەنجا", 60: "شەست",
         70: "حەفتا", 80: "هەشتا", 90: "نەوەد"}
_HUNDREDS = {
    100: "سەد", 200: "دووسەد", 300: "سێسەد", 400: "چوارسەد",
    500: "پێنجسەد", 600: "شەشسەد", 700: "حەوتسەد", 800: "هەشتسەد",
    900: "نۆسەد",
}


def number_to_sorani(number: int) -> str:
    """Spell ordinary integers without changing the spelling of input words."""
    if number < 0:
        return "نێگەتیڤ و " + number_to_sorani(-number)
    if number < 20:
        return _UNDER_20[number]
    if number < 100:
        tens, units = divmod(number, 10)
        return _TENS[tens * 10] + (f" و {_UNDER_20[units]}" if units else "")
    if number < 1000:
        hundreds, remainder = divmod(number, 100)
        result = _HUNDREDS[hundreds * 100]
        return result + (f" و {number_to_sorani(remainder)}" if remainder else "")
    if number < 1_000_000:
        thousands, remainder = divmod(number, 1000)
        result = ("هەزار" if thousands == 1 else f"{number_to_sorani(thousands)} هەزار")
        return result + (f" و {number_to_sorani(remainder)}" if remainder else "")
    if number < 1_000_000_000:
        millions, remainder = divmod(number, 1_000_000)
        result = f"{number_to_sorani(millions)} ملیۆن"
        return result + (f" و {number_to_sorani(remainder)}" if remainder else "")
    billions, remainder = divmod(number, 1_000_000_000)
    result = f"{number_to_sorani(billions)} ملیار"
    return result + (f" و {number_to_sorani(remainder)}" if remainder else "")


def _replace_number(match: re.Match[str]) -> str:
    value = match.group(0)
    digits = value.lstrip("-")
    if len(digits) >= 10 or (len(digits) > 1 and digits.startswith("0")):
        words = " ".join(_DIGIT_WORDS[digit] for digit in digits)
        return f"نێگەتیڤ {words}" if value.startswith("-") else words
    return number_to_sorani(int(value))


def expand_numbers(text: str) -> str:
    text = text.translate(_DIGIT_MAP)
    text = re.sub(r"(?<=\d)[,،](?=\d{3}(?!\d))", "", text)
    text = re.sub(r"(?<=\d)[,،](?=\d)", ".", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*[%٪]", r"\1 لەسەد", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*\$", r"\1 دۆلار", text)

    def replace_decimal(match: re.Match[str]) -> str:
        integer, fraction = match.group(0).split(".", 1)
        digits = " ".join(_DIGIT_WORDS[digit] for digit in fraction)
        return f"{number_to_sorani(int(integer))} پۆینت {digits}"

    text = re.sub(r"(?<!\w)-?\d+\.\d+", replace_decimal, text)
    return re.sub(r"(?<!\w)-?\d+", _replace_number, text)


def normalize_sorani_text(text: str) -> str:
    """Normalize safe orthographic variants while preserving Kurdish spelling."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).translate(_CHARACTER_MAP)
    text = re.sub(r"[\u064B-\u065F]", "", text)
    text = expand_numbers(text)
    text = text.replace("؟", "؟").replace("?", "؟")
    text = text.replace(";", "؛").replace(",", "،")
    return re.sub(r"\s+", " ", text).strip()


# The dataset/evaluation tooling expects this shared normalizer name.
normalize_kurdish_text = normalize_sorani_text


if __name__ == "__main__":
    assert normalize_sorani_text("هێشتا") == "هێشتا"
    assert normalize_sorani_text("ك ى ة") == "ک ی ە"
    assert normalize_sorani_text("2026") == "دوو هەزار و بیست و شەش"
    print("conservative Sorani frontend self-check passed")
