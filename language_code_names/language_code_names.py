languages = {
    "en": "English",
    "en-US": "English (United States)",
    "en-GB": "English (United Kingdom)",
    "en-AU": "English (Australia)",
    "en-CA": "English (Canada)",
    "en-IN": "English (India)",
    "en-NZ": "English (New Zealand)",
    "en-SG": "English (Singapore)",
    "en-ZA": "English (South Africa)",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "pt": "Portuguese",
    "it": "Italian",
    "ko": "Korean",
    "tr": "Turkish",
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "uk": "Ukrainian",
    "he": "Hebrew",
    "el": "Greek",
    "cs": "Czech",
    "hu": "Hungarian",
    "ro": "Romanian",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "bg": "Bulgarian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "hr": "Croatian",
    "sr": "Serbian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "fa": "Persian",
    "ur": "Urdu",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "gu": "Gujarati",
    "mr": "Marathi",
    "sw": "Swahili",
    "af": "Afrikaans",
    "is": "Icelandic",
    "mt": "Maltese",
    "ga": "Irish",
    "cy": "Welsh",
    "eu": "Basque",
    "ca": "Catalan",
    "gl": "Galician",
    "hy": "Armenian",
    "ka": "Georgian",
    "kk": "Kazakh",
    "uz": "Uzbek",
    "az": "Azerbaijani",
    "mn": "Mongolian",
    "my": "Burmese",
    "km": "Khmer",
    "lo": "Lao",
    "si": "Sinhala",
    "am": "Amharic",
    "yo": "Yoruba",
    "ha": "Hausa",
    "zu": "Zulu",
    "xh": "Xhosa",
    "bs": "Bosnian",
    "mk": "Macedonian",
    "sq": "Albanian",
    "sr-Latn": "Serbian (Latin)",
    "pl": "Polish",
    "hr": "Croatian",
    "sl": "Slovenian",
    "cy": "Welsh",
    "eu": "Basque",
    "ba": "Bashkir",
    "be": "Belarusian",
    "tt": "Tatar",
    "gl": "Galician",
    "la": "Latin",
    "mr": "Marathi",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "si": "Sinhala",
    "ca": "Catalan",
    "cy": "Welsh",
    "sq": "Albanian",
    "ky": "Kyrgyz",
    "mk": "Macedonian",
    "be": "Belarusian",
    "mo": "Moldavian"
}


def get_language(code: str) -> str:
    return languages.get(code, "Language not found")


def(main):
    code = event.get("Language")
    get_language(code)


if __name__ == "__main__":
    main()