import re
import json
import unicodedata
from num2words import num2words

# Load abbreviations
with open("apps/utils/abbreviations.json", "r", encoding="utf-8") as f:
    ABBREVIATIONS = json.load(f)

def normalize_text(text: str) -> str:
    """Normalize Unicode text and remove extra spaces."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()

def expand_abbreviations(text: str) -> str:
    """Expand abbreviations while ensuring valid replacements."""
    def replace(match):
        word = match.group(0).lower()
        return ABBREVIATIONS.get(word, match.group(0))

    # Expand standalone abbreviations (word-boundary sensitive)
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in ABBREVIATIONS.keys()) + r')\b', re.IGNORECASE)
    text = pattern.sub(replace, text)

    # Expand unit-based abbreviations (e.g., "5kg" → "5 kilogram")
    unit_pattern = re.compile(r'\b(\d+)\s*(' + '|'.join(re.escape(k) for k in ABBREVIATIONS.keys()) + r')\b', re.IGNORECASE)
    text = unit_pattern.sub(lambda m: f"{m.group(1)} {ABBREVIATIONS[m.group(2).lower()]}", text)

    return text

def convert_currency_to_words(text: str) -> str:
    """Convert currency values (VNĐ) to Vietnamese words."""
    
    def replace_currency(match):
        number = int(match.group(1).replace('.', ''))
        print(f"[DEBUG] replace_currency, after remove `.`: {number}")
        return f"{num2words(number, lang='vi')} Việt Nam đồng"

    return re.sub(r'(\d+(?:\.\d+)*)\s*VNĐ', replace_currency, text, flags=re.IGNORECASE)

def convert_unit_price_to_words(text: str) -> str:
    """Convert unit price per liter (VNĐ/lít) to Vietnamese words."""
    
    def replace_unit_price(match):
        number = int(match.group(1).replace('.', ''))
        print(f"[DEBUG] replace_unit_price, after remove `.`: {number}")
        return f"{num2words(number, lang='vi')} Việt Nam đồng một lít"

    return re.sub(r'(\d+(?:\.\d+)*)\s*VNĐ\s*/\s*lít', replace_unit_price, text, flags=re.IGNORECASE)

def convert_weight_to_words(text: str) -> str:
    """Convert weight values (kg) to Vietnamese words."""
    
    def replace_weight(match):
        number = int(match.group(1))
        return f"{num2words(number, lang='vi')} kilogram"

    return re.sub(r'(\d+)\s*kg', replace_weight, text, flags=re.IGNORECASE)

def convert_general_numbers_to_words(text: str) -> str:
    """Convert standalone numbers into Vietnamese words."""
    
    def replace_general_number(match):
        return num2words(int(match.group(0)), lang="vi")

    return re.sub(r'\b\d+\b', replace_general_number, text)

def convert_number_to_words(text: str) -> str:
    """Convert all numbers, including standalone, currency, and unit-based cases, into Vietnamese words."""
    
    print(f"[DEBUG] Input Text: {text}")
    
    text = convert_unit_price_to_words(text)      # Convert "22.500 VNĐ/lít"
    text = convert_currency_to_words(text)        # Convert "22.500 VNĐ"
    text = convert_weight_to_words(text)          # Convert "5kg"
    text = convert_general_numbers_to_words(text) # Convert general numbers

    print(f"[DEBUG] Final Output: {text}")
    
    return text

def convert_date_to_words(text: str) -> str:
    """Convert full and partial date formats into written Vietnamese words, ensuring correct formatting."""
    
    def format_vietnamese_date(day, month, year=None, prefix: str = ''):
        """Helper function to convert numbers to Vietnamese words."""
        day_str = num2words(int(day), lang="vi")
        month_str = "tư" if int(month) == 4 else num2words(int(month), lang="vi")
        
        if year:  # Handle full date (DD/MM/YYYY or YYYY-MM-DD)
            year_str = num2words(int(year), lang="vi")
            year_str = re.sub(r"\blẻ\b", "không trăm", year_str)  # Fix 'lẻ' in years
            return f"ngày {day_str} tháng {month_str} năm {year_str}"
        else:  # Handle partial date (DD/MM)
            return f"{prefix} {day_str} tháng {month_str}"

    # Convert full dates: DD/MM/YYYY → "ngày hai mươi ba tháng ba năm hai ngàn không trăm hai mươi lăm"
    text = re.sub(
        r'(\d{1,2})/(\d{1,2})/(\d{4})',
        lambda m: format_vietnamese_date(m.group(1), m.group(2), m.group(3)),
        text
    )

    # Convert full dates: YYYY-MM-DD → "ngày ba mươi tháng tư năm hai ngàn không trăm hai mươi lăm"
    text = re.sub(
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
        lambda m: format_vietnamese_date(m.group(3), m.group(2), m.group(1)),
        text
    )

    # Convert **partial dates only** (e.g., "ngày 21/3" → "ngày hai mươi mốt tháng ba"),  
    # but **skip full dates with year** (e.g., "ngày 21/3/2025" remains unchanged).
    text = re.sub(
        r'(?i)\b(ngày|trước|sau|tới|đến)\s*(\d{1,2})/(\d{1,2})\b(?!/\d{4})',
        lambda m: format_vietnamese_date(m.group(2), m.group(3), None, m.group(1)),
        text
    )

    # 🔥 **Fix repeated "ngày ngày" issue (case insensitive)**
    text = re.sub(r'\b(Ngày ngày|ngày ngày)\b', "ngày", text, flags=re.IGNORECASE)

    return text

def convert_time_to_words(text: str) -> str:
    """Convert time formats into Vietnamese words like 'lúc mười bốn giờ ba mươi phút'."""
    def format_vietnamese_time(hour, minute):
        """Format time in Vietnamese words."""
        hour_str = num2words(int(hour), lang="vi")
        minute_str = num2words(int(minute), lang="vi") if int(minute) > 0 else ""

        if minute_str:
            return f"lúc {hour_str} giờ {minute_str} phút"
        return f"lúc {hour_str} giờ"

    # Convert HH:MM format
    text = re.sub(r'(\d{1,2}):(\d{2})', lambda m: format_vietnamese_time(m.group(1), m.group(2)), text)
    
    # 🔥 **Fix: Remove repeated "ngày" (case insensitive)**
    text = re.sub(r'\b(Lúc lúc|lúc lúc)\b', "lúc", text, flags=re.IGNORECASE)

    return text

def process_text_for_tts(text: str) -> str:
    """Process text for TTS: Expand abbreviations, convert numbers, dates, and times into written forms."""
    
    print(f"[DEBUG] Original text: {text}")
    
    # replace <new_line> as '.'
    text = re.sub(r'\.+', '.', re.sub(r'\n+', '. ', text))
    
    text = normalize_text(text)
    print(f"[DEBUG] After normalize_text: {text}")
    
    text = convert_date_to_words(text)
    print(f"[DEBUG] After convert_date_to_words: {text}")
    
    text = convert_time_to_words(text)
    print(f"[DEBUG] After convert_time_to_words: {text}")
    
    text = convert_number_to_words(text)
    print(f"[DEBUG] After convert_number_to_words: {text}")
    
    text = expand_abbreviations(text)
    print(f"[DEBUG] After expand_abbreviations: {text}")
    
    print(f"[DEBUG] Final processed text: {text}")
    
    return text

