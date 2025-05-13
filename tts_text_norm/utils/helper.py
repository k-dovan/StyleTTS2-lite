import re
from typing import List
from tts_text_norm.constants import charset, regular


def get_separator(string: str) -> str:
    if "/" in string:
        separator = "/"
    elif "-" in string:
        separator = "-"
    elif "." in string:
        separator = "."
    else:
        raise TypeError(f"Unidentify separator, given string: {string}")

    return separator


def split_num_word(token: str) -> str:
    token = re.sub(
        r"(?P<id>(\d[{}]))".format(charset.VietnameseCharset.CHARSETS),
        lambda x: x.group("id")[0] + " " + x.group("id")[1],
        token,
    )
    token = re.sub(
        r"(?P<id>([{}]\d))".format(charset.VietnameseCharset.CHARSETS),
        lambda x: x.group("id")[0] + " " + x.group("id")[1],
        token,
    )

    token = re.sub(
        r"(?P<id>(\d\D))", lambda x: x.group("id")[0] + " " + x.group("id")[1], token
    )
    token = re.sub(
        r"(?P<id>(\d\D))".format(charset.VietnameseCharset.CHARSETS),
        lambda x: x.group("id")[0] + " " + x.group("id")[1],
        token,
    )

    return token


def split_punc_char(token: str) -> str:
    # tách dấu đi liền với từ
    token = re.sub(
        r"(?P<id>[{}])(?P<id1>{})(?P<id2>[{}])".format(
            charset.VietnameseCharset.CHARSETS,
            regular.PunctuationRegular.PUNCTUATION,
            charset.VietnameseCharset.CHARSETS,
        ),
        lambda x: x.group("id") + " " + x.group("id1") + " " + x.group("id2"),
        token,
    )
    token = re.sub(
        r"(?P<id>[{}])(?P<id1>{})(?P<id2>[{}])".format(
            charset.VietnameseCharset.CHARSETS,
            regular.PunctuationRegular.PUNCTUATION,
            charset.VietnameseCharset.CHARSETS,
        ),
        lambda x: x.group("id") + " " + x.group("id1") + " " + x.group("id2"),
        token,
    )

    token = re.sub(
        r"(?P<id>[{}])(?P<id1>{})(?P<id2>\d)".format(
            "-", charset.VietnameseCharset.CHARSETS
        ),
        lambda x: x.group("id") + " " + x.group("id2"),
        token,
    )
    token = re.sub(
        r"(?P<id>[{}])(?P<id1>{})(?P<id2>\d)".format(
            charset.VietnameseCharset.CHARSETS, "-"
        ),
        lambda x: x.group("id") + " " + x.group("id2"),
        token,
    )

    token = re.sub(
        r"(?P<id>\d)(?P<id1>{})(?P<id2>{})".format(
            regular.PunctuationRegular.PUNCTUATION, charset.VietnameseCharset.CHARSETS
        ),
        lambda x: x.group("id") + " " + x.group("id1") + " " + x.group("id2"),
        token,
    )
    token = re.sub(
        r"(?P<id>{})(?P<id1>{})(?P<id2>\d)".format(
            charset.VietnameseCharset.CHARSETS,
            regular.PunctuationRegular.PUNCTUATION,
        ),
        lambda x: x.group("id") + " " + x.group("id1") + " " + x.group("id2"),
        token,
    )

    token = re.sub(
        r"(?P<id>{})(?P<id1>{})".format(
            regular.PunctuationRegular.PUNCTUATION,
            charset.VietnameseCharset.CHARSETS,
        ),
        lambda x: x.group("id") + " " + x.group("id1"),
        token,
    )
    token = re.sub(
        r"(?P<id>{})(?P<id1>{})".format(
            charset.VietnameseCharset.CHARSETS,
            regular.PunctuationRegular.PUNCTUATION,
        ),
        lambda x: x.group("id") + " " + x.group("id1"),
        token,
    )

    return token


def is_number(istring: str, lst_sep: List = [",", "."]) -> bool:

    return all(ch.isdigit() or ch in lst_sep for ch in istring)


def is_short_name(istring: str):

    return istring.isupper() and all(
        len(ch) == 1 for ch in istring.replace(".", " ").split()
    )
    
def split_text_for_inference(text: str, max_length=200) -> list:
    """Splits text into chunks based on sentence boundaries to fit within model constraints,
    and further by commas if needed. Preserves sentence-ending punctuation in final output."""
    
    print (f'input text for splitting: {text}')
    
    # First split by sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Splits after ., !, or ? followed by space
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Refine any over-length chunks by splitting at commas
    refined_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            refined_chunks.append(chunk)
        else:
            subparts = re.split(r'\s+,\s+', chunk)
            temp = ""
            for part in subparts:
                if len(temp) + len(part) + 2 <= max_length:  # +2 for ", "
                    temp += part + " , "
                else:
                    refined_chunks.append(temp.strip(", "))
                    temp = part + " , "
            if temp:
                refined_chunks.append(temp.strip(", "))

    refined_chunks = [(re.sub(r'[,.!?]$', '', re.sub(r'(\s*,\s*)+', ' , ', chunk))).strip() for chunk in refined_chunks if len(chunk) > 0]

    print(f"[DEBUG] Split into {len(refined_chunks)} Text Chunks: {refined_chunks}")
    return refined_chunks
