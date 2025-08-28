from enum import Enum, auto
from typing import Iterable
from pathlib import Path


HEADER_DICT_ADDR = 0x08
HEADER_ABBR_TABLE_ADDR = 0x18
NUM_ABBREVIATIONS = 96
ENTRIES_PER_ABBR_GROUP = 32


class Alphabet(Enum):
    LOWER = auto()
    UPPER = auto()
    OTHER = auto()


ALPHABETS = {
    Alphabet.LOWER: "abcdefghijklmnopqrstuvwxyz",
    Alphabet.UPPER: "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    Alphabet.OTHER: " \n0123456789.,!?_#'\"/\\-:()"
}

ZCHAR_MASK = 0x1F
ZCHAR_END_MASK = 0x8000

ZC_SPACE = 0
ZC_ABBR_START = 1
ZC_ABBR_END = 3
ZC_SHIFT1 = 4
ZC_SHIFT2 = 5
ZC_FIRST_NORMAL = 6


def read_word(data: bytes, offset: int) -> int:
    """Read a 16-bit big-endian word from story file."""
    return (data[offset] << 8) | data[offset + 1]


def load_abbreviations(data: bytes) -> list[str]:
    """Load the 96 abbreviation strings from the story file."""
    abbr_table_addr = read_word(data, HEADER_ABBR_TABLE_ADDR)
    abbreviations = [""] * NUM_ABBREVIATIONS
    if abbr_table_addr == 0:
        return abbreviations

    for i in range(NUM_ABBREVIATIONS):
        entry_addr = read_word(data, abbr_table_addr + 2 * i) * 2
        if entry_addr > 0:
            abbreviations[i] = decode_zstring(
                data, entry_addr, len(data), abbreviations)
    return abbreviations


def decode_zstring(
    data: bytes, offset: int, maxlen: int, abbreviations: list[str] | None = None
) -> str:
    """
    Decode a packed Z-encoded string.
    """
    zscii_chars: list[int] = []
    for pos in range(offset, offset + maxlen, 2):
        zword = (data[pos] << 8) | data[pos + 1]
        # unpack 3×5-bit z-chars
        for shift in (10, 5, 0):
            zscii_chars.append((zword >> shift) & ZCHAR_MASK)
        if zword & ZCHAR_END_MASK:
            break

    result_string: list[str] = []
    alphabet = Alphabet.LOWER
    for zc in zscii_chars:
        if zc == ZC_SPACE:
            result_string.append(" ")
        elif ZC_ABBR_START <= zc <= ZC_ABBR_END:
            if not abbreviations:
                continue
            next_char_index = zscii_chars[zscii_chars.index(zc)+1]
            abbr_index = (zc - 1) * ENTRIES_PER_ABBR_GROUP + \
                next_char_index
            if 0 <= abbr_index < len(abbreviations):
                result_string.append(abbreviations[abbr_index])
        elif zc == ZC_SHIFT1:
            alphabet = Alphabet.UPPER
        elif zc == ZC_SHIFT2:
            alphabet = Alphabet.OTHER
        elif zc >= ZC_FIRST_NORMAL:
            result_string.append(ALPHABETS[alphabet][zc - ZC_FIRST_NORMAL])
            alphabet = Alphabet.LOWER
    return "".join(result_string)


def get_word_type(flags: int) -> set[str]:
    """Interpret dictionary flags"""
    types: set[str] = set()
    if flags & 0b0100_0000:
        types.add("verb")
    if flags & 0b0010_0000:
        types.add("adj")
    if flags & 0b0001_0000:
        types.add("dir")
    if flags & 0b0000_1000:
        types.add("prep")
    if not types:
        types.add("noun?")
    return types


def extract_dictionary(zfile_bytes: bytes) -> Iterable[tuple[str, set[str]]]:
    abbreviations = load_abbreviations(zfile_bytes)
    dict_addr = read_word(zfile_bytes, HEADER_DICT_ADDR)
    seperator_count = zfile_bytes[dict_addr]
    entry_length = zfile_bytes[dict_addr + seperator_count + 1]
    """Entry size"""
    entry_count = read_word(zfile_bytes, dict_addr + seperator_count + 2)
    """How many words dictionary has"""

    base = dict_addr + seperator_count + 4
    for i in range(entry_count):
        entry_offset = base + i * entry_length
        word = decode_zstring(zfile_bytes, entry_offset,
                              entry_length - 2, abbreviations)
        flags = zfile_bytes[entry_offset + entry_length - 2]
        yield word, get_word_type(flags)


def extract_dictionary_from_file(zfile: Path) -> Iterable[tuple[str, set[str]]]:
    with zfile.open("rb") as f:
        data = f.read()
    return extract_dictionary(data)
