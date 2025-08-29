import struct

# Z-machine alphabet tables
A0 = "abcdefghijklmnopqrstuvwxyz"
A1 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
A2 = " \n0123456789.,!?_#'\"/\\-:()"

ALPHABETS = [A0, A1, A2]


def decode_zchars(zchars, abbr_table, data):
    result = []
    alphabet = 0
    shift = None

    i = 0
    while i < len(zchars):
        z = zchars[i]

        if z == 0:  # space
            result.append(" ")
        elif 1 <= z <= 3:  # abbreviation
            i += 1
            if i < len(zchars):
                abbr_index = (z - 1) * 32 + zchars[i]
                abbr_addr = struct.unpack(
                    ">H", data[abbr_table + abbr_index*2: abbr_table + abbr_index*2 + 2])[0]
                abbr_zchars = decode_word(data, abbr_addr)
                result.append(decode_zchars(abbr_zchars, abbr_table, data))
        elif z == 4:  # shift A1
            shift = 1
        elif z == 5:  # shift A2
            shift = 2
        elif alphabet == 2 and z == 2:  # shift-lock A1
            alphabet = 1
        elif alphabet == 2 and z == 3:  # shift-lock A2
            alphabet = 2
        elif alphabet == 2 and z == 6:  # 10-bit ZSCII
            if i+2 < len(zchars):
                zscii = (zchars[i+1] << 5) | zchars[i+2]
                result.append(chr(zscii))
                i += 2
        elif 6 <= z <= 31:
            alpha = shift if shift is not None else alphabet
            table = ALPHABETS[alpha]
            if z-6 < len(table):
                result.append(table[z-6])
            shift = None
        i += 1

    return "".join(result).strip()


def decode_word(data, addr):
    zchars = []
    finished = False
    while not finished:
        word = struct.unpack(">H", data[addr:addr+2])[0]
        addr += 2
        finished = (word & 0x8000) != 0
        zchars.extend([(word >> 10) & 0x1F, (word >> 5) & 0x1F, word & 0x1F])
    return zchars


def extract_dictionary(filename):
    with open(filename, "rb") as f:
        data = f.read()

    abbr_table = struct.unpack(">H", data[0x18:0x1A])[0]
    dict_addr = struct.unpack(">H", data[8:10])[0]

    p = dict_addr
    num_separators = data[p]
    p += 1 + num_separators

    entry_length = data[p]
    num_entries = struct.unpack(">H", data[p+1:p+3])[0]
    p += 3

    dictionary = []
    for _ in range(num_entries):
        entry = data[p:p+entry_length]
        zchars = []
        for i in range(0, entry_length-2, 2):  # last 2 bytes = flags
            word = struct.unpack(">H", entry[i:i+2])[0]
            zchars.extend(
                [(word >> 10) & 0x1F, (word >> 5) & 0x1F, word & 0x1F])

        word_str = decode_zchars(zchars, abbr_table, data)

        # part-of-speech flags (last two bytes)
        flags = entry[entry_length-2:entry_length]
        pos = []
        if flags[0] & 1:
            pos.append("verb")
        if flags[0] & 2:
            pos.append("noun")
        if flags[0] & 4:
            pos.append("adj")
        if flags[0] & 8:
            pos.append("prep")

        dictionary.append((word_str, pos))
        p += entry_length
    return dictionary


if __name__ == "__main__":
    words = extract_dictionary("zork_285.z5")
    for w, pos in words:
        if pos:
            print(f"{w:15}  {','.join(pos)}")
        else:
            print(w)
