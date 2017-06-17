import xml.etree.ElementTree as ET
import itertools


def load_news15(xml_file):
    xml = ET.parse(xml_file)
    src_tokens = []
    trg_tokens = []
    for name in xml.findall('.//Name'):
        src = name.find('SourceName')
        trg = name.find('TargetName')
        src_tokens.append(list(src))
        trg_tokens.append(list(trg))

    return src_tokens, trg_tokens


def generate_mapping(tokens, is_trg):
    token2id = dict()
    id2token = dict()
    if is_trg:
        tokens = ["</s>", "<s>"] + sorted(list(tokens))
    else:
        tokens = ["<UNK>"] + sorted(list(tokens))
    for i, c in enumerate(tokens):
        token2id[c] = i
        id2token[i] = c

    return token2id, id2token



