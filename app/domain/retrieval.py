"""Chinese-tokenized BM25 candidate retrieval, not an evidence-support verdict.

Returns original paragraph objects in deterministic rank order. Only paragraphs
with an actual query-token match enter the result; zero matches return []. Source
text is never normalized in storage. Token normalization happens only in memory.
"""

import math
import re
from collections import Counter

import jieba

_TOKEN = re.compile(r"[\u3400-\u9fff]+|[a-z0-9]+", re.IGNORECASE)
_STOP = {"的", "了", "是", "在", "与", "和", "及", "或", "有", "为", "对", "从", "于", "中", "等", "被", "把", "这", "该", "哪些", "什么", "如何", "是否", "怎么", "一个", "进行", "以及", "为什么", "情况", "问题", "请", "the", "a", "an", "of", "in", "to", "is", "and"}


def _tokens(text: str) -> list[str]:
    tokens = []
    for chunk in jieba.cut(text.lower(), HMM=False):
        tokens.extend(part for part in _TOKEN.findall(chunk) if part not in _STOP)
    return tokens


def retrieve(question: str, paragraphs: list[dict], limit: int = 16) -> list[dict]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    if not isinstance(paragraphs, list) or type(limit) is not int or limit < 1:
        raise ValueError("paragraphs must be a list and limit a positive integer")
    query = set(_tokens(question))
    tokenized = []
    for para in paragraphs:
        if not isinstance(para, dict) or not isinstance(para.get("text"), str):
            raise ValueError("each paragraph must contain source text")
        tokenized.append(_tokens(para["text"]))
    if not paragraphs or not query:
        return []
    doc_freq = Counter(token for tokens in tokenized for token in set(tokens))
    average_length = sum(map(len, tokenized)) / len(tokenized)
    if average_length == 0:
        return []
    ranked = []
    for index, tokens in enumerate(tokenized):
        frequencies = Counter(tokens)
        matches = query & frequencies.keys()
        if not matches:
            continue
        score = 0.0
        for term in matches:
            idf = math.log(1 + (len(paragraphs) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            frequency = frequencies[term]
            score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * len(tokens) / average_length))
        ranked.append((score, index))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [paragraphs[index] for _, index in ranked[:limit]]
