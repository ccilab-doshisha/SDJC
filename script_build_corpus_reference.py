#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reference Script: Text Collection and Corpus Construction

This script demonstrates a general pipeline used to construct a corpus:

1. Collect text data
   - Scraping from websites
   - Loading from local files (TXT / CSV / JSON)

2. Extract and clean raw text

3. Normalize Japanese text

4. Tokenize using spaCy (ja_ginza)

5. Filter sentences with fewer than 5 tokens

6. Save the processed corpus

Note:
This script is provided for reproducibility reference.
Please ensure compliance with the target website's terms of service
and data usage policies before collecting or redistributing data.
"""

import requests
from bs4 import BeautifulSoup
import re
import unicodedata
import json
import csv
from tqdm import tqdm
import spacy


# =========================
# Text Normalization Utils
# =========================

def unicode_normalize(cls, s):
    """
    Normalize Unicode characters in specified ranges.
    """
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s


def remove_extra_spaces(s):
    """
    Remove unnecessary spaces between Japanese characters and symbols.
    """
    s = re.sub('[ 　]+', ' ', s)

    blocks = ''.join(('\u4E00-\u9FFF',
                      '\u3040-\u309F',
                      '\u30A0-\u30FF',
                      '\u3000-\u303F',
                      '\uFF00-\uFFEF'))

    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)

    return s


def normalize_neologd(s):
    """
    Apply normalization similar to mecab-ipadic-neologd preprocessing.
    """
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐-‒–⁃⁻₋−]+', '-', s)
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)
    s = re.sub('[~∼∾〜〰～]+', '〜', s)

    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
                  '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」')
    )

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)

    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)

    return s


def normalize_text(text):
    """
    Full text normalization pipeline.
    """
    assert "\n" not in text

    text = text.replace("\t", "")
    text = text.replace(" ", "")

    text = normalize_neologd(text)

    text = ''.join(text.splitlines())

    return text


# =========================
# Data Collection Functions
# =========================

def scrape_website(url):
    """
    Scrape visible text from a webpage.
    """
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script/style tags
    for script in soup(["script", "style"]):
        script.decompose()

    text = soup.get_text(separator="\n")

    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    return lines


def load_txt(file_path):
    """
    Load raw text from a TXT file.
    Each line is treated as one sentence or text segment.
    """
    lines = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    return lines


def load_csv(file_path, text_column=0):
    """
    Load raw text from a CSV file.

    Parameters
    ----------
    text_column : int
        Index of the column containing text.
    """
    lines = []

    with open(file_path, encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > text_column:
                text = row[text_column].strip()
                if text:
                    lines.append(text)

    return lines


def load_json(file_path, key="text"):
    """
    Load raw text from a JSON file.

    Expected format example:
    [
        {"text": "..."},
        {"text": "..."}
    ]
    """
    lines = []

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if key in item and item[key]:
            lines.append(item[key])

    return lines


# =========================
# Main Processing Pipeline
# =========================

def main():

    # Example data sources
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
    ]

    txt_files = [
        "data/sample.txt"
    ]

    csv_files = [
        "data/sample.csv"
    ]

    json_files = [
        "data/sample.json"
    ]

    print("Loading spaCy model...")
    nlp = spacy.load("ja_ginza")

    raw_lines = []

    # Scrape website text
    for url in urls:
        print(f"Scraping: {url}")
        raw_lines.extend(scrape_website(url))

    # Load TXT files
    for path in txt_files:
        print(f"Loading TXT: {path}")
        raw_lines.extend(load_txt(path))

    # Load CSV files
    for path in csv_files:
        print(f"Loading CSV: {path}")
        raw_lines.extend(load_csv(path))

    # Load JSON files
    for path in json_files:
        print(f"Loading JSON: {path}")
        raw_lines.extend(load_json(path))

    print(f"Total collected raw lines: {len(raw_lines)}")

    processed_sentences = []

    # Text processing pipeline
    for line in tqdm(raw_lines):

        normalized = normalize_text(line)

        if not normalized:
            continue

        doc = nlp(normalized)

        tokens = [token.text for token in doc if not token.is_space]

        # Filter short sentences
        if len(tokens) >= 5:
            processed_sentences.append(normalized)

    # Save processed corpus
    output_path = "processed_corpus.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for sentence in processed_sentences:
            f.write(sentence + "\n")

    print(f"Saved {len(processed_sentences)} sentences to {output_path}")


if __name__ == "__main__":
    main()
