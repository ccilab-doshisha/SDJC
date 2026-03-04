#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reference Script: Website Scraping and Corpus Construction

This script demonstrates:
1. Scraping text data from a website
2. Extracting and cleaning raw text
3. Normalizing Japanese text
4. Tokenizing using spaCy (ja_ginza)
5. Filtering sentences with fewer than 5 tokens
6. Saving the processed corpus

Note:
This script is provided for reproducibility reference.
Please ensure compliance with the target website's terms of service.
"""

import requests
from bs4 import BeautifulSoup
import re
import unicodedata
from tqdm import tqdm
import spacy


# =========================
# Text Normalization Utils
# =========================

def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s


def remove_extra_spaces(s):
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
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐-‒–⁃⁻₋−]+', '-', s)
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)
    s = re.sub('[~∼∾〜〰～]+', '〜', s)
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
                  '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s


def normalize_text(text):
    assert "\n" not in text
    text = text.replace("\t", "")
    text = text.replace(" ", "")
    text = normalize_neologd(text)
    text = ''.join(text.splitlines())
    return text


# =========================
# Scraping Function
# =========================

def scrape_website(url):
    """
    Scrape visible text from a webpage.
    """
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts and styles
    for script in soup(["script", "style"]):
        script.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    return lines


# =========================
# Main Processing Pipeline
# =========================

def main():

    # Example URLs (replace with actual target pages)
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
    ]

    print("Loading spaCy model...")
    nlp = spacy.load("ja_ginza")

    processed_sentences = []

    for url in urls:
        print(f"Scraping: {url}")
        raw_lines = scrape_website(url)

        for line in tqdm(raw_lines):
            normalized = normalize_text(line)

            if not normalized:
                continue

            doc = nlp(normalized)
            tokens = [token.text for token in doc if not token.is_space]

            # Filter sentences with fewer than 5 tokens
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