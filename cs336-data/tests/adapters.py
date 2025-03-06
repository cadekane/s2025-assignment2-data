#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any


from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    # Detect encoding of the byte string
    enc = detect_encoding(html_bytes)
    print(enc)

    # Decode the byte string into a Unicode string
    html = html_bytes.decode('utf-8')
    print(html)

    # If the encoding is not UTF-8, try to decode it using the detected encoding
    if enc != 'utf-8':
        try:
            html = html_bytes.decode(enc)
        except UnicodeDecodeError:
            return None

    # Extract text from the HTML string
    text = extract_plain_text(html)

    return text

import fasttext
def run_identify_language(text: str) -> tuple[Any, float]:
    model = fasttext.load_model('lid.176.bin')

    # Predict the language of the text
    text = text.replace('\n', ' ') # Remove newlines
    predictions = model.predict(text, k=1) # k=1 means we only want the top prediction

    predicted_language = predictions[0][0].replace('__label__', '') # Remove the '__label__' prefix
    confidence_score = predictions[1][0]

    return (predicted_language, confidence_score) # Return the language code and the confidence score


def run_mask_emails(text: str) -> tuple[str, int]:
    import re

    # Define the regular expression pattern for email addresses
    email_pattern = r'[\w\.-]+@[\w\.-]+'

    # Find all email addresses in the text
    emails = re.findall(email_pattern, text)

    # Replace each email address with a placeholder
    masked_text = re.sub(email_pattern, '|||EMAIL_ADDRESS|||', text)

    return (masked_text, len(emails)) # returns the masked string and the number of emails found that were masked


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    import re

    # Define the regular expression pattern for phone numbers
    # phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    phone_pattern = r'(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}'

    # Find all phone numbers in the text
    phones = re.findall(phone_pattern, text)

    # Replace each phone number with a placeholder
    masked_text = re.sub(phone_pattern, '|||PHONE_NUMBER|||', text)

    return (masked_text, len(phones)) # returns the masked string and the number of phone numbers found that were masked


def run_mask_ips(text: str) -> tuple[str, int]:
    import re

    # Define the regular expression pattern for IP addresses
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b' # for IPv4 addresses

    ips = re.findall(ip_pattern, text)

    masked_test = re.sub(ip_pattern, '|||IP_ADDRESS|||', text)

    return (masked_test, len(ips))


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    model = fasttext.load_model('jigsaw_fasttext_bigrams_nsfw_final.bin')

    # Predict the probability of the text being NSFW
    text = text.replace('\n', ' ') # Remove newlines

    predictions = model.predict(text, k=1) # k=1 means we only want the top prediction

    predicted_nsfw = predictions[0][0].replace('__label__', '') # Remove the '__label__' prefix
    confidence_score = predictions[1][0]

    return (predicted_nsfw, confidence_score) # Return the language code and the confidence score


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    model = fasttext.load_model('jigsaw_fasttext_bigrams_hatespeech_final.bin')

    # Predict the probability of the text being NSFW
    text = text.replace('\n', ' ') # Remove newlines

    predictions = model.predict(text, k=1) # k=1 means we only want the top prediction

    predicted_speech = predictions[0][0].replace('__label__', '') # Remove the '__label__' prefix
    confidence_score = predictions[1][0]

    return (predicted_speech, confidence_score) # Return the language code and the confidence score


def run_classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Filter out documents that contain less than 50 or more than 100,000 words
    if len(words) < 50 or len(words) > 100000:
        return False

    # Calculate the mean word length and filter out if it is outside the range of 3 to 10 characters
    avg_word_length = sum(len(word) for word in words) / len(words) if len(words) > 0 else 0
    if avg_word_length < 3 or avg_word_length > 10:
        return False

    # Filter out documents that have more than 30% of lines ending with an ellipsis
    lines = text.split('\n')
    ellipsis_count = sum(line.endswith('...') for line in lines)
    ellipsis_ratio = ellipsis_count / len(lines) if len(lines) > 0 else 0
    if ellipsis_ratio > 0.3:
        return False

    # Filter out documents that contain less than 80% of words with at least one alphabetic character
    alpha_word_count = sum(any(c.isalpha() for c in word) for word in words)
    alpha_ratio = alpha_word_count / len(words) if len(words) > 0 else 0
    if alpha_ratio < 0.8:
        return False

    return True  # Passes all filters


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    raise NotImplementedError


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    raise NotImplementedError
