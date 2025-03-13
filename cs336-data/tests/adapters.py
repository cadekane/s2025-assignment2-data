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

import nltk
nltk.download('punkt_tab')  # Ensure tokenizer is available
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
    # Ensure the output directory exists and make it if it doesn't
    os.makedirs(output_directory, exist_ok=True)

    # First pass: Count frequency of each line, using the hash of the line as the key
    line_counts = {}
    for path in input_files:
        with open(path, 'r') as file:
            for line in file:
                # Compute the hash of the line
                line_hash = hash(line) # hash() function computes a hash value of the input
                line_counts[line_hash] = line_counts.get(line_hash, 0) + 1
    
    # Second pass: write each file's unique lines (those with FREQUENCY 1) to the output directory
    for path in input_files:
        # Preserve the file name in the output directory
        output_path = os.path.join(output_directory, os.path.basename(path))
        with open(path, 'r') as file, open(output_path, 'w') as output_file:
            for line in file:
                line_hash = hash(line)
                # Only write the line if it occurs exactly once in the whole corpus.
                if line_counts[line_hash] == 1:
                    output_file.write(line)

import os
import nltk
import mmh3
from nltk import word_tokenize
import random
import shutil
import mmh3
def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    # 1. Compute minhash signatures for each document
    minhash_signatures = {}

    for path in input_files:
        with open(path, 'r') as file:
            text = file.read()
            minhash_signatures[path] = compute_minhash_signature(text, num_hashes, ngrams) # finished but unsure if works

    # 2. Use LSH with the provided # of bands to identify candidate duplicates
    candidate_pairs = lsh(minhash_signatures, num_bands, num_hashes) # unfinished
    print(candidate_pairs)

    # 3. Compute Jaccard similarity for candidate pairs and cluster pairs with common documents, such as pair AB and BC into ABC
    filtered_pairs = filter_duplicate_pairs(candidate_pairs, ngrams, jaccard_threshold)
    clusters = cluster_documents(filtered_pairs, ngrams, jaccard_threshold)

    print(filtered_pairs)
    print(clusters)

    # 4. Create a set of documents to discard (one from each duplicate cluster)
    documents_to_discard = set()
    for cluster in clusters:
        # Choose a random document to discard from each cluster
        discard_doc = random.choice(cluster)
        documents_to_discard.add(discard_doc)
    
    # 5. Write all documents that are not in the discard set to the output directory
    for doc in input_files:
        if doc not in documents_to_discard:
            output_path = os.path.join(output_directory, os.path.basename(doc))
            shutil.copy(doc, output_path)

# Helper functions

# 1. Minhash signature computation

import nltk
from nltk import word_tokenize

def compute_minhash_signature(text: str, num_hashes: int, ngrams: int) -> list[int]:
    # Tokenize the text into n-grams
    n_grams = set(nltk.ngrams(word_tokenize(text), n=ngrams)) # n number of n-grams? 

    # Initialize the signature with infinity for each hash function
    signature = [float('inf')] * num_hashes

    # Generate hash functions
    hash_funcs = generate_hash_functions(num_hashes) # k hash functions

    # Is this correct? I'm not sure if hash func for loop should be outside the n_gram for loop
    # Update the signature for each n-gram
    for n_gram in n_grams:
        n_gram_str = ' '.join(n_gram) # Convert tuple back to string for hashing
        for i, hash_func in enumerate(hash_funcs):
            hash_val = hash_func(n_gram_str)
            signature[i] = min(signature[i], hash_val)
    
    return signature # size = num_hashes

def generate_hash_functions(num_hashes: int):
    return [generate_hash_function(seed) for seed in range(num_hashes)]

# Maybe use mmh3 instead of the hash function in python
# def generate_hash_function(seed: int):
#     # Generate a hash function using the given seed
#     def hash_func(n_gram):
#         return hash(n_gram) ^ seed
#     return hash_func

def generate_hash_function(seed: int):
    def hash_func(n_gram):
        return mmh3.hash(n_gram, seed)
    return hash_func

# 2. Locality-Sensitive Hashing (LSH)

from collections import defaultdict
from itertools import combinations

def lsh(minhash_signatures: dict, num_bands: int, num_hashes: int) -> set:
    """
    Given a dictionary mapping document IDs (or file names) to their MinHash signatures,
    this function splits each signature into `num_bands` and buckets documents that share the
    same band signature. If two documents share a band, they are added as a candidate pair.

    Args:
        minhash_signatures (dict): A mapping from document ID to its MinHash signature (list of ints).
        num_bands (int): The number of bands to split each signature into.

    Returns:
        set: A set of tuples, each tuple containing a pair of document IDs that are candidate duplicates.
    """
    buckets = defaultdict(list)
    candidate_pairs = set()
    
    # Assuming all signatures are of equal length.
    # sample_signature = next(iter(minhash_signatures.values()))
    # sig_length = len(sample_signature)
    sig_length = num_hashes # num_hashes is the length of the signature
    band_size = sig_length // num_bands
    # Note: This assumes sig_length is exactly divisible by num_bands.
    
    # Process each document and its signature.
    for doc_id, signature in minhash_signatures.items():
        for band in range(num_bands):
            start = band * band_size
            end = start + band_size
            # Create a tuple that represents the signature for this band.
            band_signature = tuple(signature[start:end])
            # Use a combination of band index and band_signature as the bucket key.
            buckets[(band, band_signature)].append(doc_id) # doc_id is really just the path
    
    # For each bucket, any documents sharing the same band are candidate duplicates.
    for docs in buckets.values():
        if len(docs) > 1:
            # Generate all unique pairs from the documents in this bucket.
            for pair in combinations(sorted(docs), 2): # combinations returns all unique pairs
                candidate_pairs.add(pair)
    
    return candidate_pairs

# 3. Jaccard similarity computation and filtering, and clustering

from nltk import word_tokenize, ngrams

def compute_jaccard_similarity(file1: os.PathLike, file2: os.PathLike, n: int) -> float:
    """Computes the Jaccard similarity between two documents based on n-grams."""
    
    def get_ngrams_from_file(file_path, n):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            tokens = word_tokenize(text)
            return set(ngrams(tokens, n))

    ngrams1 = get_ngrams_from_file(file1, n)
    ngrams2 = get_ngrams_from_file(file2, n)
    
    intersection = len(ngrams1 & ngrams2)  # Intersection
    union = len(ngrams1 | ngrams2)        # Union
    
    return intersection / union if union != 0 else 0.0

from collections import defaultdict
from itertools import combinations

def filter_duplicate_pairs(candidate_pairs, n, threshold):
    """
    Filters candidate pairs based on Jaccard similarity threshold.
    
    Args:
        candidate_pairs (set): Candidate pairs identified by LSH.
        n (int): N-gram size for Jaccard similarity.
        threshold (float): Jaccard similarity threshold.

    Returns:
        list: Filtered pairs that exceed the similarity threshold.
    """
    duplicates = []
    
    for file1, file2 in candidate_pairs:
        similarity = compute_jaccard_similarity(file1, file2, n)
        print(f"Comparing {file1} and {file2}: Jaccard similarity = {similarity}")
        if similarity >= threshold:
            duplicates.append((file1, file2))
    
    print("duplicates: ", duplicates)
    return duplicates


class UnionFind:
    def __init__(self):
        self.parent = {}
        
    def find(self, doc):
        if self.parent[doc] != doc:
            self.parent[doc] = self.find(self.parent[doc])  # Path compression
        return self.parent[doc]
    
    def union(self, doc1, doc2):
        root1 = self.find(doc1)
        root2 = self.find(doc2)
        
        if root1 != root2:
            self.parent[root2] = root1  # Union the two clusters
    
    def add(self, doc):
        if doc not in self.parent:
            self.parent[doc] = doc  # Initialize itself as its own root
    
    def get_clusters(self):
        clusters = {}
        for doc in self.parent:
            root = self.find(doc)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(doc)
        return list(clusters.values())

def cluster_documents(candidate_pairs, n, threshold):
    """
    Clusters documents that exceed the Jaccard similarity threshold using Union-Find.

    Args:
        candidate_pairs (set): Candidate pairs identified by LSH.
        n (int): N-gram size for Jaccard similarity.
        threshold (float): Jaccard similarity threshold.

    Returns:
        list: A list of clusters (each cluster is a list of file paths).
    """
    duplicates = filter_duplicate_pairs(candidate_pairs, n, threshold)
    
    # Union-Find to cluster documents
    uf = UnionFind()
    
    for file1, file2 in duplicates:
        uf.add(file1)
        uf.add(file2)
        uf.union(file1, file2)
    
    return uf.get_clusters()