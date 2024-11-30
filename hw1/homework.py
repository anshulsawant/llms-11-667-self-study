import argparse
import re
import requests
import json
from utils import  read_warc_file, retrieve_bad_words
import utils
from datasets import load_dataset
from typing import Set, Dict
import bs4
import string
import html2text

def compare(warc_file, wet_file, url, output_warc=False):
    warc = utils.read_warc_file_url(warc_file, url).decode()
    wet = utils.read_wet_file_url(wet_file, url).decode()
    text = html_to_text(warc)
    cleaned_text = clean_text(text)
    cleaned_nopii_text = replace_pii(cleaned_text)
    passes_check = heuristic_quality_filter(cleaned_nopii_text)
    print(url)
    print("Passes heuristic quality filter:", passes_check)
    print(cleaned_nopii_text)
    print('\n\n\n')
    print("Wet:\n", wet)
    if output_warc:
        print('\n\n\n')
        print('Warc:\n', warc)

        
def html_to_text(html: bytes) -> str:
    """Converts HTML content to plain text..
    Args:
        html (bytes): HTML content as bytes.
    Returns:
        str: Plain text extracted from HTML.
    """
    return bs4.BeautifulSoup(html, 'html.parser').get_text()


def replace_pii(text: str) -> str:
    """Masks personally identifiable information (PII) from text with the specified masking formats.
    Args: 
        text (str): Candidate text.
    Returns:
        str: Text with PII obfuscated.
    """
    pattern = r"\d{3}-\d{2}-\d{4}"
    replacement = "XXX-XX-XXXX"
    return re.sub(pattern, replacement, text)


def clean_text(text: str) -> str:
    """Removes substrings identified as low-quality according to alphanumeric, whitespace and valid document checks.  
    Args:
        text (str): document to process.
    Returns:
        str: cleaned document
    """
    long_word = re.compile(r'[a-zA-Z0-9]{101,}')
    no_punct = re.compile('[^' + string.punctuation + ']')
    def is_clean(p: str) -> bool:
      return len(re.findall(long_word, p)) == 0 and len(re.findall(no_punct, p)) > 0
    return '\n'.join([p for p in text.split('\n') if is_clean(p)])


def heuristic_quality_filter(text: str) -> bool:
    """Rejects documents based on the presence of bad words and punctuation.
    Args:
        text (str): document to check
    Returns:
        bool: returns True if the document passes the filters, False otherwise.
    """
    bad_words = retrieve_bad_words()  # Read the bad words list

    sep = string.punctuation + string.whitespace
    words = {word.lower() for word in text.split(sep)}
    # Check for bad words
    if len(bad_words.intersection(words)) > 0:
        return False

    characters = {c for c in text}
    # Check for punctuation
    if not any(char in string.punctuation for char in characters):
        return False

    # Check for non-whitespace characters
    if not any(not char.isspace() for char in characters):
        return False

    # Check for allowed character percentage
    allowed_chars = ('[' + string.ascii_letters + string.digits +
                     string.punctuation + string.whitespace + ']')
    allowed_count = len(re.findall(allowed_chars, text))
    if allowed_count / len(text) < 0.8:
        return False

    return True


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname',
                        type=str,
                        default='',
                        help='Specify the path for your warc file.')
    parser.add_argument('--num_records',
                        type=int,
                        default=30,
                        help='Specify the number of records you want to parse'
                        ' (only used for debugging with smaller sets)')
    args = parser.parse_args()

    if args.fname:
        for url, html_text in read_warc_file(args.fname, args.num_records):
            text = html_to_text(html_text)
            cleaned_text = clean_text(text)
            cleaned_nopii_text = replace_pii(cleaned_text)
            passes_check = heuristic_quality_filter(cleaned_nopii_text)
            print(url)
            print("Passes heuristic quality filter:", passes_check)
            print(cleaned_nopii_text)
            print("\n\n\n")
    else:
        print("Usage: python homework.py --fname data.warc")
