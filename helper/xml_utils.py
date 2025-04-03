from bs4 import BeautifulSoup
from typing import List, Tuple

def read_tei(tei_xml: str) -> BeautifulSoup:
    """Parses TEI XML string using BeautifulSoup and returns the soup object."""
    return BeautifulSoup(tei_xml, features="xml")


def parse_section(section) -> Tuple[str, str]:
    """Extracts the heading and concatenated text content of a TEI XML section."""
    head = section.find('head').getText() if section.find('head') else 'No title'
    paragraphs = section.find_all('p')
    text = ''.join(p.getText() for p in paragraphs)
    return head, text


def extract_title(soup: BeautifulSoup) -> str:
    """Extracts the title of the paper from TEI XML soup."""
    return soup.find('title').getText() if soup.find('title') else 'No title'


def find_all_body_text(tei_xml: str) -> Tuple[str, str]:
    """Finds and concatenates all text in the <body> section of the TEI XML."""
    soup = read_tei(tei_xml)
    title = extract_title(soup)
    body = soup.find('text').find('body')
    sections = body.find_all('div', xmlns="http://www.tei-c.org/ns/1.0") if body else []
    body_text = ''.join(parse_section(section)[1] for section in sections)
    return title, body_text