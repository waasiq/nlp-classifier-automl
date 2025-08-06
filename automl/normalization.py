from html import unescape
from bs4 import BeautifulSoup
import re

def remove_markdowns(raw_text: str) -> str:
    """
    Remove markdown tags from the given text.
    
    Args:
        text (str): The input text containing markdown tags.
        
    Returns:
        str: The text with markdown tags removed.
    """
    fixed = re.sub(r'#(\d+);', r'&#\1;', raw_text)
    text = unescape(fixed)
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    # Remove markdown specific characters
    text = text.replace(r'\$', '$')   # \$
    text = re.sub(r"\s+([\'.,;:!?])", r'\1', text)
    text = re.sub(r'\s+', ' ', text)  # collapse spaces
    return text
