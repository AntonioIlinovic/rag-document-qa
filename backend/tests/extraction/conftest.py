"""Fixtures for document extraction tests."""

import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# PDF fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pdf_path():
    """Absolute path to the sample PDF file in dummy_docs."""
    return Path(__file__).parent.parent.parent.parent / "dummy_docs" / "Zagreb_ice_skating_info.pdf"


@pytest.fixture
def sample_pdf_bytes(sample_pdf_path):
    """Raw bytes of the sample PDF file."""
    with open(sample_pdf_path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image_path():
    """Absolute path to the sample image file in dummy_docs."""
    return Path(__file__).parent.parent.parent.parent / "dummy_docs" / "ice_skating_1.jpg"


@pytest.fixture
def sample_image_bytes(sample_image_path):
    """Raw bytes of the sample image file."""
    with open(sample_image_path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Text fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_txt_path():
    """Absolute path to the sample TXT file in dummy_docs."""
    return Path(__file__).parent.parent.parent.parent / "dummy_docs" / "rag_test" / "example_document.txt"


@pytest.fixture
def sample_txt_bytes(sample_txt_path):
    """Raw bytes of the sample TXT file."""
    with open(sample_txt_path, "rb") as f:
        return f.read()


@pytest.fixture
def sample_md_bytes():
    """Raw bytes of a sample markdown file."""
    markdown_content = """# Sample Markdown

This is a **sample markdown** document with various formatting.

## Subheading

- Item 1
- Item 2
- Item 3

Some *italic* text and `code` snippet.

> This is a blockquote

[Link](https://example.com)
"""
    return markdown_content.encode('utf-8')
