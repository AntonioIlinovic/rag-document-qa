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
