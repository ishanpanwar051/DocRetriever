"""
tests/test_pdf_parser.py — Unit tests for PDF & Table parser
"""
import pytest
from src.ingestion.pdf_parser import table_to_markdown, PDFParser


def test_table_to_markdown_basic():
    table_data = [
        ["Header 1", "Header 2", "Header 3"],
        ["Value A", "Value B", "Value C"],
        ["100", "200", "300"],
    ]
    md = table_to_markdown(table_data)
    assert "| Header 1 | Header 2 | Header 3 |" in md
    assert "| --- | --- | --- |" in md
    assert "| Value A | Value B | Value C |" in md
    assert "| 100 | 200 | 300 |" in md


def test_table_to_markdown_handles_none_and_ragged_rows():
    table_data = [
        ["Metric", "Q1", "Q2"],
        ["Revenue", "$10M", None],
        ["Net Margin", "15%"],
    ]
    md = table_to_markdown(table_data)
    assert "| Metric | Q1 | Q2 |" in md
    assert "| Revenue | $10M | - |" in md
    assert "| Net Margin | 15% | - |" in md


def test_table_to_markdown_empty():
    assert table_to_markdown([]) == ""
    assert table_to_markdown([[], []]) == ""


def test_pdf_parser_initialization():
    parser = PDFParser(chunk_size=300, overlap=30)
    assert parser.chunker.chunk_size == 300
    assert parser.chunker.overlap == 30
