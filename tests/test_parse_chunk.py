from finrag.chunk import chunk_filing, split_words
from finrag.parse import html_to_text, split_sections


def test_html_to_text_drops_hidden_xbrl_and_keeps_table_rows(html_10k):
    text = html_to_text(html_10k)
    assert "999999" not in text
    assert "Net sales | 12,345 | 11,431" in text


def test_sections_skip_the_table_of_contents(html_10k):
    sections = {s.key: s for s in split_sections(html_to_text(html_10k), "10-K")}
    assert list(sections) == ["1", "1A", "7", "8"]
    assert "single supplier in Taiwan" in sections["1A"].text
    # The TOC row "Item 1A. | Risk Factors | 8" must not win over the real section.
    assert len(sections["1A"].text) > 50
    assert sections["7"].title.startswith("Management")


def test_ten_q_items_are_keyed_by_part():
    text = "PART I\nItem 2. MD&A\nRevenue grew.\nPART II\nItem 1A. Risk Factors\nNew risks."
    keys = [s.key for s in split_sections(text, "10-Q")]
    assert keys == ["I-2", "II-1A"]


def test_split_words_respects_the_limit_and_overlaps_long_paragraphs():
    para = " ".join(f"w{i}" for i in range(700))
    chunks = split_words(para, max_words=300, overlap=50)
    assert all(len(c.split()) <= 300 for c in chunks)
    assert chunks[1].split()[0] == "w250"  # 50 words of overlap


def test_chunks_carry_their_citation(filing_10k, html_10k):
    chunks = chunk_filing(filing_10k, split_sections(html_to_text(html_10k), "10-K"))
    risk = [c for c in chunks if c.section == "1A"][0]
    assert risk.fiscal_year == 2025
    assert risk.text.startswith("Acme Widgets Inc. 10-K (2025-09-27) - Risk Factors")
    assert "Item 1A Risk Factors" in risk.cite()
    assert len({c.id for c in chunks}) == len(chunks)
