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


# ---------------------------------------------------------------- real-filing layouts
def _sections(name: str) -> dict:
    from pathlib import Path

    html = (Path(__file__).parent / "fixtures" / name).read_text(encoding="utf-8")
    return {s.key: s for s in split_sections(html_to_text(html), "10-K")}


def test_running_page_headers_do_not_cut_a_section_into_pages():
    # Microsoft repeats "PART I / Item 1A" on every page: the section is all pages.
    s = _sections("page_headers_10k.html")
    assert "intense" in s["1A"].text and "artificial intelligence" in s["1A"].text
    assert "Cyberattacks" in s["1A"].text
    assert "Azure" in s["7"].text and "Operating income" in s["7"].text
    assert "200,000 people" in s["1"].text


def test_short_item_pointing_to_a_note_receives_that_note():
    # Microsoft Item 3: "Refer to Note 14 - Contingencies"; the Note leaves Item 8.
    s = _sections("page_headers_10k.html")
    assert "Irish Data Protection Commission" in s["3"].text
    assert "Irish Data Protection Commission" not in s["8"].text
    assert "repurchased" in s["8"].text and "repurchased" not in s["3"].text


def test_cross_reference_10k_items_receive_the_pages_they_point_to():
    # JPMorgan: Items 7 / 7A / 1C / 8 only give page ranges, Item 3 a Note number.
    s = _sections("cross_reference_10k.html")
    assert "value-at-risk" in s["7A"].text  # page 47
    assert "Board Risk Committee" in s["1C"].text  # page 48 ("on page 48")
    assert "EXECUTIVE OVERVIEW" in s["7"].text and "LIQUIDITY" in s["7"].text
    # The narrowest reference wins: 7A's and 1C's pages are not duplicated in Item 7.
    assert "value-at-risk" not in s["7"].text and "Board Risk Committee" not in s["7"].text
    assert "57,048" in s["8"].text and "Pledged assets" in s["8"].text
    # Note 30 goes to Item 3, up to Note 31; its TOC row ("Note 30 - Litigation | 51")
    # in Item 15 is not mistaken for the heading.
    assert "reasonably possible losses" in s["3"].text
    assert "International operations" not in s["3"].text
    assert "reasonably possible losses" not in s["8"].text
    # Item 15 keeps its exhibits and loses the annual-report pages it used to swallow.
    assert "Restated Certificate" in s["15"].text and "EXECUTIVE OVERVIEW" not in s["15"].text


def test_a_real_section_mentioning_a_note_keeps_its_own_text():
    # NVIDIA Item 2 (137 words) cites a lease Note in passing: not a pointer.
    filler = " ".join(["We own and lease offices in Santa Clara."] * 12)
    text = (f"Item 2. Properties\n{filler} For leases, refer to Note 3.\n"
            "Item 8. Financial Statements\nNote 3 - Leases\nLease cost was $300 million.\n"
            "Note 4 - Debt\nWe repaid notes.")
    s = {x.key: x for x in split_sections(text, "10-K")}
    assert "Lease cost" in s["8"].text and "Lease cost" not in s["2"].text
