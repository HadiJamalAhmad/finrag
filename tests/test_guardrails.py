from finrag import grounding, scope


def _scope(filing_10k, sec_names):
    return scope.Scope.build([filing_10k], sec_names)


# ---------------------------------------------------------------- scope
def test_question_about_corpus_company_is_in_scope(filing_10k, sec_names):
    d = scope.check("What supply risks does Acme describe?", _scope(filing_10k, sec_names))
    assert d.in_scope and d.tickers == {"ACME"}


def test_company_outside_the_corpus_is_refused(filing_10k, sec_names):
    d = scope.check("What was Tesla's revenue in 2025?", _scope(filing_10k, sec_names))
    assert not d.in_scope and "TSLA" in d.reason


def test_ticker_outside_the_corpus_is_refused(filing_10k, sec_names):
    assert not scope.check("TSLA gross margin?", _scope(filing_10k, sec_names)).in_scope


def test_lowercase_common_word_is_not_a_company(filing_10k, sec_names):
    # "target" is Target Corp's alias, but lowercase here: not a company mention.
    d = scope.check("What is Acme's target gross margin?", _scope(filing_10k, sec_names))
    assert d.in_scope


def test_generic_first_word_is_not_an_alias(filing_10k, sec_names):
    # "First Example Bancorp" must not turn every "First, ..." into a bank.
    sc = _scope(filing_10k, sec_names)
    assert "first" not in sc.sec_aliases
    assert scope.check("First, what are Acme's segments?", sc).in_scope


def test_years_covered_by_a_10k_include_two_comparative_years(filing_10k, sec_names):
    sc = _scope(filing_10k, sec_names)
    assert scope.check("Acme net sales in fiscal 2023?", sc).in_scope
    d = scope.check("Acme net sales in fiscal 2019?", sc)
    assert not d.in_scope and "2019" in d.reason


# ---------------------------------------------------------------- numbers
EVIDENCE = ["Net sales | 12,345 | 11,431", "Total net sales increased 8% to $12,345 million."]


def test_exact_figure_is_grounded():
    assert grounding.check("Net sales were $12,345 million [1].", EVIDENCE).grounded


def test_rescaled_and_rounded_figure_is_grounded():
    # 12,345 million stated as 12.3 billion: same figure, rounded, other unit.
    assert grounding.check("Net sales reached $12.3 billion [2].", EVIDENCE).grounded


def test_invented_figure_is_caught():
    r = grounding.check("Net sales were $13,900 million, up 8% [1].", EVIDENCE)
    assert not r.grounded and r.ungrounded == ["$13,900"]


def test_years_citations_and_small_counts_are_not_checked():
    r = grounding.check("In fiscal 2025 [2], the three segments grew 8% [1].", EVIDENCE)
    assert r.grounded
