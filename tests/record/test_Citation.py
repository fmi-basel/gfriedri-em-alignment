from unittest import TestCase

from sbem.record_v2.Citation import Citation


class CitationTest(TestCase):
    def test_citation(self):
        citation = Citation("a doi", "a text", "a url")

        assert citation.get_doi() == "a doi"
        assert citation.get_text() == "a text"
        assert citation.get_url() == "a url"

        assert citation.to_dict() == {"doi": "a doi", "text": "a text", "url": "a url"}

        pretty = "a text\nDOI: a doi\nURL: a url"
        assert citation.to_pretty_str() == pretty

        citation = Citation(doi=None, text="a text", url=None)
        pretty = "a text"
        assert citation.to_pretty_str() == pretty

        citation = Citation(doi="a doi", text="a text", url=None)
        pretty = "a text\nDOI: a doi"
        assert citation.to_pretty_str() == pretty

        citation = Citation(doi=None, text="a text", url="a url")
        pretty = "a text\nURL: a url"
        assert citation.to_pretty_str() == pretty
