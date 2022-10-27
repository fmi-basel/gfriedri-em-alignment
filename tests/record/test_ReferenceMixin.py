from unittest import TestCase

from sbem.record_v2.Author import Author
from sbem.record_v2.Citation import Citation
from sbem.record_v2.ReferenceMixin import ReferenceMixin


class ReferenceMixinTest(TestCase):
    def test_reference_mixin(self):
        authors = [
            Author("author 1", "aff 1"),
            Author("author 2", "aff 2"),
            Author("author 3", "aff 1"),
        ]
        cite = [Citation("doi", "text", "url")]
        ref = ReferenceMixin(authors, cite)

        assert (
            ref.get_authors_pretty() == "author 1[1], author 2[2], "
            "author 3[1]\n[1] aff 1\n[2] aff 2"
        )
        assert ref.get_citations_pretty() == "text\nDOI: doi\nURL: url"
