from unittest import TestCase

from sbem.record.Author import Author


class AuthorTest(TestCase):
    def test_author(self):
        author = Author("Tim-Oliver", "FMI")

        assert author.get_name() == "Tim-Oliver"
        assert author.get_affiliation() == "FMI"
        assert author.to_dict() == {"name": "Tim-Oliver", "affiliation": "FMI"}
