from typing import List

from sbem.record.Author import Author
from sbem.record.Citation import Citation


class ReferenceMixin:
    def __init__(self, authors: List[Author], cite: List[Citation], **kwargs):
        super().__init__(**kwargs)
        self._authors = authors
        self._cite = cite

    def get_authors_pretty(self) -> str:
        affilitations = {}
        for a in self._authors:
            if not a.get_affiliation() in affilitations.keys():
                idx = len(affilitations) + 1
                affilitations[a.get_affiliation()] = idx

        author_affiliation = []
        for a in self._authors:
            author_affiliation.append(
                a.get_name() + f"[{affilitations.get(a.get_affiliation())}]"
            )

        pretty = ", ".join(author_affiliation)
        pretty = pretty + "\n"

        aff_index = [f"[{affilitations[a]}] {a}" for a in affilitations.keys()]
        pretty = pretty + "\n".join(aff_index)

        return pretty

    def get_citations_pretty(self) -> str:
        citations = [c.to_pretty_str() for c in self._cite]
        return "\n\n".join(citations)
