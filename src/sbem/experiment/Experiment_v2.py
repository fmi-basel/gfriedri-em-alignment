from typing import Dict, List


class Author:
    def __init__(self, name: str, affiliation: str):
        self._name = name
        self._affiliation = affiliation

    def get_name(self) -> str:
        return self._name

    def get_affiliation(self) -> str:
        return self._affiliation


class Citation:
    def __init__(self, doi: str, text: str, url: str):
        self._doi = doi
        self._text = text
        self._url = url

    def get_doi(self) -> str:
        return self._doi

    def get_text(self) -> str:
        return self._text

    def get_url(self) -> str:
        return self._url


class Experiment:
    def __init__(
        self,
        name: str,
        description: str,
        authors: List[Dict],
        license: str = "Creative Commons Attribution licence (CC " "BY)",
        cite: List[Dict] = None,
    ):
        self.name = name
        self.description = description
        self.documentation = "./README.md"
        self.license = license
        self.authors = [
            Author(name=a["name"], affiliation=a["affiliation"]) for a in authors
        ]
        self.cite = [Citation(doi=c["doi"], text=c["text"], url=c["url"]) for c in cite]
