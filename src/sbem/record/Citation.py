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

    def to_pretty_str(self) -> str:
        pretty = self.get_text()
        if self.get_doi() is not None:
            pretty = pretty + "\n" + f"DOI: {self.get_doi()}"

        if self.get_url() is not None:
            pretty = pretty + "\n" + f"URL: {self.get_url()}"

        return pretty

    def to_dict(self):
        return {"doi": self._doi, "text": self._text, "url": self._url}
