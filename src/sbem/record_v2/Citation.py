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
        pretty = self._text
        if self._doi is not None:
            pretty = pretty + "\n" + f"DOI: {self._doi}"

        if self._url is not None:
            pretty = pretty + "\n" + f"URL: {self._url}"

        return pretty

    def to_dict(self):
        return {"doi": self._doi, "text": self._text, "url": self._url}
