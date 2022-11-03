class Author:
    def __init__(self, name: str, affiliation: str):
        self._name = name
        self._affiliation = affiliation

    def get_name(self) -> str:
        return self._name

    def get_affiliation(self) -> str:
        return self._affiliation

    def to_dict(self):
        return {"name": self._name, "affiliation": self._affiliation}
