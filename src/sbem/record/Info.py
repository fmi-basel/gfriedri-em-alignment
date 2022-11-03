class Info:
    def __init__(
        self,
        name: str,
        license: str = "Creative Commons Attribution licence (CC BY)",
    ):
        self._format_version = "0.1.0"
        self._name = name
        self._license = license

    def get_format_version(self) -> str:
        return self._format_version

    def get_name(self) -> str:
        return self._name

    def get_license(self) -> str:
        return self._license
