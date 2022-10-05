from __future__ import annotations

import os
from os.path import exists, join
from typing import TYPE_CHECKING

from ruyaml import YAML

from sbem.record_v2.Author import Author
from sbem.record_v2.Citation import Citation
from sbem.record_v2.Info import Info
from sbem.record_v2.Sample import Sample

if TYPE_CHECKING:  # pragma: no cover
    from typing import Dict, List


class Experiment(Info):
    def __init__(
        self,
        name: str,
        description: str,
        documentation: str,
        authors: List[Author],
        root_dir: str,
        exist_ok: bool = False,
        license: str = "Creative Commons Attribution licence (CC " "BY)",
        cite: List[Citation] = None,
    ):
        super().__init__(name=name, license=license)
        self._description = description
        self._root_dir = root_dir
        self._documentation = documentation
        self._authors: List[Author] = authors
        self._cite: List[Citation] = cite
        self._samples: Dict[str, Sample] = {}

        if self._root_dir is not None:
            os.makedirs(self._root_dir, exist_ok=exist_ok)

    def add_sample(self, sample: Sample):
        if sample.get_experiment() is None:
            sample.set_experiment(self)
        else:
            assert sample.get_experiment() == self, (
                "Sample belongs to " "another experiment."
            )
        self._samples[sample.get_name()] = sample

    def get_sample(self, name: str) -> Sample:
        return self._samples[name]

    def get_description(self) -> str:
        return self._description

    def get_documentation(self) -> str:
        return self._documentation

    def get_root_dir(self) -> str:
        return self._root_dir

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

    def to_dict(self) -> Dict:
        samples = []
        for k in self._samples.keys():
            s = self._samples.get(k)
            samples.append(s.get_name())

        return {
            "name": self.get_name(),
            "license": self.get_license(),
            "format_version": self.get_format_version(),
            "description": self._description,
            "root_dir": self._root_dir,
            "documentation": self._documentation,
            "authors": [a.to_dict() for a in self._authors],
            "cite": [c.to_dict() for c in self._cite],
            "samples": samples,
        }

    def _dump(self, path: str, overwrite: bool = False, section_to_subdir: bool = True):
        yaml = YAML(typ="rt")
        with open(join(path, "experiment.yaml"), "w") as f:
            yaml.dump(self.to_dict(), f)

        for s in self._samples.values():
            s.save(path, overwrite=overwrite, section_to_subdir=section_to_subdir)

    def save(self, overwrite: bool = False, section_to_subdir: bool = True):
        out_path = join(self._root_dir, self.get_name())
        if not exists(out_path):
            os.makedirs(out_path, exist_ok=True)
            self._dump(
                path=out_path, overwrite=overwrite, section_to_subdir=section_to_subdir
            )
        else:
            if overwrite:
                self._dump(
                    path=out_path,
                    overwrite=overwrite,
                    section_to_subdir=section_to_subdir,
                )
            else:
                raise FileExistsError()

    @staticmethod
    def load(path: str) -> Experiment:
        yaml = YAML(typ="rt")
        with open(path) as f:
            data = yaml.load(f)

        exp = Experiment(
            name=data["name"],
            description=data["description"],
            documentation=data["documentation"],
            authors=[
                Author(name=a["name"], affiliation=a["affiliation"])
                for a in data["authors"]
            ],
            root_dir=data["root_dir"],
            exist_ok=True,
            license=data["license"],
            cite=[
                Citation(doi=d["doi"], text=d["text"], url=d["url"])
                for d in data["cite"]
            ],
        )

        for s in data["samples"]:
            sample = Sample.load(
                join(exp.get_root_dir(), exp.get_name(), s, "sample.yaml")
            )
            exp.add_sample(sample)

        return exp
