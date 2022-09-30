from __future__ import annotations

import os
from os.path import exists, join
from typing import TYPE_CHECKING

from ruyaml import YAML

from sbem.record_v2.Info import Info

if TYPE_CHECKING:
    from typing import Dict

    from sbem.experiment.Experiment_v2 import Experiment
    from sbem.record_v2.Section import Section


class Sample(Info):
    def __init__(
        self,
        experiment: Experiment,
        name: str,
        description: str,
        documentation: str,
        aligned_data: str,
        license: str = "Creative Commons Attribution licence (CC " "BY)",
    ):
        super().__init__(name=name, license=license)
        self._experiment = experiment
        self._description = description
        self._documentation = documentation
        self._aligned_data = aligned_data
        self.sections: Dict[str, Section] = {}

        if self._experiment is not None:
            self._experiment.add_sample(self)

    def add_section(self, section: Section):
        self.sections[section.get_section_id()] = section

    def get_section(self, section_id: str) -> Section:
        return self.sections[section_id]

    def get_documentation(self) -> str:
        return self._documentation

    def get_experiment(self) -> Experiment:
        return self._experiment

    def get_aligned_data(self):
        return self._aligned_data

    def to_dict(self, section_to_subdir: bool = True) -> Dict:
        sections = {}
        for k in self.sections.keys():
            s = self.sections.get(k)
            sec_dict = {
                "name": s.get_name(),
                "acquisition": s.get_acquisition(),
                "stitched": s.is_stitched(),
                "skip": s.is_stitched(),
            }
            if section_to_subdir:
                sec_dict["details"] = join(".", s.get_section_id())
            else:
                sec_dict["details"] = s.to_dict()

            sections[s.get_section_id()] = sec_dict

        return {
            "name": self.get_name(),
            "license": self.get_license(),
            "format_version": self.get_format_version(),
            "description": self._description,
            "documentation": self._documentation,
            "aligned_data": self._aligned_data,
            "sections": sections,
        }

    def _save_sections(self, root: str, sec_dicts: Dict, overwrite: bool = False):
        for key in sec_dicts:
            s = self.sections.get(key)
            os.makedirs(root, exist_ok=overwrite)
            s.save(root, overwrite=overwrite)

    def _dump(self, path: str, overwrite: bool = False, section_to_subdir: bool = True):
        yaml = YAML(typ="rt")
        data = self.to_dict(section_to_subdir=section_to_subdir)
        with open(join(path, "sample.yaml"), "w") as f:
            yaml.dump(data, f)

        if isinstance(list(data["sections"].values())[0]["details"], str):
            self._save_sections(path, data["sections"], overwrite=overwrite)

    def save(self, path: str, overwrite: bool = False, section_to_subdir: bool = True):
        out_path = join(path, self.get_name())
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
