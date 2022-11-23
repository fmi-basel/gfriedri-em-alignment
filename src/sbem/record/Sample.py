from __future__ import annotations

import os
from os.path import exists, join
from typing import TYPE_CHECKING

from ruyaml import YAML

from sbem.record.Info import Info
from sbem.record.Section import Section

if TYPE_CHECKING:  # pragma: no cover
    from typing import Dict

    from sbem.experiment.Experiment import Experiment


class Sample(Info):
    def __init__(
        self,
        experiment: Experiment,
        name: str,
        description: str,
        documentation: str,
        aligned_data: str,
        license: str = "Creative Commons Attribution licence (CC BY)",
    ):
        super().__init__(name=name, license=license)
        self._experiment = experiment
        self._description = description
        self._documentation = documentation
        self._aligned_data = aligned_data
        self.sections: Dict[str, Section] = {}
        self._min_section_num = {}
        self._max_section_num = {}

        if self._experiment is not None:
            self._experiment.add_sample(self)

    def add_section(self, section: Section):
        if section.get_sample() is None:
            section.set_sample(self)
        else:
            assert section.get_sample() == self, "Section belongs to another " "sample."
        self.sections[section.get_name()] = section
        grid_num = section.get_tile_grid_num()
        if grid_num not in self._min_section_num.keys():
            self._min_section_num[grid_num] = section.get_section_num()
            self._max_section_num[grid_num] = section.get_section_num()
        else:
            if self._min_section_num[grid_num] > section.get_section_num():
                self._min_section_num[grid_num] = section.get_section_num()
            elif self._max_section_num[grid_num] < section.get_section_num():
                self._max_section_num[grid_num] = section.get_section_num()

    def get_section(self, section_name: str) -> Section:
        if section_name in self.sections.keys():
            return self.sections[section_name]
        else:
            return None

    def get_min_section_num(self, tile_grid_num: int):
        return self._min_section_num[tile_grid_num]

    def get_max_section_num(self, tile_grid_num: int):
        return self._max_section_num[tile_grid_num]

    def get_documentation(self) -> str:
        return self._documentation

    def get_description(self) -> str:
        return self._description

    def set_experiment(self, experiment):
        self._experiment = experiment

    def get_experiment(self) -> Experiment:
        return self._experiment

    def get_aligned_data(self):
        return self._aligned_data

    def get_section_range(
        self,
        start_section_num: int,
        end_section_num: int,
        tile_grid_num: int,
        include_skipped: bool = False,
    ):
        sections = []
        for sec in self.sections.values():
            above_start = sec.get_section_num() >= start_section_num
            below_end = sec.get_section_num() <= end_section_num
            match_tile_grid = sec.get_tile_grid_num() == tile_grid_num
            skip = sec.skip() if not include_skipped else False
            if above_start and below_end and match_tile_grid and not skip:
                sections.append(sec)

        sections.sort(key=lambda s: s.get_section_num())
        return sections

    def get_sections_of_acquisition(
        self, acquisition: str, tile_grid_num: int, include_skipped: bool = False
    ):
        sections = []
        for sec in self.sections.values():
            match = sec.get_acquisition() == acquisition
            match_tile_grid = sec.get_tile_grid_num() == tile_grid_num
            skip = sec.skip() if not include_skipped else False
            if match and match_tile_grid and not skip:
                sections.append(sec)

        sections.sort(key=lambda s: s.get_section_num())
        return sections

    def to_dict(self, section_to_subdir: bool = True) -> Dict:
        sections = []
        for s in sorted(self.sections.values(), key=lambda s: s.get_section_num()):
            sec_dict = {
                "name": s.get_name(),
                "section_num": s.get_section_num(),
                "tile_grid_num": s.get_tile_grid_num(),
                "acquisition": s.get_acquisition(),
                "stitched": s.is_stitched(),
                "skip": s.skip(),
            }
            if section_to_subdir:
                sec_dict["details"] = join(s.get_name(), "section.yaml")
            else:
                sec_dict["details"] = s.to_dict()

            sections.append(sec_dict)

        return {
            "name": self.get_name(),
            "license": self.get_license(),
            "format_version": self.get_format_version(),
            "description": self._description,
            "documentation": self._documentation,
            "aligned_data": self._aligned_data,
            "sections": sections,
        }

    def delete_sections(
        self,
        start_section_num: int,
        end_section_num: int,
        tile_grid_num: int,
    ):
        sections = self.get_section_range(
            start_section_num, end_section_num, tile_grid_num, include_skipped=True
        )

        section_names = []
        for section in sections:
            section_names.append(section.get_name())
            del self.sections[section.get_name()]

        return section_names

    def _save_sections(self, root: str, sec_dicts: Dict, overwrite: bool = False):
        for sec_dict in sec_dicts:
            s = self.sections.get(sec_dict["name"])
            s.save(root, overwrite=overwrite)

    def _dump(
        self,
        path: str,
        overwrite: bool = False,
        section_to_subdir: bool = True,
        sample_yaml_only: bool = False,
    ):
        yaml = YAML(typ="rt")
        data = self.to_dict(section_to_subdir=section_to_subdir)
        with open(join(path, "sample.yaml"), "w") as f:
            yaml.dump(data, f)

        if not sample_yaml_only:
            if len(data["sections"]) > 0 and isinstance(
                data["sections"][0]["details"], str
            ):
                self._save_sections(path, data["sections"], overwrite=overwrite)

    def save(
        self,
        path: str,
        overwrite: bool = False,
        section_to_subdir: bool = True,
        sample_yaml_only: bool = False,
    ):
        out_path = join(path, self.get_name())
        if not exists(out_path):
            os.makedirs(out_path, exist_ok=True)
            self._dump(
                path=out_path,
                overwrite=overwrite,
                section_to_subdir=section_to_subdir,
                sample_yaml_only=sample_yaml_only,
            )
        else:
            if overwrite:
                self._dump(
                    path=out_path,
                    overwrite=overwrite,
                    section_to_subdir=section_to_subdir,
                    sample_yaml_only=sample_yaml_only,
                )

    @staticmethod
    def load(path: str) -> Sample:
        yaml = YAML(typ="rt")
        with open(path) as f:
            data = yaml.load(f)

        sample = Sample(
            experiment=None,
            name=data["name"],
            description=data["description"],
            documentation=data["documentation"],
            aligned_data=data["aligned_data"],
            license=data["license"],
        )

        for sec_dict in data["sections"]:
            sec = Section.lazy_loading(**sec_dict)
            sample.add_section(sec)

        return sample
