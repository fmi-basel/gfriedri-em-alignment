from __future__ import annotations

import json
from os import mkdir
from os.path import exists, join
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from sbem.record.SectionRecord import SectionRecord

if TYPE_CHECKING:
    from sbem.experiment import Experiment


class BlockRecord:
    """
    A block of an SBEM experiment. A block contains many sections. If an
    experiment is provided the block registers itself with the experiment.
    """

    def __init__(
        self,
        experiment: Experiment,
        sbem_root_dir: str,
        block_id: str,
        save_dir: str,
        logger=None,
    ):
        """

        :param experiment: to which this block belongs.
        :param sbem_root_dir: where the image data is saved.
        :param block_id: name of the block.
        :param save_dir: where this block is saved.
        """
        self.logger = logger
        self.experiment = experiment

        if sbem_root_dir is not None:
            assert exists(sbem_root_dir), f"{sbem_root_dir} does not exist."
        self.sbem_root_dir = sbem_root_dir

        self.block_id = block_id

        if self.block_id in self.experiment.zarr_root.keys():
            self.zarr_block = self.experiment.zarr_root[self.block_id]
        else:
            self.zarr_block = self.experiment.zarr_root.create_group(
                self.block_id, overwrite=False
            )

        if save_dir is not None:
            self.save_dir = join(save_dir, self.block_id)
        else:
            self.save_dir = None

        self.section_keys = []
        self.sections = {}

        if self.experiment is not None:
            self.experiment.add_block(self)

    def add_section(self, section: SectionRecord):
        """
        Add a section to this block.

        :param section: to register.
        """
        self.sections[section.section_id] = section

    def get_section(self, section_num: int, tile_grid_num: int = 0,
                    not_found_warning: bool=True):
        """
        Get a section registered with this block.

        :param section_num: Number of the section.
        :param tile_grid_num: Tile grid number of the section.
        :return: `SectionRecord`
        """
        id_ = tuple([section_num, tile_grid_num])
        if id_ in self.sections.keys():
            return self.sections[id_]
        else:
            if not_found_warning:
                msg = f"Section not found: section{section_num}, grid{tile_grid_num}"
                self.logger.warning(msg)
            return None

    def get_section_range(self):
        """
        :return: Sorted array containing all section numbers.
        """
        if self.section_keys is None:
            section_keys = self.sections.keys()
        else:
            section_keys = self.section_keys
        return np.array(sorted(x[0] for x in section_keys))

    def has_missing_section(self):
        """
        :return: True if a section is missing.
        """
        z_diff = np.diff(self.get_section_range())
        return (z_diff > 1).any()

    def get_missing_sections(self):
        """
        :return: Array of missing sections.
        """
        existing_sections = self.get_section_range()
        missing_sections = []
        for i in range(existing_sections[0], existing_sections[-1]):
            if i not in existing_sections:
                missing_sections.append(i)
        return np.array(missing_sections)

    def save(self):
        """
        Save this block to disk.
        """
        assert self.save_dir is not None, "Save dir is not set."
        if not exists(self.save_dir):
            mkdir(self.save_dir)
        block_dict = {
            "block_id": self.block_id,
            "save_dir": self.save_dir,
            "sbem_root_dir": self.sbem_root_dir,
            "n_section": len(self.sections),
            "sections": list(self.sections.keys()),
        }
        with open(join(self.save_dir, "block.json"), "w") as f:
            json.dump(block_dict, f, indent=4)

        for section in self.sections.values():
            section.save()

    def load(self, path):
        """
        Load block from disk.
        :param path: to block directory.
        """
        path_ = join(path, "block.json")
        if not exists(path_) and self.logger is not None:
            self.logger.warning(f"Block not found: {path_}")
        else:
            with open(path_) as f:
                block_dict = json.load(f)

            self.block_id = block_dict["block_id"]
            self.save_dir = block_dict["save_dir"]
            self.sbem_root_dir = block_dict["sbem_root_dir"]
            self.section_keys = list(map(tuple, block_dict["sections"]))

    def _init_sections(self, section_keys):
        """
        Initiate section objects (only used in loading block)

        :param section_keys: list of section keys, each key is
                             (section_num, tile_grid_num)
        """
        for (section_num, tile_grid_num) in tqdm(
            section_keys, desc="Initiating sections"
        ):
            if (section_num, tile_grid_num) not in self.section_keys:
                continue
            section = SectionRecord(
                self,
                section_num,
                tile_grid_num,
                save_dir=self.save_dir,
                logger=self.logger,
            )

    def _load_sections(self, section_keys):
        """
        Load sections

        :param section_keys: list of section keys, each key is
                             (section_num, tile_grid_num)
        :return: list of sections.
                 If the section is not initated,
                 the section is None.
                 If the section cannot be loaded,
                 section.is_loaded is False.
        """
        section_list = self.get_sections(section_keys)
        for section in tqdm(section_list,
                            desc="Loading sections"):
            if section is not None:
                section.load(join(self.save_dir, section.get_name()))
        return section_list

    def get_sections(self, section_keys):
        """
        Get sections

        :param section_keys: list of section keys, each key is
                             (section_num, tile_grid_num)
        :return: list of sections.
                 If the section is not initated,
                 the section is None.
        """
        section_list = [self.get_section(*sk) for sk in section_keys]
        return section_list

    def init_load_section_range(self, start_section: int, end_section: int,
                                grid_num: int):
        """
        Initiate and load a range of secions sections

        :param start_section: the number of the start section.
        :param end_section: the number of the end section.
        :param grid_num: the number of the grid

        :return: list of sections.
                 If the section cannot be initiated,
                 the section is None.
                 If the section cannot be loaded,
                 section.is_loaded is False.
        """
        section_keys = [(s, grid_num) for s in range(start_section, end_section)]
        self._init_sections(section_keys)
        section_list = self._load_sections(section_keys)
        return section_list

    def init_load_section(self, section_num: int, grid_num: int):
        section_keys = [(section_num, grid_num)]
        self._init_sections(section_keys)
        section =  self._load_sections(section_keys)[0]
        return section
