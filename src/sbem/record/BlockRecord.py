import json
from os import mkdir
from os.path import exists, join

import numpy as np
from tqdm import tqdm

from src.sbem.experiment import Experiment
from src.sbem.record.SectionRecord import SectionRecord


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
        if save_dir is not None:
            self.save_dir = join(save_dir, self.block_id)
        else:
            self.save_dir = None

        self.sections = {}

        if self.experiment is not None:
            self.experiment.add_block(self)

    def add_section(self, section: SectionRecord):
        """
        Add a section to this block.

        :param section: to register.
        """
        self.sections[section.section_id] = section

    def get_section(self, section_num: int, tile_grid_num: int = 0):
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
            return None

    def get_section_range(self):
        """
        :return: Sorted array containing all section numbers.
        """
        return np.array(
            sorted(section.section_num for section in self.sections.values())
        )

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
            for (section_num, tile_grid_num) in tqdm(
                block_dict["sections"], desc="Loading Sections"
            ):
                section = SectionRecord(
                    self, section_num, tile_grid_num, self.save_dir, logger=self.logger
                )
                section.load(join(path, section.get_name()))
