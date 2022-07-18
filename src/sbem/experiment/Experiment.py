import json
from glob import glob
from os import mkdir
from os.path import exists, join

import zarr
from tqdm import tqdm

from sbem.experiment.parse_utils import get_tile_metadata
from sbem.record.BlockRecord import BlockRecord
from sbem.record.SectionRecord import SectionRecord
from sbem.record.TileRecord import TileRecord


class Experiment:
    """
    An experiment consists of multiple blocks. Every block is made up of
    many sections. Each section contains many tiles.

    `Experiment` is a data structure that keeps track of the blocks,
    sections, and tiles once they are acquired by the Friedrich lab at FMI.

    Once the `Experiment` data structure is built it can be saved to disk.
    Saving creates a new directory in `save_dir` with the `name` of the
    `Experiment`. The directory contains a `experiment.json` file. Among
    general information this file contains a list of blocks assigned to this
    `Experiment`. The blocks itself are saved as additional directories in
    `Experiment`.
    """

    def __init__(self, name: str = None, save_dir: str = None, logger=None):
        """
        :param name: Used as save name.
        :param save_dir: The directory where this data structure and all
        processed data should be saved. Preferably this is different from
        `sbem_run_dir`.
        """
        self.logger = logger
        self.name = name
        if self.name is None:
            self.name = "Experiment"

        if save_dir is not None:
            assert exists(save_dir), f"{save_dir} does not exist."
            self.save_dir = join(save_dir, self.name)
            self.zarr_root = zarr.open(zarr.N5FSStore(self.save_dir), mode="a")
        else:
            self.save_dir = join(".", self.name)
            self.zarr_root = None

        self.blocks = {}
        self.section_ranges = None

    def add_block(self, block: BlockRecord):
        """
        Add a block to this experiment.

        :param block: to register
        """
        self.blocks[block.block_id] = block

    def parse_block(
        self,
        sbem_root_dir: str,
        name: str,
        tile_grid: str,
        resolution_xy: float,
        tile_width: int,
        tile_height: int,
        tile_overlap: int,
    ):
        """
        A helper function to parse the SBEM directory structure of a block.

        A new block is created and all sections are created and regsiter
        with this block. To every section all tiles of `tile_grid` and
        `resolution_xy` are added. For every section the tile-id-map is
        computed.

        :param sbem_root_dir: to the directory containing the block acquisition
         data.
        :param name: name of the block
        :param tile_grid: identifier e.g. 'g0001'
        :param resolution_xy: of the acquisitions in nm
        :param tile_width: Width of the acquired tile in px.
        :param tile_height: Height of the acquired tile in px.
        :param tile_overlap: Overlap of acquired tiles in px.
        """
        block = BlockRecord(
            self,
            block_id=name,
            save_dir=self.save_dir,
            sbem_root_dir=sbem_root_dir,
            logger=self.logger,
        )

        tile_grid_num = int(tile_grid[1:])

        metadata_files = sorted(glob(join(sbem_root_dir, "meta", "logs", "metadata_*")))

        tile_specs = get_tile_metadata(
            sbem_root_dir, metadata_files, tile_grid_num, resolution_xy
        )

        for tile_spec in tqdm(tile_specs, desc="Build Block Record"):
            section = block.get_section(tile_spec["z"], tile_grid_num)
            if section is None:
                section = SectionRecord(
                    block=block,
                    section_num=tile_spec["z"],
                    tile_grid_num=tile_grid_num,
                    tile_width=tile_width,
                    tile_height=tile_height,
                    tile_overlap=tile_overlap,
                    save_dir=block.save_dir,
                    logger=self.logger,
                )

            tile = section.get_tile(tile_spec["tile_id"])
            if tile is None:
                TileRecord(
                    section,
                    path=tile_spec["tile_file"],
                    tile_id=tile_spec["tile_id"],
                    x=tile_spec["x"],
                    y=tile_spec["y"],
                    resolution_xy=resolution_xy,
                    logger=self.logger,
                )
            else:
                if self.logger is not None:
                    self.logger.warning(
                        "Tile {} in section {} already exists. "
                        "Skipping. "
                        "Existing tile path: {}; "
                        "Skipped tile path: {}".format(
                            tile.tile_id,
                            section.section_num,
                            tile.path,
                            tile_spec["tile_file"],
                        )
                    )

        for section in tqdm(block.sections.values(), desc="Build " "tile-id-maps"):
            section.compute_tile_id_map()
            section.is_loaded = True

    def save(self):
        """
        Saves to `save_dir`.
        Each block is saved in a sub-dir. Every section is saved in a
        sub-dir of the block-dir. Every section contains the `tile-id-map`
        saved as json and a list of all tiles in json.
        """
        assert self.save_dir is not None, "Save directory not set."
        if not exists(self.save_dir):
            mkdir(self.save_dir)
        self._save_exp_dict()

        for block_name, block in self.blocks.items():
            block.save()

    def load(self, path):
        """
        Load an experiment from disk.

        :param path: to the experiment directory.
        """
        path_ = join(path, "experiment.json")
        if not exists(path_) and self.logger is not None:
            self.logger.warning(f"Experiment not found: {path_}")
        else:
            with open(path_) as f:
                exp_dict = json.load(f)

            self.name = exp_dict["name"]
            self.save_dir = exp_dict["save_dir"]
            self.zarr_root = zarr.open(zarr.N5FSStore(join(self.save_dir)), mode="a")
            for block_name in exp_dict["blocks"]:
                block = BlockRecord(
                    self,
                    block_id=block_name,
                    save_dir=self.save_dir,
                    sbem_root_dir=None,
                    logger=self.logger,
                )
                block.load(join(path, block_name))

    def _compute_section_ranges(self):
        """
        Compute start and end section numbers for each block
        """
        section_ranges = dict()
        for block_name, block in self.blocks.items():
             secs = block.get_section_range()
             section_ranges[block_name] = (secs[0], secs[-1])
        self.section_ranges = section_ranges

    def _sort_section_ranges(self):
        sranges = list(self.section_ranges.items())
        sranges = sorted(sranges, key=lambda x: x[1][0])
        return sranges


    def _divide_sections_to_blocks(self, start_section: int, end_section: int):
        """
        Divide a range of sections into each block

        :param start_section: the number of the start section.
        :param end_section: the number of the end section.
        :return: a list, in which each element is
                 (block_name, [start_section_number_in _block,
                 end_section_number_in_block])
        """
        if not end_section >= start_section:
            raise ValueError("End section should be larger equal to start section.")

        if self.section_ranges is None:
            self._compute_section_ranges()

        sranges = self._sort_section_ranges()

        after_start = [i for i, x in enumerate(sranges)
                       if x[1][0]<=start_section]
        if len(after_start) == 0:
            msg = "Start section outside the ranges of blocks."
            raise ValueError(msg)
        else:
            start_block_idx = after_start[-1]

        end_block_idx = next((i for i, x in enumerate(sranges)
                        if x[1][1]>=end_section), -1)
        if end_block_idx == -1:
            msg = "End section outside the ranges of blocks."
            raise ValueError(msg)

        assert start_block_idx <= end_block_idx

        divided_ranges = dict()
        if start_block_idx == end_block_idx:
            divided_ranges[sranges[start_block_idx][0]] = \
            (start_section, end_section)
        else:
            divided_ranges[sranges[start_block_idx][0]] = \
              (start_section, sranges[start_block_idx][1][1])

            for idx in range(start_block_idx+1, end_block_idx):
                divided_ranges[sranges[idx][0]] = sranges[idx][1]

            divided_ranges[sranges[end_block_idx][0]] = \
            (sranges[end_block_idx][1][0], end_section)
        return divided_ranges

    def load_sections(self, start_section, end_section, grid_num):
        """
        Load a range of sections,
        possibly from multiple blocks

        :param start_section: the number of the start section.
        :param end_section: the number of the end section.
        :param grid_num: the number of the grid
        :return: a list of `SectionRecord` objects.
                 If the section cannot be loaded,
                 section.is_loaded is False.
        """
        divided_ranges = self._divide_sections_to_blocks(start_section, end_section)

        section_list = []
        for block_name, srange in divided_ranges.items():
            sections = self.blocks[block_name].init_load_section_range(*srange, grid_num)
            section_list.extend(sections)

        return section_list

    def get_block_for_section(self, section_num):
        if self.section_ranges is None:
            self._compute_section_ranges()

        sranges = self._sort_section_ranges()

        after_start = [i for i, x in enumerate(sranges)
                       if x[1][0]<=section_num]
        if len(after_start) == 0:
            msg = "Start section outside the ranges of blocks."
            raise ValueError(msg)
        else:
            start_block_idx = after_start[-1]

        end_block_idx = next((i for i, x in enumerate(sranges)
                        if x[1][1]>=section_num), -1)
        if end_block_idx == -1:
            msg = "End section outside the ranges of blocks."
            raise ValueError(msg)

        if start_block_idx == end_block_idx:
            block_idx = start_block_idx
        else:
            raise ValueError("Found section in multiple blocks.")

        block_name = sranges[block_idx][0]
        return block_name

    def load_section_list(self, section_num_list, grid_num):
        block_list = [self.get_block_for_section(s) for s in section_num_list]
        section_list = []

        for block_name, sec in zip(block_list, section_num_list):
            section = self.blocks[block_name].init_load_section(sec, grid_num)
            section_list.append(section)

        return section_list

    def _save_exp_dict(self):
        """
        Save the experiment metadata in json.
        """
        exp_dict = {
            "name": self.name,
            "save_dir": self.save_dir,
            "n_blocks": len(self.blocks),
            "blocks": list(self.blocks.keys()),
        }
        with open(join(self.save_dir, "experiment.json"), "w") as f:
            json.dump(exp_dict, f, indent=4)

    def save_block(self, block_name):
        """
        Save a block and update the experiment json file.

        :param block_name: the name of the block.
        """
        assert self.save_dir is not None, "Save directory not set."
        if not exists(self.save_dir):
            mkdir(self.save_dir)
        self._save_exp_dict()
        assert block_name in self.blocks.keys(), f"Block {block_name} does not exist."
        self.blocks[block_name].save()
