import json
from glob import glob
from logging import Logger
from os import mkdir
from os.path import exists, join

from sbem.experiment.parse_utils import get_tile_metadata
from sbem.record.BlockRecord import BlockRecord
from sbem.record.SectionRecord import SectionRecord
from sbem.record.TileRecord import TileRecord
from tqdm import tqdm


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

    def __init__(self, name: str = None, save_dir: str = None):
        """
        :param name: Used as save name.
        :param save_dir: The directory where this data structure and all
        processed data should be saved. Preferably this is different from
        `sbem_run_dir`.
        """
        self.logger = Logger(f"Experiment[{name}]")
        self.name = name

        if save_dir is not None:
            assert exists(save_dir), f"{save_dir} does not exist."
            self.save_dir = join(save_dir, self.name)
        else:
            self.save_dir = None

        self.blocks = {}

    def register_block(self, block: BlockRecord):
        """
        Register a block with this experiment.

        :param block: to register
        """
        self.blocks[block.block_id] = block

    def add_block(
        self, sbem_root_dir: str, name: str, tile_grid: str, resolution_xy: float
    ):
        """
        A helper function to parse the SBEM directory structure of a block.

        A new block is created and all sections are created and regsiter
        with this block. To every section all tiles of `tile_grid` and
        `resolution_xy` are added. For every section the tile-id-map is
        computed.

        :param path: to the directory containing the block acquisition data.
        :param tile_grid: identifier e.g. 'g0001'
        :param resolution_xy: of the acquisitions in nm
        """
        block = BlockRecord(
            self, block_id=name, save_dir=self.save_dir, sbem_root_dir=sbem_root_dir
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
                    block, tile_spec["z"], tile_grid_num, block.save_dir
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
                )
            else:
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

    def save(self):
        """
        Saves to `save_dir`.
        Each block is saved in a sub-dir. Every section is saved in a
        sub-dir of the block-dir. Every section contains the `tile-id-map`
        saved as npz and a list of all tiles in json.
        """
        assert self.save_dir is not None, "Save directory not set."
        mkdir(self.save_dir)
        exp_dict = {
            "name": self.name,
            "save_dir": self.save_dir,
            "n_blocks": len(self.blocks),
            "blocks": list(self.blocks.keys()),
        }
        with open(join(self.save_dir, "experiment.json"), "w") as f:
            json.dump(exp_dict, f, indent=4)

        for block_name, block in self.blocks.items():
            block.save()

    def load(self, path):
        """
        Load an experiment from disk.

        :param path: to the experiment directory.
        """
        path_ = join(path, "experiment.json")
        if not exists(path_):
            self.logger.warning(f"Experiment not found: {path_}")
        else:
            with open(path_) as f:
                exp_dict = json.load(f)

            self.name = exp_dict["name"]
            self.save_dir = exp_dict["save_dir"]
            for block_name in exp_dict["blocks"]:
                block = BlockRecord(None, None, None, None)
                block.load(join(path, block_name))
                self.register_block(block)
