import json
from glob import glob
from logging import Logger
from os import mkdir
from os.path import basename, exists, join

from sbem.experiment.parse_utils import get_tile_metadata
from sbem.record import BlockRecord, SectionRecord, TileRecord
from tqdm import tqdm


class Experiment:
    def __init__(self, name: str = None, sbem_run_path: str = None):
        self.logger = Logger(f"Experiment[{name}]")
        self.name = name

        if sbem_run_path is not None:
            assert exists(sbem_run_path), f"{sbem_run_path} does not exist."
        self.sbem_run_path = sbem_run_path

        self.blocks = {}

    def register_block(self, block: BlockRecord):
        self.blocks[block.block_id] = block

    def add_block(self, path: str, tile_grid: str, resolution_xy: float):
        block_id = basename(path)
        block = BlockRecord(self, block_id=block_id)

        tile_grid_num = int(tile_grid[1:])

        metadata_files = sorted(glob(join(path, "meta", "logs", "metadata_*")))

        tile_specs = get_tile_metadata(
            self.sbem_run_path, metadata_files, tile_grid_num, resolution_xy
        )

        for tile_spec in tqdm(tile_specs, desc="Build Block Record"):
            section = block.get_section(tile_spec["z"], tile_grid_num)
            if section is None:
                section = SectionRecord(block, tile_spec["z"], tile_grid_num)

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

                for section in block.sections.values():
                    section.compute_tile_id_map()

    def save(self, path):
        if exists(path):
            self.logger.warning("Experiment already exists.")
        else:
            mkdir(path)
            exp_dict = {
                "name": self.name,
                "sbem_run_path": self.sbem_run_path,
                "n_blocks": len(self.blocks),
                "blocks": list(self.blocks.keys()),
            }
            with open(join(path, "experiment.json"), "w") as f:
                json.dump(exp_dict, f, indent=4)

            for block_name, block in self.blocks.items():
                block_dir = join(path, block_name)
                block.save(block_dir)

    def load(self, path):
        path_ = join(path, "experiment.json")
        if not exists(path_):
            self.logger.warning(f"Experiment not found: {path_}")
        else:
            with open(path_) as f:
                exp_dict = json.load(f)

            self.name = exp_dict["name"]
            self.sbem_run_path = exp_dict["sbem_run_path"]
            for block_name in exp_dict["blocks"]:
                block = BlockRecord(self, block_name)
                block.load(join(path, block_name))
