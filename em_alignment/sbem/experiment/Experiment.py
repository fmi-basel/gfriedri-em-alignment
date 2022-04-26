import json
from glob import glob
from logging import Logger
from os import mkdir
from os.path import basename, exists, join

from sbem.experiment.parse_utils import get_tile_metadata
from sbem.record.BlockRecord import BlockRecord
from sbem.record.SectionRecord import SectionRecord
from sbem.record.TileRecord import TileRecord
from tqdm import tqdm


class Experiment:
    """
    from sbem.experiment import Experiment
    exp = Experiment("20210630_Dp_190326Bb_run04", "/tungstenfs/landing/gmicro_sem/gemini_data/20210630_Dp_190326Bb_run04")
    exp.add_block("/tungstenfs/landing/gmicro_sem/gemini_data"
                  "/20210630_Dp_190326Bb_run04", "g0001", 11)
    exp.save("/home/tibuch/Data/gfriedri/20210630_Dp_190326Bb_run04")
    exp.load("/home/tibuch/Data/gfriedri/20210630_Dp_190326Bb_run04")

    """

    def __init__(
        self, name: str = None, sbem_run_path: str = None, save_dir: str = None
    ):
        self.logger = Logger(f"Experiment[{name}]")
        self.name = name

        if sbem_run_path is not None:
            assert exists(sbem_run_path), f"{sbem_run_path} does not exist."
        self.sbem_run_path = sbem_run_path

        if save_dir is not None:
            assert exists(save_dir), f"{save_dir} does not exist."
            self.save_dir = join(save_dir, self.name)
        else:
            self.save_dir = None

        self.blocks = {}

    def register_block(self, block: BlockRecord):
        self.blocks[block.block_id] = block

    def add_block(self, path: str, tile_grid: str, resolution_xy: float):
        block_id = basename(path)
        block = BlockRecord(self, block_id=block_id, save_dir=self.save_dir)

        tile_grid_num = int(tile_grid[1:])

        metadata_files = sorted(glob(join(path, "meta", "logs", "metadata_*")))

        tile_specs = get_tile_metadata(
            self.sbem_run_path, metadata_files, tile_grid_num, resolution_xy
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
        assert self.save_dir is not None, "Save directory not set."
        mkdir(self.save_dir)
        exp_dict = {
            "name": self.name,
            "save_dir": self.save_dir,
            "sbem_run_path": self.sbem_run_path,
            "n_blocks": len(self.blocks),
            "blocks": list(self.blocks.keys()),
        }
        with open(join(self.save_dir, "experiment.json"), "w") as f:
            json.dump(exp_dict, f, indent=4)

        for block_name, block in self.blocks.items():
            block.save()

    def load(self, path):
        path_ = join(path, "experiment.json")
        if not exists(path_):
            self.logger.warning(f"Experiment not found: {path_}")
        else:
            with open(path_) as f:
                exp_dict = json.load(f)

            self.name = exp_dict["name"]
            self.save_dir = exp_dict["save_dir"]
            self.sbem_run_path = exp_dict["sbem_run_path"]
            for block_name in exp_dict["blocks"]:
                block = BlockRecord(self, block_name, self.save_dir)
                block.load(join(path, block_name))
