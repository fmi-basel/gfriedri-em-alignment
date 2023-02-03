import os
import json
import numpy as np
import tensorstore as ts
from numpy.typing import ArrayLike
from ruyaml import YAML
from numcodecs import Blosc
from ome_zarr.writer import write_image
from sbem.record.Author import Author
from sbem.record.Citation import Citation
from sbem.storage.Volume import Volume
from connectomics.common import bounding_box

class VolumeTs(Volume):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    @staticmethod
    async def load(path: str):
        yaml = YAML(typ="rt")
        with open(path) as f:
            data = yaml.load(f)

        vol = VolumeTs(
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
            save=False,
        )
        vol._section_list = data["sections"]
        vol._section_offset_map = {k: np.array(v) for k, v in data["offsets"].items()}
        vol._section_shape_map = {k: tuple(v) for k, v in data["shapes"].items()}
        vol._origin = np.array(data["origin"])
        await vol.open_dataset_read()
        return vol


    def get_dataset_path(self):
        return os.path.join(self._data_path, "0")


    async def open_dataset_read(self):
        dataset_future = ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': self.get_dataset_path()
            },
        },read=True)
        self.dataset_read = await dataset_future


    async def open_dataset_write(self):
        dataset_future = ts.open({
        "driver": "zarr",
        "kvstore": {
            "driver": "file",
            "path": self.get_dataset_path()
            },
        })
        self.dataset_write = await dataset_future


    def estimate_volume_size(self, margin_xy=[0,0]):
        offsets = np.array([v.tolist() for v in self._section_offset_map.values()])
        shapes = np.array([v for v in self._section_shape_map.values()])

        max_zyx = offsets.max(axis=0)
        shape_max = shapes.max(axis=0)
        width = shape_max[2] + max_zyx[2] + margin_xy[0]
        height = shape_max[1] + max_zyx[1] + margin_xy[1]
        depth = max_zyx[0] + 1
        volume_size = list(map(int, [depth, height, width]))
        return volume_size

    async def create_dataset(self, chunks: list = [1, 2744, 2744],
                             delete_existing=False):
        """
        Create a zarr volume

        :param chunk_size
        :param resolution
        """
        volume_size = self.estimate_volume_size()

        # Use ome-zarr to write the first image
        # so that the metadata is initiated
        write_image(
            image=np.zeros((100, 100, 100), dtype=np.uint8),
            group=self.zarr_root,
            scaler=self.scaler,
            axes="zyx",
            storage_options=dict(
                chunks=chunks,
                compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
                overwrite=delete_existing,
                ))
        self.zarr_root["0"].resize(volume_size)

        metafile = os.path.join(self.get_dataset_path(), ".zarray")
        with open(metafile, "r") as f:
            array_dict = json.load(f)

        array_dict["dimension_separator"] =  "/"

        with open(metafile, "w") as f:
            array_dict = json.dump(array_dict, f, indent=4)

        await self.open_dataset_write()


    def get_section_bounding_box(self, section_num:int):
        z, y, x = self.get_section_origin(section_num)
        zs, ys, xs = self._section_shape_map[section_num]
        sbox = bounding_box.BoundingBox(start=(z, y, x),
                                        size=(zs, ys, xs))
        return sbox


    async def write_section(self, section_num:int, img: ArrayLike):
        sbox = self.get_section_bounding_box(section_num)
        self.dataset_write[sbox.start[0]:sbox.end[0],
                                 sbox.start[1]:sbox.end[1],
                                 sbox.start[2]:sbox.end[2]] = img


    async def get_section_data(self, section_num: int):
        sbox = self.get_section_bounding_box(section_num)
        return self.dataset_read[sbox.start[0]:sbox.end[0],
                                 sbox.start[1]:sbox.end[1],
                                 sbox.start[2]:sbox.end[2]]


    def set_sections(self, sections, xy_coords):
        for k, section in enumerate(sections):
            section_num = section["section_num"]
            self._section_list.append(section_num)
            self._section_offset_map[section_num] = np.array([k, xy_coords[k][1],
                                                        xy_coords[k][0]])
            self._section_shape_map[section_num] = section["shape"]
