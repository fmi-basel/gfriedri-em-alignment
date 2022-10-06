from __future__ import annotations

import os
from os.path import join
from typing import TYPE_CHECKING

import zarr
from numpy.typing import ArrayLike
from ome_zarr.io import parse_url
from ruyaml import YAML

from sbem.record_v2.Author import Author
from sbem.record_v2.Citation import Citation
from sbem.record_v2.Info import Info

if TYPE_CHECKING:
    from typing import List


class Volume(Info):
    def __init__(
        self,
        name: str,
        description: str,
        documentation: str,
        authors: List[Author],
        root_dir: str,
        exist_ok: bool = False,
        license: str = "Creative Commons Attribution licence (CC BY)",
        cite: List[Citation] = [],
    ):
        super().__init__(name=name, license=license)
        self._description = description
        self._documentation = documentation
        self._authors = authors
        self._root_dir = root_dir
        self._cite = cite

        self._section_list = []

        self._data_path = join(self._root_dir, self.get_name(), "ngff_volume.zarr")

        if self._root_dir is not None:
            os.makedirs(self._data_path, exist_ok=exist_ok)

        store = parse_url(self._data_path, mode="w").store
        self.zarr_root = zarr.group(store=store)
        # write_image(image=np.zeros((1, 1960, 1960), dtype=np.uint8),
        #             group=self.zarr_root,
        #             scaler=Scaler(max_layer=2),
        #             axes="zyx",
        #             storage_options=dict(
        #                 chunks=(1, 2744, 2744),
        #                 compressor=Blosc(cname="zstd", clevel=3,
        #                                  shuffle=Blosc.SHUFFLE),
        #                 overwrite=True
        #             ))

        self.save()

    def write_section(self, section_num: int, data: ArrayLike):
        # manually go through all scale levels
        # insert should be possible with moving dirs on filesystem
        # remove should also work with moving dirs on filesystem
        # resize multiscale by chunk-size only.
        pass

    def to_dict(self):
        return {
            "name": self.get_name(),
            "root_dir": self._root_dir,
            "license": self.get_license(),
            "format_version": self.get_format_version(),
            "description": self._description,
            "documentation": self._documentation,
            "authors": [a.to_dict() for a in self._authors],
            "cite": [c.to_dict() for c in self._cite],
            "data": self._data_path,
            "sections": self._section_list,
        }

    def _dump(self, out_path: str):
        yaml = YAML(typ="rt")
        with open(join(out_path, "volume.yaml"), "w") as f:
            yaml.dump(self.to_dict(), f)

    def save(self):
        out_path = join(self._root_dir, self.get_name())
        os.makedirs(out_path, exist_ok=True)
        self._dump(out_path=out_path)

    @staticmethod
    def load(path: str) -> Volume:
        yaml = YAML(typ="rt")
        with open(path) as f:
            data = yaml.load(f)

        return Volume(
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
