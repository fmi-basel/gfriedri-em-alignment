from __future__ import annotations

import json
import logging
import os
from glob import glob
from math import ceil
from os.path import join, split
from shutil import move
from typing import TYPE_CHECKING, Tuple

import numpy as np
import zarr
from numcodecs import Blosc
from numpy.typing import ArrayLike
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
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
        logger=logging,
    ):
        super().__init__(name=name, license=license)
        self._description = description
        self._documentation = documentation
        self._authors = authors
        self._root_dir = root_dir
        self._cite = cite
        self.logger = logger

        self._section_list = []

        self._data_path = join(self._root_dir, self.get_name(), "ngff_volume.zarr")

        if self._root_dir is not None:
            os.makedirs(self._data_path, exist_ok=exist_ok)

        store = parse_url(self._data_path, mode="w").store
        self.zarr_root = zarr.group(store=store)
        self.scaler = Scaler(max_layer=0)

        self.save()

    def write_section(
        self, section_num: int, data: ArrayLike, offset: Tuple[int] = tuple([0, 0, 0])
    ):
        """

        :param section_num:
        :param data:
        :param offset: With respect to top-most z-slices self._section_list[
        -1]
        :return:
        """
        # insert should be possible with moving dirs on filesystem
        # remove should also work with moving dirs on filesystem
        # resize multiscale by chunk-size only.
        if len(self._section_list) == 0:
            assert len(data.shape) == 3
            chunks = (1, 1000, 1000)
            print(chunks)
            write_image(
                image=data,
                group=self.zarr_root,
                scaler=self.scaler,
                axes="zyx",
                storage_options=dict(
                    chunks=chunks,
                    compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
                    overwrite=True,
                ),
            )
            self._section_list.append(section_num)
        else:
            # scale_level = self.zarr_root.attrs["multiscales"][0][
            #     "datasets"]
            z_level = self.zarr_root[0]
            chunk_size = z_level.chunks
            print(f"chunk_size = {chunk_size}")

            slices = []
            for i, (o, cs, current_shape, new_shape) in enumerate(
                zip(offset, chunk_size, z_level.shape, data.shape)
            ):

                insert_start = o
                if o < 0:
                    n_chunks = o // cs
                    self._extend(n_chunks=n_chunks, axis=i, z_level=z_level)
                    insert_start = 0

                overhang = o + new_shape - current_shape
                space_left = ceil(current_shape / cs) * cs - current_shape
                print(f"overhang = {overhang}")
                print(f"space_left = {space_left}")
                if overhang > 0:
                    if cs == 1:
                        n_chunks = (overhang - space_left) // cs
                    else:
                        n_chunks = (overhang - space_left) // cs + 1
                    print(f"Extend axis={i} by {n_chunks}.")
                    self._extend(n_chunks=n_chunks, axis=i, z_level=z_level)

                if o >= 0:
                    insert_end = insert_start + new_shape
                else:
                    insert_end = insert_start + new_shape + cs + o
                slices.append(slice(insert_start, insert_end))

            print(slices)
            print(data.shape)
            print(z_level.shape)
            new_shape = []
            pad = []
            for o, cs, ns, zs, s in zip(
                offset, chunk_size, data.shape, z_level.shape, slices
            ):
                if o < 0:
                    if zs >= s.stop:
                        new_shape.append(zs + cs)  # multiply cs by
                        # n_chunks
                        pad.append([cs + o, 0])
                    else:
                        new_shape.append(cs + zs)
                        pad.append([cs + o, 0])
                else:
                    if zs >= s.stop:
                        new_shape.append(zs)
                        pad.append([0, 0])
                    else:
                        new_shape.append(s.stop)
                        pad.append([0, 0])

            print(f"new_shape = {new_shape}")
            self.reshape_zlevel(new_shape, z_level)
            padded = np.pad(data, pad_width=pad)
            print(f"padding = {pad}")
            print(padded.shape)
            z_level[tuple(slices)] = padded

        self._section_list.append(section_num)

    def _extend(self, n_chunks, axis, z_level):
        if n_chunks < 0:
            # prepend
            if axis == 0:
                chunks = glob(
                    join(self.zarr_root.chunk_store.dir_path(), z_level.basename, "*")
                )
            elif axis == 1:
                chunks = glob(
                    join(
                        self.zarr_root.chunk_store.dir_path(),
                        z_level.basename,
                        "*",
                        "*",
                    )
                )
            elif axis == 2:
                chunks = glob(
                    join(
                        self.zarr_root.chunk_store.dir_path(),
                        z_level.basename,
                        "*",
                        "*",
                        "*",
                    )
                )
            else:
                raise RuntimeError("Axis must be in [0, 2].")
            chunks.sort(reverse=True)
            for c in chunks:
                dir_name, chunk = split(c)
                move(c, join(dir_name, str(int(chunk) + abs(n_chunks))))

    def reshape_zlevel(self, new_shape, z_level):
        with open(
            join(self.zarr_root.chunk_store.dir_path(), z_level.basename, ".zarray"),
        ) as f:
            array_dict = json.load(f)
        z_level.shape = new_shape
        # Overwrite json otherwise "dimension_separator" gets dropped.
        with open(
            join(self.zarr_root.chunk_store.dir_path(), z_level.basename, ".zarray"),
            "w",
        ) as f:
            array_dict["shape"] = new_shape
            json.dump(array_dict, f, indent=4)

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
