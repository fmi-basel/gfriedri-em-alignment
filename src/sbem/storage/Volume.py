from __future__ import annotations

import json
import logging
import os
from glob import glob
from math import ceil
from os.path import join, split
from shutil import move, rmtree
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

    def remove_section(self, section_num: int):
        index = self._section_list.index(section_num)
        dir_name = join(self.zarr_root.chunk_store.dir_path(), "0")
        rmtree(join(dir_name, str(index)))
        shape = self.zarr_root["0"].shape
        new_shape = [shape[0] - 1, shape[1], shape[2]]
        self.reshape_zlevel(new_shape, self.zarr_root["0"])
        for z in range(index + 1, shape[0]):
            src = join(dir_name, str(z))
            dst = join(dir_name, str(z - 1))
            move(src, dst)

    def append_section(
        self,
        section_num: int,
        data: ArrayLike,
        relative_offsets: Tuple[int] = tuple([1, 0, 0]),
    ):
        # Need to store offsets of all sections relative to the first section
        # Get offset of previous section and compute offset of this section
        # Add offsets to section_list?
        pass

    def write_section(
        self, section_num: int, data: ArrayLike, offsets: Tuple[int] = tuple([0, 0, 0])
    ):
        """

        :param section_num:
        :param data:
        :param offsets: With respect to top-most z-slices self._section_list[
        -1]
        :return:
        """
        # insert should be possible with moving dirs on filesystem
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
        else:
            self._reshape_storage(offsets, data.shape)

            data = self._pad_data(offsets, data)

            slices = self._compute_slices(offsets, data.shape)

            self.zarr_root["0"][tuple(slices)] = data

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

    def _reshape_storage(self, offsets, shape):
        storage = self.zarr_root["0"]
        new_shape = []
        for i, (offset, chunk_size, storage_size, data_size) in enumerate(
            zip(offsets, storage.chunks, storage.shape, shape)
        ):
            new_size = storage_size
            # extend before
            if offset < 0:
                n_chunks = offset // chunk_size
                self._extend(n_chunks=n_chunks, axis=i, z_level=storage)
                new_size += abs(n_chunks) * chunk_size

            overhang = offset + data_size - storage_size
            total_chunk_space = ceil(storage_size / chunk_size) * chunk_size
            remaining_space = total_chunk_space - storage_size

            # extend after
            if overhang > 0:
                if chunk_size == 1:
                    n_chunks = (overhang - remaining_space) // chunk_size
                else:
                    n_chunks = (overhang - remaining_space) // chunk_size + 1

                self._extend(n_chunks=n_chunks, axis=i, z_level=storage)
                new_size += overhang

            new_shape.append(new_size)

        if tuple(new_shape) != storage.shape:
            print(f"new_shape = {new_shape}")
            self.reshape_zlevel(new_shape, storage)

    def _pad_data(self, offsets, data):
        storage = self.zarr_root["0"]

        padding = []
        for i, (offset, chunk_size, storage_size, data_size) in enumerate(
            zip(offsets, storage.chunks, storage.shape, data.shape)
        ):
            if offset < 0:
                pad_before = chunk_size - (abs(offset) % chunk_size)
                padding.append([pad_before, 0])
            else:
                padding.append([0, 0])

        return np.pad(data, padding)

    def _compute_slices(self, offsets, shape):
        slices = []
        print(shape)
        for offset, size in zip(offsets, shape):
            print(offset, size)
            if offset < 0:
                slices.append(slice(0, size))
            else:
                slices.append(slice(offset, offset + size))
        print(slices)
        print(self.zarr_root["0"].shape)
        return slices
