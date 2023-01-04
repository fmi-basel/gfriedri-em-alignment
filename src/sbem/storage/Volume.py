from __future__ import annotations

import json
import logging
import os
from glob import glob
from math import ceil
from os.path import exists, join, split
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

from sbem.record.Author import Author
from sbem.record.Citation import Citation
from sbem.record.Info import Info
from sbem.record.ReferenceMixin import ReferenceMixin

if TYPE_CHECKING:  # pragma: no cover
    from typing import List


class Volume(ReferenceMixin, Info):
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
        save=True,
    ):
        super().__init__(name=name, license=license, authors=authors, cite=cite)
        self._description = description
        self._documentation = documentation
        self._root_dir = root_dir
        self.logger = logger

        self._section_list = []
        self._section_offset_map = {}
        self._section_shape_map = {}
        self._origin = np.array([0, 0, 0], dtype=int)

        self._data_path = join(self._root_dir, self.get_name(), "ngff_volume.zarr")

        if self._root_dir is not None:
            os.makedirs(self._data_path, exist_ok=exist_ok)

        store = parse_url(self._data_path, mode="w").store
        self.zarr_root = zarr.group(store=store)
        self.scaler = Scaler(max_layer=0)

        if save:
            self.save()

    def remove_section(self, section_num: int):
        index = self._section_list.index(section_num)
        dir_name = join(self.zarr_root.chunk_store.dir_path(), "0")
        rmtree(join(dir_name, str(index)))
        shape = self.zarr_root["0"].shape
        for z in range(index + 1, shape[0]):
            src = join(dir_name, str(z))
            dst = join(dir_name, str(z - 1))
            if exists(src):
                move(src, dst)
                self._section_offset_map[self._section_list[z]][0] -= 1

        new_shape = [shape[0] - 1, shape[1], shape[2]]
        self._reshape_multiscale_level(new_shape, self.zarr_root["0"])
        self._section_list.remove(section_num)
        self._section_offset_map.pop(section_num)
        self._section_shape_map.pop(section_num)

    def append_section(
        self,
        section_num: int,
        data: ArrayLike,
        relative_offsets: Tuple[int, int, int] = tuple([1, 0, 0]),
    ):
        if len(self._section_list) == 0:
            self.write_section(section_num=section_num, data=data)
        else:
            previous_section_num = None
            i = -1
            while previous_section_num is None:
                previous_section_num = self._section_list[i]
                i -= 1
            previous_offsets = self._section_offset_map[previous_section_num]
            total_offsets = tuple(
                [
                    int(previous_offsets[0] + relative_offsets[0]),
                    int(previous_offsets[1] + relative_offsets[1]),
                    int(previous_offsets[2] + relative_offsets[2]),
                ]
            )
            self.write_section(
                section_num=section_num, data=data, offsets=total_offsets
            )

    def write_section(
        self,
        section_num: int,
        data: ArrayLike,
        offsets: Tuple[int, int, int] = tuple([0, 0, 0]),
    ):
        """

        :param section_num:
        :param data:
        :param offsets: to volume origin
        :return:
        """
        assert section_num not in self._section_list, (
            f"Section " f"{section_num} exists already."
        )
        assert offsets[0] >= 0, "Z offset has to be >= 0."
        self._section_offset_map[section_num] = np.array(offsets)
        self._section_shape_map[section_num] = data.shape
        # insert should be possible with moving dirs on filesystem
        if len(self._section_list) == 0:
            assert len(data.shape) == 3
            write_image(
                image=data,
                group=self.zarr_root,
                scaler=self.scaler,
                axes="zyx",
                storage_options=dict(
                    chunks=(1, 2744, 2744),
                    compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
                    overwrite=True,
                ),
            )
        else:
            if offsets[0] >= len(self._section_list):
                # append in Z
                self._reshape_storage(offsets, data.shape)

                data = self._pad_data(offsets, data)

                slices = self._compute_slices(offsets, data.shape)

                self.zarr_root["0"][tuple(slices)] = data
            else:
                # insert into stack
                self._reshape_storage(
                    offsets=tuple([len(self._section_list), offsets[1], offsets[2]]),
                    shape=data.shape,
                )

                data = self._pad_data(offsets, data)

                slices = self._compute_slices(offsets, data.shape)

                # move slices above
                for z in range(len(self._section_list), offsets[0], -1):
                    src = join(self.zarr_root.chunk_store.dir_path(), "0", str(z - 1))
                    dst = join(self.zarr_root.chunk_store.dir_path(), "0", str(z))
                    move(src, dst)
                    self._section_offset_map[self._section_list[z - 1]][0] += 1

                self.zarr_root["0"][tuple(slices)] = data

        for i in range(len(self._section_list), offsets[0]):
            self._section_list.insert(i, None)
        self._section_list.insert(offsets[0], section_num)

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

    def _reshape_multiscale_level(self, new_shape, level):
        with open(
            join(self.zarr_root.chunk_store.dir_path(), level.basename, ".zarray"),
        ) as f:
            array_dict = json.load(f)
        level.shape = new_shape
        # Overwrite json otherwise "dimension_separator" gets dropped.
        with open(
            join(self.zarr_root.chunk_store.dir_path(), level.basename, ".zarray"),
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
            "offsets": {k: v.tolist() for k, v in self._section_offset_map.items()},
            "shapes": self._section_shape_map,
            "origin": [int(o) for o in self._origin],
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

        vol = Volume(
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
        return vol

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
                self._update_origin(axis=i, shift=abs(n_chunks) * chunk_size)

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
            self._reshape_multiscale_level(new_shape, storage)

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
        for offset, size in zip(offsets, shape):
            if offset < 0:
                slices.append(slice(0, size))
            else:
                slices.append(slice(offset, offset + size))
        return slices

    def _update_origin(self, axis, shift):
        self._origin[axis] += shift
        for offsets in self._section_offset_map.values():
            offsets[axis] += shift

    def get_section_origin(self, section_num: int):
        return self._origin + self._section_offset_map[section_num]

    def get_section_data(self, section_num: int):
        z, y, x = self._section_offset_map[section_num]
        zs, ys, xs = self._section_shape_map[section_num]
        return self.get_zarr_volume()["0"][z : z + zs, y : y + ys, x : x + xs]

    def get_description(self):
        return self._description

    def get_documentation(self):
        return self._documentation

    def get_dir(self):
        return join(self._root_dir, self.get_name())

    def get_origin(self):
        return self._origin

    def get_zarr_volume(self):
        return self.zarr_root
