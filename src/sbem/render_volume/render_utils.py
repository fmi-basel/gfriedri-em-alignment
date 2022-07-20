import os
import asyncio
import logging
import tensorstore as ts

from sbem.recorde import SectionRecord

async def read_n5(path: str):
    """
    Read N5 dataset

    :param path: path to the N5 dataset

    :return: `tensorstore.TensorStore`
    """
    dataset_future = ts.open({
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': path,
            }
        },read=True)
    dataset = await dataset_future
    return dataset


async def read_stitched(section: SectionRecord):
    """
    Read stitched image of a section

    :param section: `SectionRecord`

    :return: `tensorstore.TensorStore`
    """

    stitched_path = section.get_stitched_path()
    stitched = await read_n5(stitched_path)
    return stitched


async def read_stitched_sections(sections: List(SectionRecord)):
    stitched_sections = []

    for section in sections:
        stitched = await read_stitched(section)
        stitched_sections.append(stitched)

    return stitched_sections


def get_sharding_spec():
    sharding_spec =  {
        "@type": "neuroglancer_uint64_sharded_v1",
        "data_encoding": "gzip",
        "hash": "identity",
        "minishard_bits": 6,
        "minishard_index_encoding": "gzip",
        "preshift_bits": 9,
        "shard_bits": 15
        }
    return sharding_spec


async def create_volume(path: str,
                        size: list,
                        chunk_size: list,
                        resolution: list,
                        sharding: bool=True,
                        sharding_spec: dict=get_sharding_spec()):
    """
    Create a multiscale neuroglancer_precomputed volume
    with a specified single scale

    :param path
    :param size
    :param chunk_size
    :param resolution
    :param sharding
    """
    volume_spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": {"driver": "file",
                    "path": path
                    },
        "multiscale_metadata": {
            "type": "image",
            "data_type": "uint8",
            "num_channels": 1
            },
        "scale_metadata": {
            "size": size,
            "encoding": "raw",
            "chunk_size": chunk_size,
            "resolution": resolution,
            },
        "create": True,
        "delete_existing": True
        }

    if sharding:
        volume_spec["scale_metadata"]["sharding"] = sharding_spec

    volume_future = ts.open(volume_spec)

    volume = await volume_future
    return volume


async def estimate_volume_size(stitched_sections, xy_coords):
    n_sections = len(stitched_sections)
    max_xy = xy_coords.max(axis=0)
    shape_list = [s.shape for s in stitched_sections]
    width = max([shape_list[0]]) + max_xy[0]
    height = max([shape_list[1]]) + max_xy[1]
    volume_size = [width, height, n_sections]
    return volume_size


async def render_volume(volume_path, sections, xy_coords,
                        chunk_size: list=[64, 64, 64]):
    """
    Render a range of sections into a 3d wolume
    volume_path: the path for writing the volume.
    sections: a list of SectionRecord objects
    xy_coords: not used at the moment

    #TODO make sure that xy_coords are non-negative
    """
    logger = logging.getLogger(__name__)

    stitched_sections = await read_stitched_sections(sections)
    volume_size = await estimate_volume_size(stitched_sections, xy_coords)
    volume = await create_volume(volume_path, volume_size, chunk_size,
                                 resolution, sharding=True)

    for k, section in tqdm(enumerate(sections), "Writing sections"):
        stitched = stitched_sections[k]
        xyo = xy_coords[k]
        await volume[xyo[0]:stitched.shape[0]+xyo[0],
                     xyo[1]:stitched.shape[1]+xyo[1],
                     k, 0].write(stitched)

    return volume
