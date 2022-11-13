import asyncio
import tensorstore as ts
from sbem.render_volume.render_utils import (
    open_volume, create_volume_n5
    )

async def main(volume_path, scale_key,
               out_volume_path, out_chunk_size,
               downsample_factors=None,
               downsample_method="mean"):
    volume = await open_volume(volume_path, scale_key=scale_key)
    if downsample_factors:
        volume_in = ts.downsample(volume, downsample_factors, downsample_method)
    else:
        volume_in = volume
    out_volume = await create_volume_n5(out_volume_path,
                                        volume_in.shape[:-1],
                                        out_chunk_size)

    await out_volume[...].write(volume_in[..., 0])


if __name__ == "__main__":
    volume_path = "/tungstenfs/scratch/gfriedri/hubo/em_alignment/results/sbem_experiments/20220524_Bo_juv20210731/volume/ob_substack.precomputed"
    # scale_key = "176_176_528"
    scale_key = "88_88_264"
    downsample_factors = [3, 3, 1, 1]
    out_volume_path = "/tungstenfs/scratch/gfriedri/hubo/em_alignment/results/sbem_experiments/20220524_Bo_juv20210731/volume/ob_substack_264nm.n5"
    out_chunk_size = [512, 512, 512]

    asyncio.run(main(volume_path, scale_key, out_volume_path, out_chunk_size,
                     downsample_factors))
