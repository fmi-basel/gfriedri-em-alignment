import os
import asyncio
from sbem.render_volume.multiscale import make_multiscale

if __name__ == "__main__":
    sbem_experiment="/tungstenfs/scratch/gfriedri/hubo/em_alignment/results/sbem_experiments/20220524_Bo_juv20210731"
    volume_name = f"ob_substack.precomputed"
    volume_path = os.path.join(sbem_experiment, "volume", volume_name)

    n_scales = 6
    downsample_factors_list = [[2, 2, 2, 1]] * n_scales
    chunk_size_list = [[64, 64, 64]] * n_scales
    # base_scale_key="44_44_132"
    # base_scale_key="176_176_528"

    sharding_list = [(9, 6, 10), (9, 6, 7), (9, 6, 4),
                     (6, 6, 4), (3, 3, 7), (3, 3, 4)]
    base_scale_key=None

    asyncio.run(make_multiscale(volume_path,
                                downsample_factors_list,
                                chunk_size_list,
                                sharding_list=sharding_list,
                                base_scale_key=base_scale_key,
                                downsample_method="mean"))
