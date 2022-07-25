import os
import asyncio
from sbem.render_volume.multiscale import make_multiscale

if __name__ == "__main__":
    sbem_experiment="/tungstenfs/scratch/gfriedri/hubo/em_alignment/results/sbem_experiments/20220524_Bo_juv20210731"
    grid_index=1
    start_section=6250
    end_section=6752
    volume_name = f"s{start_section}_s{end_section}"
    volume_path = os.path.join(sbem_experiment, "volume", volume_name+"_ng")

    n_scales = 6
    downsample_factors_list = [[2, 2, 2, 1]] * n_scales
    chunk_size_list = [[64, 64, 64]] * n_scales
    # base_scale_key="44_44_132"
    # base_scale_key="176_176_528"
    base_scale_key=None
    asyncio.run(make_multiscale(volume_path,
                                downsample_factors_list,
                                chunk_size_list,
                                base_scale_key=base_scale_key,
                                downsample_method="stride"))
