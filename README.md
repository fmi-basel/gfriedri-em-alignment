gfriedri-em-alignment: stitching and alignment of  EM stacks
-----------------------------------------------------
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![tests](https://github.com/fmi-basel/gfriedri-em-alignment/workflows/tests/badge.svg)](https://github.com/fmi-basel/gfriedri-em-alignment/actions)
[![codecov](https://codecov.io/gh/fmi-basel/gfriedri-em-alignment/branch/main/graph/badge.svg)](https://app.codecov.io/gh/fmi-basel/gfriedri-em-alignment)

gfriedri-em-alignment is a tool that does stitching and alignment of tiled
volumentric electron microscopy (EM) data. It takes the images and metadata of
the EM stack, acquired using
the [SBEMimage](https://github.com/SBEMimage/SBEMimage) software, and uses
the [SOFIMA](https://github.com/google-research/sofima) tool to perform the
stitching and alignment.

# Installation

We recommend creating new conda environment with

```shell
  conda create --name sbem_stitching python=3.8
```

Then this package can be installed with

```shell
  pip install git+https://github.com/fmi-basel/gfriedri-em-alignment
```

Parts of [SOFIMA](https://github.com/google-research/sofima)
utilise [JAX](https://github.com/google/jax) to automatically take advantage of
GPU acceleration if the hardware is available. To enable JAX-GPU support follow
these [installation instructions](https://github.com/google/jax#pip-installation-gpu-cuda)
.

# Input data

The folder structure of the input data generated
by [SBEMimage](https://github.com/SBEMimage/SBEMimage) is described in
its [software documentation](https://sbemimage.readthedocs.io/en/latest/datasets.html).<br/>
Each experiment consists of one or many SBEMimage datasets, which is referred
to as blocks. Each block then contains several 2D sections that locates in a
range of z-positions. Each section contains a grid of tile images.<br/>
gfriedri-em-alignment will take the SBEMimage dataset folders as input and
parse the metadata into the Experiment -> Block -> Section -> Tile hierarchy
for downstream processing.

# Usage

- Parse SBEMimage metadata Before the stitching, the metadata created by
  SBEMimage needs to be converted to the tile ID maps that can be used as input
  metadata for [SOFIMA](https://github.com/google-research/sofima).

1. Create an Experiment instance

``` python
from sbem.experiment.Experiment import Experiment

# directory of the SBEMimage dataset (e.g. 2020101_sampleid_run01/)
exp_name = "Example"
save_dir = '/dir/to/which/this/exp/is/saved/'

# A new Experiment is created with:
exp = Experiment(name=exp_name,
                 save_dir=save_dir)

```

2. Add a block of sections to the experiment

``` python
sbem_root_dir = "/dir/containig/the/sbem/data/'

# Add a Block
exp.parse_block(sbem_root_dir=sbem_root_dir,
                name="example-block",
                tile_grid="g0001", # string that refers to the grid folder
                resolution_xy=11.0, # pixel size in nano-meters of the tiles
                tile_width=3072, # tile-width in pixels
                tile_height=2304, # tile-height in pixels
                tile_overlap=200) # tile-overlap in pixels

exp.save()
```

3. Stitch tiles in each section (2D stitching)
Fill out the [tile_registration.confg](./scripts/tile_registration.config).

Run tile-registration script:
```shell
python scripts/run_tile_registration.py --config scripts/tile_registration.config
```

Run warp-and-save script:
```shell
python scripts/run_warp_and_save.py --config scripts/tile_registration.config
```

__Note:__ If a SLURM cluster is available the `slurm_tile_registration.py`
script can be used as well. Make sure to set the SLURM parameters
accordingly in the [tile_registraion.config](./scripts/tile_registration.config).
The slurm-script will create batches of 75 sections and submit for every
batch two jobs. The first job (with a GPU) will execute
`run_tile_registration.py` and the second job (CPU only) will execute
`run_warp_and_save.py`. The second job is depending on the first job and
only started after it has finished.

- Align all sections across z-direction (3D alignment)

# License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this software except in compliance with the License. You may obtain a copy of
the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
