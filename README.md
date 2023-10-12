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
  conda create --name sbem_stitching python=3.9
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
its [software documentation](https://sbemimage.readthedocs.io/en/latest/datasets.html). <br/>
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

# License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this software except in compliance with the License. You may obtain a copy of
the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
