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


## 1. Create an Experiment instance
Create a new `em_alignment` experiment. An experiment can contain mulitple
samples and each sample can contain multiple sections.

```python
from sbem.experiment.Experiment import Experiment
from sbem.record.Author import Author
from sbem.record.Citation import Citation

# directory of the SBEMimage dataset (e.g. 2020101_sampleid_run01/)

# A new Experiment is created with:
exp = Experiment(
        name="Experiment",
        description="A short description.",
        documentation="README.md",
        authors=[Author(name="", affiliation="")],
        root_dir="/dir/to/which/this/exp/is/saved/",
        exist_ok=True,
        cite=[Citation(doi="", text="", url="")],
    )
    exp.save()
```


## 2. Add a sample to the experiment
A sample corresponds to a whole imaged object and can consist of many
thousand sections. The acquired sections can come from different
acquisition runs.

```python
from sbem.record.Sample import Sample
Sample(
    experiment=exp,
    name="Sample_00",
    description="",
    documentation="",
    aligned_data="",
)

exp.save(overwrite=True)
```

## 3. Add sections to a sample
In this step the medatadata from a SBEM acquisition is parsed and the
sections and their tiles are added to the sample.

```python
from sbem.experiment.parse_utils import parse_and_add_sections

sample = exp.get_sample("Sample_00")

parse_and_add_sections(
    sbem_root_dir="/path/to/sbem/acquisition_dir",
    sample=sample,
    acquisition="run_0", # name of the acquisition run for these sections
    tile_grid="g0001", # string that refers to the grid folder
    thickness=25.0, # section thickness nano-meters
    resolution_xy=11.0, # pixel size in nano-meters of the tiles
    tile_width=3072, # tile-width in pixels
    tile_height=2304, # tile-height in pixels
    tile_overlap=200) # tile-overlap in pixels
    overwrite=True,
)

exp.save(overwrite=True)
```


# License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this software except in compliance with the License. You may obtain a copy of
the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
