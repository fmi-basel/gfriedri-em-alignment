import json
import shutil
import tempfile
from os.path import exists, join
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal
from ruyaml import YAML
from tifffile import imsave

from sbem.experiment.Experiment_v2 import Experiment
from sbem.record_v2.Author import Author
from sbem.record_v2.Citation import Citation
from sbem.record_v2.Sample import Sample
from sbem.record_v2.Section import Section
from sbem.record_v2.Tile import Tile


class ExperimentTest(TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

        self.img = np.random.randint(0, 3200, size=(1320, 3021), dtype=np.uint16)

        imsave(join(self.tmp_dir, "img.tif"), self.img)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_experiment(self):
        name = "exp"
        description = "description"
        documentation = "documentation"
        authors = [
            Author("author 1", "aff 1"),
            Author("author 2", "aff 2"),
            Author("author 3", "aff 1"),
        ]
        root_dir = join(self.tmp_dir, "my_experiments")
        license = "A license"
        cite = [Citation("doi", "text", "url")]

        exp = Experiment(
            name, description, documentation, authors, root_dir, False, license, cite
        )

        assert exists(root_dir)
        assert exp.get_name() == name
        assert exp.get_format_version() == "0.1.0"
        assert exp.get_description() == description
        assert exp.get_documentation() == documentation
        assert (
            exp.get_authors_pretty() == "author 1[1], author 2[2], "
            "author 3[1]\n[1] aff 1\n[2] aff 2"
        )
        assert exp.get_citations_pretty() == "text\nDOI: doi\nURL: url"
        assert exp.get_root_dir() == root_dir
        assert exp.get_sample("sample") is None

        dict = exp.to_dict()
        assert dict["name"] == name
        assert dict["description"] == description
        assert dict["documentation"] == documentation
        assert dict["root_dir"] == root_dir
        assert dict["license"] == license
        assert dict["authors"] == [
            {"affiliation": "aff 1", "name": "author 1"},
            {"affiliation": "aff 2", "name": "author 2"},
            {"affiliation": "aff 1", "name": "author 3"},
        ]
        assert dict["cite"] == [{"doi": "doi", "text": "text", "url": "url"}]
        assert dict["format_version"] == "0.1.0"
        assert dict["samples"] == []

        sample = Sample(exp, "sample", "desc", "docu", "./data")
        dict = exp.to_dict()
        assert dict["samples"] == ["sample"]

        assert exp.get_sample("sample") == sample

        sample_1 = Sample(None, "sample_1", "desc", "docu", "./data")
        exp.add_sample(sample_1)
        assert sample_1.get_experiment() == exp

    def test_save_load(self):
        name = "exp"
        description = "description"
        documentation = "documentation"
        authors = [
            Author("author 1", "aff 1"),
            Author("author 2", "aff 2"),
            Author("author 3", "aff 1"),
        ]
        root_dir = join(self.tmp_dir, "my_experiments")
        license = "A license"
        cite = [Citation("doi", "text", "url")]

        exp = Experiment(
            name, description, documentation, authors, root_dir, True, license, cite
        )
        Sample(exp, "sample", "desc", "docu", "./data")

        exp.save(overwrite=False, section_to_subdir=True)
        assert exists(join(root_dir, "exp", "experiment.yaml"))
        assert exists(join(root_dir, "exp", "sample", "sample.yaml"))

        yaml = YAML(typ="rt")
        with open(join(root_dir, name, "experiment.yaml")) as f:
            dict = yaml.load(f)

        assert dict["name"] == name
        assert dict["description"] == description
        assert dict["documentation"] == documentation
        assert dict["root_dir"] == root_dir
        assert dict["license"] == license
        assert dict["authors"] == [
            {"affiliation": "aff 1", "name": "author 1"},
            {"affiliation": "aff 2", "name": "author 2"},
            {"affiliation": "aff 1", "name": "author 3"},
        ]
        assert dict["cite"] == [{"doi": "doi", "text": "text", "url": "url"}]
        assert dict["format_version"] == "0.1.0"
        assert dict["samples"] == ["sample"]

        self.assertRaises(
            FileExistsError, exp.save, overwrite=False, section_to_subdir=True
        )

        exp._description = "another desc"
        exp.save(overwrite=True, section_to_subdir=True)
        with open(join(root_dir, name, "experiment.yaml")) as f:
            dict = yaml.load(f)

        assert dict["name"] == name
        assert dict["description"] == "another desc"

        exp_loaded = Experiment.load(join(root_dir, name, "experiment.yaml"))
        assert exp_loaded.get_sample("sample").get_name() == "sample"

    def test_full_save_load(self):
        authors = [
            Author("First Author", "Inst1"),
            Author("Collab Author", "Inst2"),
            Author("Last Author", "Inst1"),
        ]

        cite = [
            Citation(
                doi="0000-0000-0000-0000",
                text="A long title to cite",
                url="https://it-is-published.ch",
            )
        ]

        exp = Experiment(
            name="20220930-Data-structure_v2",
            description="This is a test data structure.",
            documentation="README.md",
            authors=authors,
            root_dir=self.tmp_dir,
            exist_ok=True,
            cite=cite,
        )

        sample_0 = Sample(
            experiment=exp,
            name="Sample_00",
            description="This is sample 0.",
            documentation=None,
            aligned_data=None,
        )

        sec_0 = Section(
            name="sec_0",
            stitched=False,
            skip=False,
            acquisition="run_0",
            sample=sample_0,
            section_num=123,
            tile_grid_num=1,
            thickness=11,
            tile_height=3072,
            tile_width=2304,
            tile_overlap=200,
        )

        Tile(
            sec_0,
            path="/not/important.tif",
            tile_id=3,
            stage_x=0,
            stage_y=0,
            resolution_xy=1.2,
        )
        Tile(
            sec_0,
            path="/not/important.tif",
            tile_id=4,
            stage_x=2104,
            stage_y=0,
            resolution_xy=1.2,
        )
        Tile(
            sec_0,
            path="/not/important.tif",
            tile_id=5,
            stage_x=0,
            stage_y=2872,
            resolution_xy=1.2,
        )
        Tile(
            sec_0,
            path="/not/important.tif",
            tile_id=6,
            stage_x=2104,
            stage_y=2872,
            resolution_xy=1.2,
        )

        sec_1 = Section(
            name="sec_1",
            stitched=False,
            skip=False,
            acquisition="run_1",
            sample=sample_0,
            section_num=124,
            tile_grid_num=1,
            thickness=11,
            tile_height=3072,
            tile_width=2304,
            tile_overlap=200,
        )

        Tile(
            sec_1,
            path="/not/important.tif",
            tile_id=4,
            stage_x=2104,
            stage_y=0,
            resolution_xy=1.2,
        )
        Tile(
            sec_1,
            path="/not/important.tif",
            tile_id=5,
            stage_x=0,
            stage_y=2872,
            resolution_xy=1.2,
        )
        Tile(
            sec_1,
            path="/not/important.tif",
            tile_id=6,
            stage_x=2104,
            stage_y=2872,
            resolution_xy=1.2,
        )

        exp.save(overwrite=True, section_to_subdir=True)

        assert exists(
            join(self.tmp_dir, "20220930-Data-structure_v2", "experiment.yaml")
        )
        assert exists(
            join(self.tmp_dir, "20220930-Data-structure_v2", "Sample_00", "sample.yaml")
        )
        assert exists(
            join(
                self.tmp_dir,
                "20220930-Data-structure_v2",
                "Sample_00",
                "sec_0",
                "section.yaml",
            )
        )
        assert exists(
            join(
                self.tmp_dir,
                "20220930-Data-structure_v2",
                "Sample_00",
                "sec_1",
                "section.yaml",
            )
        )

        exp_load = Experiment.load(
            join(self.tmp_dir, "20220930-Data-structure_v2", "experiment.yaml")
        )

        assert exp.to_dict() == exp_load.to_dict()
        dict_1 = exp.get_sample("Sample_00").to_dict()
        dict_2 = exp_load.get_sample("Sample_00").to_dict()
        assert dict_1 == dict_2
        section_loaded = exp_load.get_sample("Sample_00").get_section("sec_0")
        assert not section_loaded._fully_initialized
        section_loaded.load_from_yaml()
        dict_1 = exp.get_sample("Sample_00").get_section("sec_0").to_dict()
        dict_2 = exp_load.get_sample("Sample_00").get_section("sec_0").to_dict()
        assert dict_1 == dict_2

        tile_id_map_path = join(section_loaded.get_section_dir(), "tile_id_map.json")
        assert_array_equal(
            exp.get_sample("Sample_00").get_section("sec_0").get_tile_id_map(),
            exp_load.get_sample("Sample_00")
            .get_section("sec_0")
            .get_tile_id_map(path=tile_id_map_path),
        )

        assert exists(tile_id_map_path)
        tim = section_loaded.get_tile_id_map(tile_id_map_path)
        tim[0, 0] = -1
        with open(tile_id_map_path, "w") as f:
            json.dump(tim.tolist(), f)

        assert_array_equal(
            tim,
            section_loaded.get_tile_id_map(
                join(section_loaded.get_section_dir(), "tile_id_map.json")
            ),
        )
        assert section_loaded.get_tile_id_map()[0, 0] == 3
