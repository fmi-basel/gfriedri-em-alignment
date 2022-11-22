import shutil
import tempfile
from os.path import exists, join
from unittest import TestCase

import numpy as np
from ruyaml import YAML
from tifffile import imwrite

from sbem.record.Sample import Sample
from sbem.record.Section import Section


class SectionTest(TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

        self.img = np.random.randint(0, 3200, size=(1320, 3021), dtype=np.uint16)

        imwrite(join(self.tmp_dir, "img.tif"), self.img)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_sample(self):
        name = "sample_0"
        license = "A license."
        experiment = None
        description = "Description"
        documentation = "README.md"
        aligned_data = "/path/to/zarr"

        sample = Sample(
            experiment, name, description, documentation, aligned_data, license
        )

        assert sample.get_experiment() == experiment
        assert sample.get_name() == name
        assert sample.get_description() == description
        assert sample.get_documentation() == documentation
        assert sample.get_aligned_data() == aligned_data
        assert sample.get_license() == license

        sample_1 = Sample(
            experiment, name, description, documentation, aligned_data, license
        )

        assert sample.get_section("sec") is None

        section = Section.lazy_loading(
            "sec", 123, 1, False, True, "run0", "details.yaml"
        )

        sample_1.add_section(section)
        assert section.get_sample() == sample_1
        assert sample_1.get_section("sec") == section
        assert sample_1.get_min_section_num(1) == 123
        assert sample_1.get_max_section_num(1) == 123
        section.set_sample(sample_1)

        self.assertRaises(AssertionError, sample.add_section, section=section)

        sample_dict = sample.to_dict(section_to_subdir=True)
        assert sample_dict["name"] == name
        assert sample_dict["license"] == license
        assert sample_dict["format_version"] == "0.1.0"
        assert sample_dict["description"] == description
        assert sample_dict["documentation"] == documentation
        assert sample_dict["aligned_data"] == aligned_data
        assert sample_dict["sections"] == []

        sample_1_dict = sample_1.to_dict(section_to_subdir=True)
        assert sample_1_dict["sections"] == [
            {
                "name": "sec",
                "section_num": 123,
                "tile_grid_num": 1,
                "acquisition": "run0",
                "stitched": False,
                "skip": True,
                "details": "sec/section.yaml",
            }
        ]

        sample_1_dict = sample_1.to_dict(section_to_subdir=False)
        assert sample_1_dict["sections"] == [
            {
                "name": "sec",
                "section_num": 123,
                "tile_grid_num": 1,
                "acquisition": "run0",
                "stitched": False,
                "skip": True,
                "details": section.to_dict(),
            }
        ]

    def test_save(self):
        name = "sample_0"
        license = "A license."
        experiment = None
        description = "Description"
        documentation = "README.md"
        aligned_data = "/path/to/zarr"

        sample = Sample(
            experiment, name, description, documentation, aligned_data, license
        )

        section = Section.lazy_loading(
            "sec", 123, 1, False, True, "run0", "details.yaml"
        )

        sample.add_section(section)

        sample.save(path=self.tmp_dir, overwrite=False, section_to_subdir=True)
        assert exists(join(self.tmp_dir, name, "sample.yaml"))

        yaml = YAML(typ="rt")
        with open(join(self.tmp_dir, name, "sample.yaml")) as f:
            dict = yaml.load(f)

        assert dict["name"] == name
        assert dict["license"] == license
        assert dict["format_version"] == "0.1.0"
        assert dict["description"] == description
        assert dict["documentation"] == documentation
        assert dict["aligned_data"] == aligned_data
        assert dict["sections"] == [
            {
                "name": "sec",
                "section_num": 123,
                "tile_grid_num": 1,
                "acquisition": "run0",
                "stitched": False,
                "skip": True,
                "details": "sec/section.yaml",
            }
        ]
        assert exists(join(self.tmp_dir, name, "sec", "section.yaml"))

        sample._description = "new description"
        sample.save(path=self.tmp_dir, overwrite=True, section_to_subdir=True)

        with open(join(self.tmp_dir, name, "sample.yaml")) as f:
            dict = yaml.load(f)

        assert dict["name"] == name
        assert dict["license"] == license
        assert dict["format_version"] == "0.1.0"
        assert dict["description"] == "new description"
        assert dict["documentation"] == documentation
        assert dict["aligned_data"] == aligned_data
        assert dict["sections"] == [
            {
                "name": "sec",
                "section_num": 123,
                "tile_grid_num": 1,
                "acquisition": "run0",
                "stitched": False,
                "skip": True,
                "details": "sec/section.yaml",
            }
        ]
        assert exists(join(self.tmp_dir, name, "sec", "section.yaml"))

        shutil.rmtree(join(self.tmp_dir, name, "sec"))
        sample.save(path=self.tmp_dir, overwrite=True, section_to_subdir=False)

        with open(join(self.tmp_dir, name, "sample.yaml")) as f:
            dict = yaml.load(f)
        assert dict["name"] == name
        assert dict["license"] == license
        assert dict["format_version"] == "0.1.0"
        assert dict["description"] == "new description"
        assert dict["documentation"] == documentation
        assert dict["aligned_data"] == aligned_data
        assert dict["sections"] == [
            {
                "name": "sec",
                "section_num": 123,
                "tile_grid_num": 1,
                "acquisition": "run0",
                "stitched": False,
                "skip": True,
                "details": section.to_dict(),
            }
        ]
        assert not exists(join(self.tmp_dir, name, "sec", "section.yaml"))

        sample_loaded = Sample.load(join(self.tmp_dir, name, "sample.yaml"))
        assert sample_loaded.get_experiment() is None
        assert sample_loaded.get_name() == name
        assert sample_loaded.get_description() == "new description"
        assert sample_loaded.get_documentation() == documentation
        assert sample_loaded.get_aligned_data() == aligned_data
        assert sample_loaded.get_license() == license
        assert sample_loaded.get_section("sec").to_dict() == section.to_dict()
        assert sample_loaded.get_section("sec").get_sample() == sample_loaded

    def test_get_section_range(self):
        sample = Sample(None, "sample", "Desc", "Docu", None)

        sec_2 = Section(
            sample=sample,
            name="2",
            stitched=False,
            skip=False,
            acquisition="run_0",
            section_num=2,
            tile_grid_num=1,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        sec_3 = Section(
            sample=sample,
            name="3",
            stitched=False,
            skip=True,
            acquisition="run_0",
            section_num=3,
            tile_grid_num=1,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        Section(
            sample=sample,
            name="1",
            stitched=False,
            skip=False,
            acquisition="run_0",
            section_num=1,
            tile_grid_num=1,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        sec_4 = Section(
            sample=sample,
            name="4",
            stitched=False,
            skip=False,
            acquisition="run_0",
            section_num=4,
            tile_grid_num=2,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        sec_5 = Section(
            sample=sample,
            name="5",
            stitched=False,
            skip=False,
            acquisition="run_0",
            section_num=5,
            tile_grid_num=1,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        sec_range = sample.get_section_range(
            start_section_num=2,
            end_section_num=6,
            tile_grid_num=1,
            include_skipped=False,
        )
        assert len(sec_range) == 2
        assert sec_range[0] == sec_2
        assert sec_range[1] == sec_5

        sec_range = sample.get_section_range(
            start_section_num=2,
            end_section_num=6,
            tile_grid_num=1,
            include_skipped=True,
        )
        assert len(sec_range) == 3
        assert sec_range[0] == sec_2
        assert sec_range[1] == sec_3
        assert sec_range[2] == sec_5

        sec_range = sample.get_section_range(
            start_section_num=2,
            end_section_num=5,
            tile_grid_num=1,
            include_skipped=True,
        )
        assert len(sec_range) == 3
        assert sec_range[0] == sec_2
        assert sec_range[1] == sec_3
        assert sec_range[2] == sec_5

        sec_range = sample.get_section_range(
            start_section_num=1,
            end_section_num=5,
            tile_grid_num=2,
            include_skipped=True,
        )
        assert len(sec_range) == 1
        assert sec_range[0] == sec_4

    def test_get_sections_of_acquisition(self):
        sample = Sample(None, "sample", "Desc", "Docu", None)

        sec_2 = Section(
            sample=sample,
            name="2",
            stitched=False,
            skip=False,
            acquisition="run_0",
            section_num=2,
            tile_grid_num=1,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        sec_3 = Section(
            sample=sample,
            name="3",
            stitched=False,
            skip=True,
            acquisition="run_1",
            section_num=3,
            tile_grid_num=1,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        sec_1 = Section(
            sample=sample,
            name="1",
            stitched=False,
            skip=False,
            acquisition="run_0",
            section_num=1,
            tile_grid_num=1,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        sec_4 = Section(
            sample=sample,
            name="4",
            stitched=False,
            skip=False,
            acquisition="run_1",
            section_num=4,
            tile_grid_num=2,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        sec_5 = Section(
            sample=sample,
            name="5",
            stitched=False,
            skip=False,
            acquisition="run_1",
            section_num=5,
            tile_grid_num=1,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        sec_range = sample.get_sections_of_acquisition(
            acquisition="run_0", tile_grid_num=1, include_skipped=False
        )
        assert len(sec_range) == 2
        assert sec_range[0] == sec_1
        assert sec_range[1] == sec_2

        sec_range = sample.get_sections_of_acquisition(
            acquisition="run_1", tile_grid_num=1, include_skipped=False
        )
        assert len(sec_range) == 1
        assert sec_range[0] == sec_5

        sec_range = sample.get_sections_of_acquisition(
            acquisition="run_1", tile_grid_num=1, include_skipped=True
        )
        assert len(sec_range) == 2
        assert sec_range[0] == sec_3
        assert sec_range[1] == sec_5

        sec_range = sample.get_sections_of_acquisition(
            acquisition="run_1", tile_grid_num=2, include_skipped=True
        )
        assert len(sec_range) == 1
        assert sec_range[0] == sec_4

    def test_delete_sections_no_dir(self):
        sample = Sample(None, "sample", "Desc", "Docu", None)

        Section(
            sample=sample,
            name="2",
            stitched=False,
            skip=False,
            acquisition="run_0",
            section_num=2,
            tile_grid_num=1,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        Section(
            sample=sample,
            name="3",
            stitched=False,
            skip=True,
            acquisition="run_0",
            section_num=3,
            tile_grid_num=1,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        Section(
            sample=sample,
            name="1",
            stitched=False,
            skip=False,
            acquisition="run_0",
            section_num=1,
            tile_grid_num=1,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        Section(
            sample=sample,
            name="4",
            stitched=False,
            skip=False,
            acquisition="run_0",
            section_num=4,
            tile_grid_num=2,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        Section(
            sample=sample,
            name="5",
            stitched=False,
            skip=False,
            acquisition="run_0",
            section_num=5,
            tile_grid_num=1,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        sample.save(self.tmp_dir, overwrite=True, section_to_subdir=False)

        assert exists(join(self.tmp_dir, sample.get_name()))
        assert not exists(join(self.tmp_dir, sample.get_name(), "1"))
        assert not exists(join(self.tmp_dir, sample.get_name(), "2"))
        assert not exists(join(self.tmp_dir, sample.get_name(), "3"))
        assert not exists(join(self.tmp_dir, sample.get_name(), "4"))
        assert not exists(join(self.tmp_dir, sample.get_name(), "5"))

        yaml = YAML(typ="rt")
        with open(join(self.tmp_dir, sample.get_name(), "sample.yaml")) as f:
            dict = yaml.load(f)

        for sec, name in zip(dict["sections"], ["1", "2", "3", "4", "5"]):
            assert sec["name"] == name

        secs_to_delete = sample.delete_sections(2, 3, 1)
        assert secs_to_delete == ["2", "3"]

        sample.save(self.tmp_dir, overwrite=True, section_to_subdir=False)

        assert exists(join(self.tmp_dir, sample.get_name()))
        assert not exists(join(self.tmp_dir, sample.get_name(), "1"))
        assert not exists(join(self.tmp_dir, sample.get_name(), "4"))
        assert not exists(join(self.tmp_dir, sample.get_name(), "5"))

        yaml = YAML(typ="rt")
        with open(join(self.tmp_dir, sample.get_name(), "sample.yaml")) as f:
            dict = yaml.load(f)

        for sec, name in zip(dict["sections"], ["1", "4", "5"]):
            assert sec["name"] == name

    def test_delete_sections_dir(self):
        sample = Sample(None, "sample", "Desc", "Docu", None)

        Section(
            sample=sample,
            name="2",
            stitched=False,
            skip=False,
            acquisition="run_0",
            section_num=2,
            tile_grid_num=1,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        Section(
            sample=sample,
            name="3",
            stitched=False,
            skip=True,
            acquisition="run_0",
            section_num=3,
            tile_grid_num=1,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        Section(
            sample=sample,
            name="1",
            stitched=False,
            skip=False,
            acquisition="run_0",
            section_num=1,
            tile_grid_num=1,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        Section(
            sample=sample,
            name="4",
            stitched=False,
            skip=False,
            acquisition="run_0",
            section_num=4,
            tile_grid_num=2,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        Section(
            sample=sample,
            name="5",
            stitched=False,
            skip=False,
            acquisition="run_0",
            section_num=5,
            tile_grid_num=1,
            thickness=11.1,
            tile_height=123,
            tile_width=123,
            tile_overlap=2,
        )

        sample.save(self.tmp_dir, overwrite=True, section_to_subdir=True)

        assert exists(join(self.tmp_dir, sample.get_name()))
        assert exists(join(self.tmp_dir, sample.get_name(), "1"))
        assert exists(join(self.tmp_dir, sample.get_name(), "2"))
        assert exists(join(self.tmp_dir, sample.get_name(), "3"))
        assert exists(join(self.tmp_dir, sample.get_name(), "4"))
        assert exists(join(self.tmp_dir, sample.get_name(), "5"))

        yaml = YAML(typ="rt")
        with open(join(self.tmp_dir, sample.get_name(), "sample.yaml")) as f:
            dict = yaml.load(f)

        for sec, name in zip(dict["sections"], ["1", "2", "3", "4", "5"]):
            assert sec["name"] == name

        secs_to_delete = sample.delete_sections(2, 3, 1)
        for d in secs_to_delete:
            shutil.rmtree(join(self.tmp_dir, sample.get_name(), d))

        sample.save(self.tmp_dir, overwrite=True, section_to_subdir=True)

        assert exists(join(self.tmp_dir, sample.get_name()))
        assert exists(join(self.tmp_dir, sample.get_name(), "1"))
        assert not exists(join(self.tmp_dir, sample.get_name(), "2"))
        assert not exists(join(self.tmp_dir, sample.get_name(), "3"))
        assert exists(join(self.tmp_dir, sample.get_name(), "4"))
        assert exists(join(self.tmp_dir, sample.get_name(), "5"))

        yaml = YAML(typ="rt")
        with open(join(self.tmp_dir, sample.get_name(), "sample.yaml")) as f:
            dict = yaml.load(f)

        for sec, name in zip(dict["sections"], ["1", "4", "5"]):
            assert sec["name"] == name

        assert sample.get_section("1").get_section_dir() is None
