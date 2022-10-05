import shutil
import tempfile
from os.path import exists, join
from unittest import TestCase

import numpy as np
from ruyaml import YAML
from tifffile import imsave

from sbem.record_v2.Sample import Sample
from sbem.record_v2.Section import Section


class SectionTest(TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

        self.img = np.random.randint(0, 3200, size=(1320, 3021), dtype=np.uint16)

        imsave(join(self.tmp_dir, "img.tif"), self.img)

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

        section = Section.lazy_loading("sec", False, True, "run0", "details.yaml")

        sample_1.add_section(section)
        assert section.get_sample() == sample_1
        assert sample_1.get_section("sec")
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

        section = Section.lazy_loading("sec", False, True, "run0", "details.yaml")

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
