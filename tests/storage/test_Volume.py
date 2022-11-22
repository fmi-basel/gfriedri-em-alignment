import logging
import shutil
import tempfile
from os.path import exists, join
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from sbem.record.Author import Author
from sbem.record.Citation import Citation
from sbem.storage.Volume import Volume


class VolumeTest(TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_create_volume(self):
        vol = Volume(
            "test-volume",
            "description",
            "documentation",
            [Author(name="author 1", affiliation="aff 1")],
            self.tmp_dir,
            False,
            "license",
            [Citation(doi="doi", text="text", url="url")],
            logging,
        )

        assert vol.get_name() == "test-volume"
        assert vol.get_license() == "license"
        assert vol.get_description() == "description"
        assert vol.get_documentation() == "documentation"
        assert vol.get_authors_pretty() == "author 1[1]\n[1] aff 1"
        assert vol.get_citations_pretty() == "text\nDOI: doi\nURL: url"
        assert vol.get_dir() == join(self.tmp_dir, "test-volume")
        assert_array_equal(vol.get_origin(), np.array([0, 0, 0]))
        assert vol.get_zarr_volume().chunk_store.dir_path() == join(
            self.tmp_dir, "test-volume", "ngff_volume.zarr/"
        )

        self.assertRaises(
            FileExistsError,
            Volume,
            "test-volume",
            "description",
            "documentation",
            [Author(name="author 1", affiliation="aff 1")],
            self.tmp_dir,
            False,
            "license",
            [Citation(doi="doi", text="text", url="url")],
            logging,
        )

        # overwrite
        vol = Volume(
            "test-volume1",
            "description1",
            "documentation1",
            [Author(name="author 11", affiliation="aff 11")],
            self.tmp_dir,
            True,
            "license1",
            [Citation(doi="doi1", text="text1", url="url1")],
            logging,
        )

        assert vol.get_name() == "test-volume1"
        assert vol.get_license() == "license1"
        assert vol.get_description() == "description1"
        assert vol.get_documentation() == "documentation1"
        assert vol.get_authors_pretty() == "author 11[1]\n[1] aff 11"
        assert vol.get_citations_pretty() == "text1\nDOI: doi1\nURL: url1"
        assert vol.get_dir() == join(self.tmp_dir, "test-volume1")
        assert_array_equal(vol.get_origin(), np.array([0, 0, 0]))
        assert vol.get_zarr_volume().chunk_store.dir_path() == join(
            self.tmp_dir, "test-volume1", "ngff_volume.zarr/"
        )

    def test_write_section(self):
        vol = Volume(
            name="test-volume",
            description="description",
            documentation="documentation",
            authors=[Author(name="author 1", affiliation="aff 1")],
            root_dir=self.tmp_dir,
            exist_ok=False,
            license="license",
            cite=[Citation(doi="doi", text="text", url="url")],
            logger=logging,
        )

        # Write first section
        data = np.random.randint(0, 255, size=(1, 123, 342))
        vol.write_section(123, data, (0, 0, 0))

        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "0", "0", "0")
        )
        assert vol.get_zarr_volume()["0"].shape == data.shape
        assert_array_equal(vol.get_zarr_volume()["0"][0:1, :123, :342], data)
        assert_array_equal(vol.get_section_data(123), data)
        assert vol._section_list == [123]
        assert 123 in vol._section_offset_map.keys()
        assert len(vol._section_offset_map.keys()) == 1
        assert_array_equal(vol._section_offset_map[123], np.array([0, 0, 0]))
        assert_array_equal(vol.get_origin(), np.array([0, 0, 0]))

        # Prepend a section
        data1 = np.random.randint(0, 255, size=(1, 2744, 2744))
        vol.write_section(124, data1, (0, 0, 0))

        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "0", "0", "0")
        )
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "1", "0", "0")
        )
        assert vol.get_zarr_volume()["0"].shape == (2, 2744, 2744)
        assert_array_equal(vol.get_zarr_volume()["0"][0:1], data1)
        assert_array_equal(vol.get_section_data(124), data1)
        assert_array_equal(vol.get_zarr_volume()["0"][1:2, :123, :342], data)
        assert_array_equal(vol.get_section_data(123), data)
        assert vol._section_list == [124, 123]
        assert 124 in vol._section_offset_map.keys()
        assert len(vol._section_offset_map.keys()) == 2
        assert_array_equal(vol._section_offset_map[124], np.array([0, 0, 0]))
        assert_array_equal(vol._section_offset_map[123], np.array([1, 0, 0]))
        assert_array_equal(vol.get_origin(), np.array([0, 0, 0]))

        # Append a section
        data2 = np.random.randint(0, 255, size=(1, 3000, 2744))
        vol.write_section(125, data2, (2, 0, 0))
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "2", "0", "0")
        )
        assert vol.get_zarr_volume()["0"].shape == (3, 3000, 2744)
        assert_array_equal(vol.get_zarr_volume()["0"][0:1, :2744], data1)
        assert_array_equal(vol.get_section_data(124), data1)
        assert_array_equal(vol.get_zarr_volume()["0"][1:2, :123, :342], data)
        assert_array_equal(vol.get_section_data(123), data)
        assert_array_equal(vol.get_zarr_volume()["0"][2:3], data2)
        assert_array_equal(vol.get_section_data(125), data2)
        assert vol._section_list == [124, 123, 125]
        assert 125 in vol._section_offset_map.keys()
        assert len(vol._section_offset_map.keys()) == 3
        assert_array_equal(vol._section_offset_map[124], np.array([0, 0, 0]))
        assert_array_equal(vol._section_offset_map[123], np.array([1, 0, 0]))
        assert_array_equal(vol._section_offset_map[125], np.array([2, 0, 0]))
        assert_array_equal(vol.get_origin(), np.array([0, 0, 0]))

        # Insert a section
        data3 = np.random.randint(0, 255, size=(1, 2744, 3000))
        vol.write_section(126, data3, (1, 0, 0))
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "3", "0", "0")
        )
        assert vol.get_zarr_volume()["0"].shape == (4, 3000, 3000)
        assert_array_equal(vol.get_zarr_volume()["0"][0:1, :2744, :2744], data1)
        assert_array_equal(vol.get_section_data(124), data1)
        assert_array_equal(
            vol.get_zarr_volume()["0"][
                1:2,
                :2744,
            ],
            data3,
        )
        assert_array_equal(vol.get_section_data(126), data3)
        assert_array_equal(vol.get_zarr_volume()["0"][2:3, :123, :342], data)
        assert_array_equal(vol.get_section_data(123), data)
        assert_array_equal(vol.get_zarr_volume()["0"][3:4, :, :2744], data2)
        assert_array_equal(vol.get_section_data(125), data2)
        assert vol._section_list == [124, 126, 123, 125]
        assert 126 in vol._section_offset_map.keys()
        assert len(vol._section_offset_map.keys()) == 4
        assert_array_equal(vol._section_offset_map[124], np.array([0, 0, 0]))
        assert_array_equal(vol._section_offset_map[126], np.array([1, 0, 0]))
        assert_array_equal(vol._section_offset_map[123], np.array([2, 0, 0]))
        assert_array_equal(vol._section_offset_map[125], np.array([3, 0, 0]))
        assert_array_equal(vol.get_origin(), np.array([0, 0, 0]))

    def test_write_with_offsets(self):
        vol = Volume(
            name="test-volume",
            description="description",
            documentation="documentation",
            authors=[Author(name="author 1", affiliation="aff 1")],
            root_dir=self.tmp_dir,
            exist_ok=False,
            license="license",
            cite=[Citation(doi="doi", text="text", url="url")],
            logger=logging,
        )

        # Add first section
        data = np.random.randint(0, 255, size=(1, 123, 342))
        vol.write_section(123, data, (0, 0, 0))

        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "0", "0", "0")
        )
        assert vol.get_zarr_volume()["0"].shape == data.shape
        assert vol._section_list == [123]
        assert 123 in vol._section_offset_map.keys()
        assert len(vol._section_offset_map.keys()) == 1
        assert_array_equal(vol._section_offset_map[123], np.array([0, 0, 0]))
        assert_array_equal(vol.get_origin(), np.array([0, 0, 0]))
        assert_array_equal(vol.get_section_data(123), data)

        # Add second section with xy-offset = (100, 100) fitting in chunks
        data1 = np.random.randint(0, 255, size=(1, 234, 423))
        vol.write_section(124, data1, (1, 100, 100))
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "1", "0", "0")
        )
        assert vol.get_zarr_volume()["0"].shape == (2, 334, 523)
        assert vol._section_list == [123, 124]
        assert len(vol._section_offset_map.keys()) == 2
        assert_array_equal(vol._section_offset_map[123], np.array([0, 0, 0]))
        assert_array_equal(vol._section_offset_map[124], np.array([1, 100, 100]))
        assert_array_equal(vol.get_origin(), np.array([0, 0, 0]))
        assert_array_equal(vol.get_zarr_volume()["0"][0, :123, :342], data[0])
        assert_array_equal(vol.get_section_data(123), data)
        assert_array_equal(vol.get_zarr_volume()["0"][1, 100:334, 100:523], data1[0])
        assert_array_equal(vol.get_section_data(124), data1)

        # Add 3rd section with xy-offset = (2700, 100) not fitting in chunks
        data2 = np.random.randint(0, 255, size=(1, 421, 532))
        vol.write_section(125, data2, (2, 2700, 100))
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "0", "0", "0")
        )
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "1", "0", "0")
        )
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "2", "0", "0")
        )
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "2", "1", "0")
        )
        assert vol.get_zarr_volume()["0"].shape == (3, 3121, 632)
        assert vol._section_list == [123, 124, 125]
        assert len(vol._section_offset_map.keys()) == 3
        assert_array_equal(vol._section_offset_map[123], np.array([0, 0, 0]))
        assert_array_equal(vol._section_offset_map[124], np.array([1, 100, 100]))
        assert_array_equal(vol._section_offset_map[125], np.array([2, 2700, 100]))
        assert_array_equal(vol.get_origin(), np.array([0, 0, 0]))
        assert_array_equal(vol.get_zarr_volume()["0"][0, :123, :342], data[0])
        assert_array_equal(vol.get_section_data(123), data)
        assert_array_equal(vol.get_zarr_volume()["0"][1, 100:334, 100:523], data1[0])
        assert_array_equal(vol.get_section_data(124), data1)
        assert_array_equal(vol.get_zarr_volume()["0"][2, 2700:3121, 100:632], data2[0])
        assert_array_equal(vol.get_section_data(125), data2)

        # Add 4th section with xy-offset = (-100, -2800) not fitting in chunks
        data3 = np.random.randint(0, 255, size=(1, 121, 332))
        vol.write_section(126, data3, (3, -100, -2800))
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "0", "1", "2")
        )
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "1", "1", "2")
        )
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "2", "1", "2")
        )
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "2", "2", "2")
        )
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "3", "0", "0")
        )
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "3", "0", "1")
        )
        assert not exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "3", "0", "2")
        )
        print(vol.get_zarr_volume()["0"].shape)
        assert vol.get_zarr_volume()["0"].shape == (4, 5865, 6120)
        assert vol._section_list == [123, 124, 125, 126]
        assert len(vol._section_offset_map.keys()) == 4
        assert_array_equal(vol._section_offset_map[123], np.array([0, 2744, 5488]))
        assert_array_equal(vol._section_offset_map[124], np.array([1, 2844, 5588]))
        assert_array_equal(vol._section_offset_map[125], np.array([2, 5444, 5588]))
        assert_array_equal(
            vol._section_offset_map[126], np.array([3, 2744 - 100, 5488 - 2800])
        )
        assert_array_equal(vol.get_origin(), np.array([0, 2744, 5488]))
        assert_array_equal(
            vol.get_zarr_volume()["0"][0, 2744 : 2744 + 123, 5488 : 5488 + 342], data[0]
        )
        assert_array_equal(vol.get_section_data(123), data)
        assert_array_equal(
            vol.get_zarr_volume()["0"][
                1, 2744 + 100 : 2744 + 334, 5488 + 100 : 5488 + 523
            ],
            data1[0],
        )
        assert_array_equal(vol.get_section_data(124), data1)
        assert_array_equal(
            vol.get_zarr_volume()["0"][
                2, 2744 + 2700 : 2744 + 3121, 5488 + 100 : 5488 + 632
            ],
            data2[0],
        )
        assert_array_equal(vol.get_section_data(125), data2)
        assert_array_equal(
            vol.get_zarr_volume()["0"][3, 2644 : 2644 + 121, 2688 : 2688 + 332],
            data3[0],
        )
        assert_array_equal(vol.get_section_data(126), data3)

    def test_remove_section(self):
        vol = Volume(
            name="test-volume",
            description="description",
            documentation="documentation",
            authors=[Author(name="author 1", affiliation="aff 1")],
            root_dir=self.tmp_dir,
            exist_ok=False,
            license="license",
            cite=[Citation(doi="doi", text="text", url="url")],
            logger=logging,
        )

        # Add first section
        data = np.random.randint(0, 255, size=(1, 123, 342))
        vol.write_section(123, data, (0, 0, 0))

        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "0", "0", "0")
        )
        assert_array_equal(vol.get_section_data(123), data)
        assert vol._section_list == [123]
        assert 123 in vol._section_offset_map.keys()
        assert 123 in vol._section_shape_map.keys()

        # Remove first section
        vol.remove_section(123)
        assert not exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "0")
        )
        assert vol._section_list == []
        assert 123 not in vol._section_offset_map.keys()
        assert 123 not in vol._section_shape_map.keys()

        # Add section back
        vol.write_section(123, data, (0, 0, 0))
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "0", "0", "0")
        )
        assert_array_equal(vol.get_section_data(123), data)
        assert vol._section_list == [123]
        assert 123 in vol._section_offset_map.keys()
        assert 123 in vol._section_shape_map.keys()

        # Add another two section
        data1 = np.random.randint(0, 255, size=(1, 123, 342))
        vol.write_section(124, data1, (1, 0, 0))
        data2 = np.random.randint(0, 255, size=(1, 123, 342))
        vol.write_section(125, data2, (3, 0, 0))

        # Remove center section
        vol.remove_section(124)
        assert not exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "3")
        )
        assert vol._section_list == [123, None, 125]
        assert 124 not in vol._section_offset_map.keys()
        assert 124 not in vol._section_shape_map.keys()
        assert_array_equal(vol.get_section_origin(123), np.array([0, 0, 0]))
        assert_array_equal(vol.get_section_origin(125), np.array([2, 0, 0]))
        assert vol.get_zarr_volume()["0"].shape == (3, 123, 342)

        assert_array_equal(vol.get_section_data(123), data)
        assert_array_equal(vol.get_section_data(125), data2)

    def test_append_section(self):
        vol = Volume(
            name="test-volume",
            description="description",
            documentation="documentation",
            authors=[Author(name="author 1", affiliation="aff 1")],
            root_dir=self.tmp_dir,
            exist_ok=False,
            license="license",
            cite=[Citation(doi="doi", text="text", url="url")],
            logger=logging,
        )

        # Add first section
        data = np.random.randint(0, 255, size=(1, 123, 342))
        vol.append_section(123, data, (0, 0, 0))

        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "0", "0", "0")
        )
        assert_array_equal(vol.get_section_data(123), data)
        assert vol._section_list == [123]
        assert 123 in vol._section_offset_map.keys()
        assert 123 in vol._section_shape_map.keys()

        # Add 2nd section with negative offset
        data1 = np.random.randint(0, 255, size=(1, 123, 342))
        vol.append_section(124, data1, (1, 0, -100))

        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "0", "0", "1")
        )
        assert_array_equal(vol.get_section_data(123), data)
        assert 123 in vol._section_offset_map.keys()
        assert 123 in vol._section_shape_map.keys()

        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "1", "0", "0")
        )
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "1", "0", "1")
        )
        assert_array_equal(vol.get_section_data(124), data1)
        assert vol._section_list == [123, 124]
        assert 124 in vol._section_offset_map.keys()
        assert 124 in vol._section_shape_map.keys()
        assert_array_equal(vol.get_origin(), np.array([0, 0, 2744]))

        # Add 3rd section with positive offset
        # Append inserts with the offset relative to the previous section
        data2 = np.random.randint(0, 255, size=(1, 123, 342))
        vol.append_section(125, data2, (1, 0, 100))

        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "0", "0", "1")
        )
        assert_array_equal(vol.get_section_data(123), data)
        assert 123 in vol._section_offset_map.keys()
        assert 123 in vol._section_shape_map.keys()
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "1", "0", "0")
        )
        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "1", "0", "1")
        )
        assert_array_equal(vol.get_section_data(124), data1)
        assert 124 in vol._section_offset_map.keys()
        assert 124 in vol._section_shape_map.keys()

        assert exists(
            join(self.tmp_dir, "test-volume", "ngff_volume.zarr", "0", "2", "0", "1")
        )
        assert_array_equal(vol.get_section_data(125), data2)
        assert_array_equal(
            vol.get_zarr_volume()[0][2, :123, 2744 : 2744 + 342], data2[0]
        )
        assert vol._section_list == [123, 124, 125]
        assert 125 in vol._section_offset_map.keys()
        assert 125 in vol._section_shape_map.keys()

    def test_save_and_load(self):
        vol = Volume(
            name="test-volume",
            description="description",
            documentation="documentation",
            authors=[Author(name="author 1", affiliation="aff 1")],
            root_dir=self.tmp_dir,
            exist_ok=False,
            license="license",
            cite=[Citation(doi="doi", text="text", url="url")],
            logger=logging,
        )

        # Add first section
        data = np.random.randint(0, 255, size=(1, 123, 342))
        vol.append_section(123, data, (0, 0, 0))
        vol.save()

        vol_load = Volume.load(join(self.tmp_dir, "test-volume", "volume.yaml"))
        assert vol.get_name() == vol_load.get_name()
        assert vol.get_license() == vol_load.get_license()
        assert vol.get_description() == vol_load.get_description()
        assert vol.get_documentation() == vol_load.get_documentation()
        assert vol.get_authors_pretty() == vol_load.get_authors_pretty()
        assert vol.get_citations_pretty() == vol_load.get_citations_pretty()
        assert vol.get_dir() == vol_load.get_dir()
        assert_array_equal(vol.get_origin(), vol_load.get_origin())
        vol_store_path = vol.get_zarr_volume().chunk_store.dir_path()
        vol_load_store_path = vol_load.get_zarr_volume().chunk_store.dir_path()
        assert vol_store_path == vol_load_store_path
        assert vol._section_list == vol_load._section_list
        for k in vol._section_offset_map.keys():
            assert_array_equal(
                vol._section_offset_map[k], vol_load._section_offset_map[k]
            )
        for k in vol._section_shape_map.keys():
            assert_array_equal(
                vol._section_shape_map[k], vol_load._section_shape_map[k]
            )

        # Add 2nd section with negative offset
        data1 = np.random.randint(0, 255, size=(1, 123, 342))
        vol.append_section(124, data1, (1, 0, -100))

        vol.save()

        vol_load = Volume.load(join(self.tmp_dir, "test-volume", "volume.yaml"))
        assert vol.get_name() == vol_load.get_name()
        assert vol.get_license() == vol_load.get_license()
        assert vol.get_description() == vol_load.get_description()
        assert vol.get_documentation() == vol_load.get_documentation()
        assert vol.get_authors_pretty() == vol_load.get_authors_pretty()
        assert vol.get_citations_pretty() == vol_load.get_citations_pretty()
        assert vol.get_dir() == vol_load.get_dir()
        assert_array_equal(vol.get_origin(), vol_load.get_origin())
        vol_store_path = vol.get_zarr_volume().chunk_store.dir_path()
        vol_load_store_path = vol_load.get_zarr_volume().chunk_store.dir_path()
        assert vol_store_path == vol_load_store_path
        assert vol._section_list == vol_load._section_list
        for k in vol._section_offset_map.keys():
            assert_array_equal(
                vol._section_offset_map[k], vol_load._section_offset_map[k]
            )
        for k in vol._section_shape_map.keys():
            assert_array_equal(
                vol._section_shape_map[k], vol_load._section_shape_map[k]
            )
