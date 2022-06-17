import os
import shutil
import tempfile
from os.path import join
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from sbem.experiment import Experiment
from sbem.record.BlockRecord import BlockRecord
from sbem.record.SectionRecord import SectionRecord


class BlockTest(TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_block_record(self):
        exp_dir = join(self.tmp_dir, "exp")
        os.mkdir(exp_dir)
        exp = Experiment("test", exp_dir)

        block = BlockRecord(
            experiment=exp,
            sbem_root_dir=str(self.tmp_dir),
            block_id="bloc1",
            save_dir=self.tmp_dir,
        )

        section = SectionRecord(block, 1, 1, save_dir=block.save_dir)

        assert block.get_section(1, 1) == section
        assert block.get_section(2, 0) is None

        assert block.get_section_range() == np.array([1])

        SectionRecord(block, 2, 1, save_dir=block.save_dir)
        assert_array_equal(block.get_section_range(), np.array([1, 2]))
        assert not block.has_missing_section()
        assert block.get_missing_sections().size == 0

        SectionRecord(block, 4, 1, save_dir=block.save_dir)
        assert_array_equal(block.get_section_range(), np.array([1, 2, 4]))
        assert block.has_missing_section()
        assert_array_equal(block.get_missing_sections(), np.array([3]))

        block.save()

        block_load = BlockRecord(exp, str(self.tmp_dir), "bloc1", None)
        block_load.load(join(self.tmp_dir, "bloc1"))

        assert len(block_load.sections) == 3
        assert_array_equal(block_load.get_section_range(), np.array([1, 2, 4]))
        assert block_load.has_missing_section()
        assert_array_equal(block_load.get_missing_sections(), np.array([3]))
