from os.path import join

import numpy as np
from experiment import Experiment
from numpy.testing import assert_array_equal
from record import SectionRecord
from record.BlockRecord import BlockRecord


def test_block_record(tmpdir):
    exp = Experiment(None, None)

    block = BlockRecord(
        experiment=exp, sbem_root_dir=str(tmpdir), block_id="bloc1", save_dir=tmpdir
    )

    section = SectionRecord(block, 1, 1, block.save_dir)

    assert block.get_section(1, 1) == section
    assert block.get_section(2, 0) is None

    assert block.get_section_range() == np.array([1])

    SectionRecord(block, 2, 1, block.save_dir)
    assert_array_equal(block.get_section_range(), np.array([1, 2]))
    assert not block.has_missing_section()
    assert block.get_missing_sections().size == 0

    SectionRecord(block, 4, 1, block.save_dir)
    assert_array_equal(block.get_section_range(), np.array([1, 2, 4]))
    assert block.has_missing_section()
    assert_array_equal(block.get_missing_sections(), np.array([3]))

    block.save()

    block_load = BlockRecord(None, None, None, None)
    block_load.load(join(tmpdir, "bloc1"))

    assert len(block_load.sections) == 3
    assert_array_equal(block_load.get_section_range(), np.array([1, 2, 4]))
    assert block_load.has_missing_section()
    assert_array_equal(block_load.get_missing_sections(), np.array([3]))
