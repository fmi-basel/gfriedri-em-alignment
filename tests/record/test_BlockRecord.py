from os.path import join

import numpy as np
from numpy.testing import assert_array_equal
from sbem.experiment.Experiment import Experiment
from sbem.record.BlockRecord import BlockRecord
from sbem.record.SectionRecord import SectionRecord


def test_block_record(tmpdir):
    exp = Experiment(None, None)

    block = BlockRecord(experiment=exp, block_id="bloc1")

    section = SectionRecord(block, 1, 1)

    assert block.get_section(1, 1) == section
    assert block.get_section(2, 0) is None

    assert block.get_section_range() == np.array([1])

    SectionRecord(block, 2, 1)
    assert_array_equal(block.get_section_range(), np.array([1, 2]))
    assert not block.has_missing_section()
    assert block.get_missing_sections().size == 0

    SectionRecord(block, 4, 1)
    assert_array_equal(block.get_section_range(), np.array([1, 2, 4]))
    assert block.has_missing_section()
    assert_array_equal(block.get_missing_sections(), np.array([3]))

    block_path = join(tmpdir, "block")
    block.save(block_path)

    block_load = BlockRecord(None, None)
    block_load.load(block_path)

    assert len(block_load.sections) == 3
    assert_array_equal(block_load.get_section_range(), np.array([1, 2, 4]))
    assert block_load.has_missing_section()
    assert_array_equal(block_load.get_missing_sections(), np.array([3]))
