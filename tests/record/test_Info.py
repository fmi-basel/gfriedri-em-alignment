from unittest import TestCase

from sbem.record.Info import Info


class InfoTest(TestCase):
    def test_info(self):
        info = Info("name", "license")

        assert info.get_format_version() == "0.1.0"
        assert info.get_name() == "name"
        assert info.get_license() == "license"
