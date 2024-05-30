from typing import Dict
import os
import pathlib
import pytest

from dlumt.io import *


class TestJsonIO:

    @pytest.fixture
    def dummy_dict(self) -> Dict:
        return {"dummy1": 5, "dummy2": "aaaa", "dummy5": 2.56, "dummy7": True}

    def test_writes_file_from_dict(
        self, dummy_dict: Dict, tmp_path: pathlib.Path
    ) -> None:

        filename = tmp_path / "filename.json"

        write_json_file(dummy_dict, str(filename))

        assert os.path.isfile(str(filename))

    def test_reads_file_from_dict(
        self, dummy_dict: Dict, tmp_path: pathlib.Path
    ) -> None:

        filename = tmp_path / "filename.json"

        write_json_file(dummy_dict, str(filename))
        read_dict = read_json_file(str(filename))

        assert read_dict == dummy_dict

    def test_raises_exception_when_not_json_extension_provided(
        self, tmp_path: pathlib.Path
    ) -> None:

        with pytest.raises(AssertionError):
            check_JSON_file_path_integrity(str(tmp_path / "data.xml"))

    def test_raises_exception_when_not_valid_basepath_provided(
        self, tmp_path: pathlib.Path
    ) -> None:

        with pytest.raises(AssertionError):
            check_JSON_file_path_integrity(
                str(tmp_path / "not_existing_folder" / "data.json")
            )
