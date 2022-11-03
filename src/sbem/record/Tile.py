from __future__ import annotations

from typing import TYPE_CHECKING

from tifffile import imread

if TYPE_CHECKING:  # pragma: no cover
    from typing import Dict

    from numpy.typing import ArrayLike

    from sbem.record.Section import Section


class Tile:
    def __init__(
        self,
        section: Section,
        tile_id: int,
        path: str,
        stage_x: float,
        stage_y: float,
        resolution_xy: float,
        unit: str = "nm",
    ):
        self._section = section
        self._tile_id = tile_id
        self._path = path
        self.x = stage_x
        self.y = stage_y
        self._resolution_xy = resolution_xy
        self._unit = unit

        if self._section is not None:
            self._section.add_tile(self)

    def get_tile_id(self) -> int:
        return self._tile_id

    def get_tile_data(self) -> ArrayLike:
        return imread(self._path)

    def get_tile_path(self) -> str:
        return self._path

    def get_resolution(self) -> float:
        return self._resolution_xy

    def get_unit(self) -> str:
        return self._unit

    def set_section(self, section: Section):
        self._section = section

    def get_section(self) -> Section:
        return self._section

    def to_dict(self) -> Dict:
        return {
            "tile_id": self._tile_id,
            "path": self._path,
            "stage_x": self.x,
            "stage_y": self.y,
            "resolution_xy": self._resolution_xy,
            "unit": self._unit,
        }

    @staticmethod
    def from_dict(dict: Dict):
        return Tile(
            section=None,
            tile_id=dict["tile_id"],
            path=dict["path"],
            stage_x=dict["stage_x"],
            stage_y=dict["stage_y"],
            resolution_xy=dict["resolution_xy"],
            unit=dict["unit"],
        )
