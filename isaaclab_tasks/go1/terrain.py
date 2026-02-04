from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from isaacgym import terrain_utils

from isaaclab_tasks.go1.config import Go1TerrainCfg


@dataclass(frozen=True)
class TerrainGrid:
    cfg: Go1TerrainCfg
    proportions: tuple[float, ...]
    num_sub_terrains: int
    width_per_env_pixels: int
    length_per_env_pixels: int
    border: int
    tot_cols: int
    tot_rows: int
    row_indices: np.ndarray
    col_indices: np.ndarray
    x_offset: int
    rows_offset: int
    env_origins: np.ndarray


class Go1IsaacLabTerrain:
    """Terrain generator aligned with go1_gym/utils/terrain.py behavior."""

    def __init__(self, cfg: Go1TerrainCfg, num_robots: int, eval_cfg: Go1TerrainCfg | None = None) -> None:
        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.num_robots = num_robots
        self.mesh_type = cfg.mesh_type
        self.train_grid, self.eval_grid = self._build_grids()
        self.tot_rows = self.train_grid.tot_rows + (self.eval_grid.tot_rows if self.eval_grid else 0)
        self.tot_cols = max(
            self.train_grid.tot_cols,
            self.eval_grid.tot_cols if self.eval_grid else 0,
        )

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.vertices: np.ndarray | None = None
        self.triangles: np.ndarray | None = None

        if self.mesh_type not in {"none", "plane"}:
            self._initialize_terrains()
            if self.mesh_type == "trimesh":
                self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                    self.height_field_raw,
                    self.cfg.horizontal_scale,
                    self.cfg.vertical_scale,
                    self.cfg.slope_treshold,
                )
        else:
            self._populate_flat_origins(self.train_grid)
            if self.eval_grid:
                self._populate_flat_origins(self.eval_grid)

        self.env_origins = self.train_grid.env_origins

    def _build_grids(self) -> tuple[TerrainGrid, TerrainGrid | None]:
        train_grid = self._build_grid(self.cfg, x_offset=0, rows_offset=0)
        eval_grid = None
        if self.eval_cfg is not None:
            eval_grid = self._build_grid(
                self.eval_cfg,
                x_offset=train_grid.tot_rows,
                rows_offset=self.cfg.num_rows,
            )
        return train_grid, eval_grid

    def _build_grid(self, cfg: Go1TerrainCfg, x_offset: int, rows_offset: int) -> TerrainGrid:
        proportions = tuple(np.cumsum(cfg.terrain_proportions))
        num_sub_terrains = cfg.num_rows * cfg.num_cols
        width_per_env_pixels = int(cfg.terrain_length / cfg.horizontal_scale)
        length_per_env_pixels = int(cfg.terrain_width / cfg.horizontal_scale)
        border = int(cfg.border_size / cfg.horizontal_scale)
        tot_cols = int(cfg.num_cols * width_per_env_pixels) + 2 * border
        tot_rows = int(cfg.num_rows * length_per_env_pixels) + 2 * border
        row_indices = np.arange(x_offset, x_offset + tot_rows)
        col_indices = np.arange(0, tot_cols)
        env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3), dtype=np.float32)
        return TerrainGrid(
            cfg=cfg,
            proportions=proportions,
            num_sub_terrains=num_sub_terrains,
            width_per_env_pixels=width_per_env_pixels,
            length_per_env_pixels=length_per_env_pixels,
            border=border,
            tot_cols=tot_cols,
            tot_rows=tot_rows,
            row_indices=row_indices,
            col_indices=col_indices,
            x_offset=x_offset,
            rows_offset=rows_offset,
            env_origins=env_origins,
        )

    def _initialize_terrains(self) -> None:
        self._initialize_terrain(self.train_grid)
        if self.eval_grid is not None:
            self._initialize_terrain(self.eval_grid)

    def _initialize_terrain(self, grid: TerrainGrid) -> None:
        if grid.cfg.curriculum:
            self._curriculum(grid)
        elif grid.cfg.selected:
            self._selected_terrain(grid)
        else:
            self._randomized_terrain(grid)

    def _randomized_terrain(self, grid: TerrainGrid) -> None:
        for k in range(grid.num_sub_terrains):
            i, j = np.unravel_index(k, (grid.cfg.num_rows, grid.cfg.num_cols))
            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self._make_terrain(grid, choice, difficulty)
            self._add_terrain_to_map(grid, terrain, i, j)

    def _curriculum(self, grid: TerrainGrid) -> None:
        for j in range(grid.cfg.num_cols):
            for i in range(grid.cfg.num_rows):
                difficulty = i / grid.cfg.num_rows * grid.cfg.difficulty_scale
                choice = j / grid.cfg.num_cols + 0.001
                terrain = self._make_terrain(grid, choice, difficulty)
                self._add_terrain_to_map(grid, terrain, i, j)

    def _selected_terrain(self, grid: TerrainGrid) -> None:
        if not grid.cfg.terrain_kwargs:
            raise ValueError("terrain_kwargs must be provided when selected terrain is enabled.")
        terrain_type = grid.cfg.terrain_kwargs.get("type")
        terrain_args = grid.cfg.terrain_kwargs.get("kwargs", {})
        terrain_fn = self._resolve_terrain_callable(terrain_type)
        for k in range(grid.num_sub_terrains):
            i, j = np.unravel_index(k, (grid.cfg.num_rows, grid.cfg.num_cols))
            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=grid.width_per_env_pixels,
                length=grid.width_per_env_pixels,
                vertical_scale=grid.cfg.vertical_scale,
                horizontal_scale=grid.cfg.horizontal_scale,
            )
            terrain_fn(terrain, **terrain_args)
            self._add_terrain_to_map(grid, terrain, i, j)

    def _resolve_terrain_callable(self, terrain_type: Any) -> Callable[..., Any]:
        if callable(terrain_type):
            return terrain_type
        if isinstance(terrain_type, str):
            terrain_type_name = terrain_type.split(".")[-1]
            if hasattr(terrain_utils, terrain_type_name):
                return getattr(terrain_utils, terrain_type_name)
            return eval(terrain_type, {"terrain_utils": terrain_utils})
        raise ValueError("terrain_kwargs['type'] must be a callable or string.")

    def _make_terrain(self, grid: TerrainGrid, choice: float, difficulty: float) -> terrain_utils.SubTerrain:
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=grid.width_per_env_pixels,
            length=grid.width_per_env_pixels,
            vertical_scale=grid.cfg.vertical_scale,
            horizontal_scale=grid.cfg.horizontal_scale,
        )
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * (grid.cfg.max_platform_height - 0.05)
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        proportions = grid.proportions
        if choice < proportions[0]:
            if choice < proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
        elif choice < proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-0.05,
                max_height=0.05,
                step=grid.cfg.terrain_smoothness,
                downsampled_scale=0.2,
            )
        elif choice < proportions[3]:
            if choice < proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(
                terrain,
                step_width=0.31,
                step_height=step_height,
                platform_size=3.0,
            )
        elif choice < proportions[4]:
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,
                rectangle_min_size=1.0,
                rectangle_max_size=2.0,
                num_rectangles=20,
                platform_size=3.0,
            )
        elif choice < proportions[5]:
            terrain_utils.stepping_stones_terrain(
                terrain,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                max_height=0.0,
                platform_size=4.0,
            )
        elif choice < proportions[6]:
            pass
        elif choice < proportions[7]:
            pass
        elif choice < proportions[8]:
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-grid.cfg.terrain_noise_magnitude,
                max_height=grid.cfg.terrain_noise_magnitude,
                step=0.005,
                downsampled_scale=0.2,
            )
        elif choice < proportions[9]:
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-0.05,
                max_height=0.05,
                step=grid.cfg.terrain_smoothness,
                downsampled_scale=0.2,
            )
            terrain.height_field_raw[0 : terrain.length // 2, :] = 0

        return terrain

    def _add_terrain_to_map(self, grid: TerrainGrid, terrain: terrain_utils.SubTerrain, row: int, col: int) -> None:
        start_x = grid.border + row * grid.length_per_env_pixels + grid.x_offset
        end_x = grid.border + (row + 1) * grid.length_per_env_pixels + grid.x_offset
        start_y = grid.border + col * grid.width_per_env_pixels
        end_y = grid.border + (col + 1) * grid.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (row + 0.5) * grid.cfg.terrain_length + grid.x_offset * terrain.horizontal_scale
        env_origin_y = (col + 0.5) * grid.cfg.terrain_width
        env_origin_z = np.max(self.height_field_raw[start_x:end_x, start_y:end_y]) * terrain.vertical_scale
        grid.env_origins[row, col] = [env_origin_x, env_origin_y, env_origin_z]

    def _populate_flat_origins(self, grid: TerrainGrid) -> None:
        for j in range(grid.cfg.num_cols):
            for i in range(grid.cfg.num_rows):
                env_origin_x = (i + 0.5) * grid.cfg.terrain_length + grid.x_offset * grid.cfg.horizontal_scale
                env_origin_y = (j + 0.5) * grid.cfg.terrain_width
                grid.env_origins[i, j] = [env_origin_x, env_origin_y, 0.0]
