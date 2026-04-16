"""Output-stage memory planning for WorldMirror inference artifacts."""

from dataclasses import dataclass
from math import ceil
from typing import Optional


_BYTES_PER_GAUSSIAN_SOURCE = 192
_DEFAULT_MEMORY_BUDGET_GB = 4.0
_MIN_CHUNK_ELEMENTS = 16_384


@dataclass(frozen=True)
class OutputSavePlan:
    """Concrete chunking plan for bounded-memory output serialization."""

    num_frames: int
    height: int
    width: int
    memory_budget_gb: float
    gaussian_chunk_elements: int
    point_frame_chunk_size: int
    estimated_gaussian_source_gb: float

    @property
    def pixels_per_frame(self) -> int:
        return self.height * self.width

    @property
    def total_pixels(self) -> int:
        return self.num_frames * self.pixels_per_frame

    @property
    def gaussian_chunks(self) -> int:
        if self.total_pixels == 0:
            return 0
        return ceil(self.total_pixels / self.gaussian_chunk_elements)


def build_output_save_plan(
    num_frames: int,
    height: int,
    width: int,
    *,
    memory_budget_gb: Optional[float] = _DEFAULT_MEMORY_BUDGET_GB,
    save_chunk_frames: Optional[int] = None,
) -> OutputSavePlan:
    """Build a conservative output-stage chunk plan.

    The budget only covers transient host-side serialization buffers. It does
    not include model activations or the final artifact size, which are governed
    by inference resolution, frame count, and output compression settings.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    memory_budget_gb = (
        _DEFAULT_MEMORY_BUDGET_GB
        if memory_budget_gb is None or memory_budget_gb <= 0
        else float(memory_budget_gb)
    )

    pixels_per_frame = int(height) * int(width)
    total_pixels = int(num_frames) * pixels_per_frame
    estimated_gaussian_source_gb = (
        total_pixels * _BYTES_PER_GAUSSIAN_SOURCE / (1024**3)
    )

    if save_chunk_frames is not None:
        if save_chunk_frames <= 0:
            raise ValueError("save_chunk_frames must be positive")
        point_frame_chunk_size = min(int(save_chunk_frames), int(num_frames))
        gaussian_chunk_elements = min(total_pixels, point_frame_chunk_size * pixels_per_frame)
    else:
        # Use a fraction of the budget for raw chunk tensors; voxel grouping and
        # writer buffers need additional headroom in the same process.
        chunk_budget_bytes = memory_budget_gb * (1024**3) * 0.20
        gaussian_chunk_elements = int(chunk_budget_bytes // _BYTES_PER_GAUSSIAN_SOURCE)
        gaussian_chunk_elements = max(_MIN_CHUNK_ELEMENTS, gaussian_chunk_elements)
        gaussian_chunk_elements = min(total_pixels, gaussian_chunk_elements)
        point_frame_chunk_size = max(1, gaussian_chunk_elements // pixels_per_frame)
        point_frame_chunk_size = min(point_frame_chunk_size, int(num_frames))

    return OutputSavePlan(
        num_frames=int(num_frames),
        height=int(height),
        width=int(width),
        memory_budget_gb=memory_budget_gb,
        gaussian_chunk_elements=int(gaussian_chunk_elements),
        point_frame_chunk_size=int(point_frame_chunk_size),
        estimated_gaussian_source_gb=float(estimated_gaussian_source_gb),
    )
