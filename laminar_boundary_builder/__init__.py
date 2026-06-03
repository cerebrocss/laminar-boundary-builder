"""Self-contained package for building laminar boundaries and depth fields."""


def run_laminar_boundary_pipeline(*args, **kwargs):
    """Lazy package-level shortcut that keeps GUI startup light."""

    from .core import run_laminar_boundary_pipeline as _run_laminar_boundary_pipeline

    return _run_laminar_boundary_pipeline(*args, **kwargs)


__all__ = ["run_laminar_boundary_pipeline"]
