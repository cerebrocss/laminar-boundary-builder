"""Self-contained package for 3D laminar surface depth workflows."""


def run_3d_surface_depth_pipeline(*args, **kwargs):
    """Lazy package-level shortcut that keeps GUI startup light."""

    from .core import run_3d_surface_depth_pipeline as _run_3d_surface_depth_pipeline

    return _run_3d_surface_depth_pipeline(*args, **kwargs)


__all__ = ["run_3d_surface_depth_pipeline"]
