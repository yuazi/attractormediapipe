__all__ = [
    "SceneRenderer",
    "SceneState",
    "SnapshotController",
    "SnapshotExportResult",
    "SnapshotRequest",
    "export_attractor_snapshot",
]


def __getattr__(name):
    if name in {"SceneRenderer", "SceneState"}:
        from .scene import SceneRenderer, SceneState

        return SceneRenderer if name == "SceneRenderer" else SceneState
    if name in {"SnapshotController", "SnapshotExportResult", "SnapshotRequest", "export_attractor_snapshot"}:
        from .snapshot import SnapshotController, SnapshotExportResult, SnapshotRequest, export_attractor_snapshot

        if name == "SnapshotController":
            return SnapshotController
        if name == "SnapshotExportResult":
            return SnapshotExportResult
        if name == "SnapshotRequest":
            return SnapshotRequest
        return export_attractor_snapshot
    raise AttributeError(f"module 'renderer' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
