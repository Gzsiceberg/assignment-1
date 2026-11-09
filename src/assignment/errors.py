"""Fatal infrastructure failures, distinct from recoverable tool errors."""


class InfrastructureError(Exception):
    """The environment cannot safely continue the current task."""
