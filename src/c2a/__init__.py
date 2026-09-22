"""crawl2action: crawl e-commerce catalogs, build datasets, post-train, benchmark and serve."""

__version__ = "0.1.0"


class NotYetImplemented(NotImplementedError):
    """Raised by stubs for work scheduled in a later milestone (see docs/PLAN.md)."""

    def __init__(self, what: str, milestone: str) -> None:
        super().__init__(f"{what} is not implemented yet (planned for {milestone}).")
        self.what = what
        self.milestone = milestone
