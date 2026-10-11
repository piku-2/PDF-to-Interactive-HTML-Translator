"""Errors shared by the translator and the provider chain."""


class OutputTooLarge(Exception):
    """The model stopped before the structured payload was complete."""


class ProviderCallError(Exception):
    """This provider failed and the next one in the priority list should run."""


class AllProvidersFailed(RuntimeError):
    """Every configured provider failed for this request."""

    def __init__(self, errors: list[tuple[str, str]]) -> None:
        self.errors = errors
        detail = (
            "; ".join(f"{name}: {message}" for name, message in errors) or "原因不明"
        )
        super().__init__(f"全プロバイダーが失敗しました。{detail}")
