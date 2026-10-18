class UnsupportedPrimitiveError(NotImplementedError):
    """Raised when a primitive acts on a relaxed operand and no rule covers that case."""
