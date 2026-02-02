from enum import Enum


class PrecisionMode(Enum):
    """Supported precision modes."""

    FP32 = "32"
    FP16_MIXED = "16-mixed"
    BF16_MIXED = "bf16-mixed"
    FP16_TRUE = "16-true"
    BF16_TRUE = "bf16-true"
