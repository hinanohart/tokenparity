"""Three honest axes for tokenizer evaluation.

Axis A — Downstream Transferability  (axes/transfer.py)
Axis B — Information Bottleneck       (axes/ib.py)
Axis C — Per-Domain Normalized Reconstruction  (axes/reconstruction.py)

Scalar fusion across axes is deliberately absent from this package.
"""

from tokenparity.axes.ib import information_bottleneck
from tokenparity.axes.reconstruction import (
    average_rank,
    normalize_within_domain,
    reconstruction_error,
)
from tokenparity.axes.transfer import aggregate_transferability, transferability

__all__ = [
    "transferability",
    "aggregate_transferability",
    "information_bottleneck",
    "reconstruction_error",
    "normalize_within_domain",
    "average_rank",
]
