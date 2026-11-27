"""Pilot sequence generators and orthogonality verification for Massive MIMO."""

import numpy as np
from scipy.linalg import hadamard


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def generate_hadamard_pilots(
    nt: int,
    tp: int,
    normalized: bool = False,
    require_orthogonal: bool = True,
) -> np.ndarray:
    """Generate a BPSK pilot matrix X of shape (nt, tp) using Hadamard matrices.

    When tp >= nt and tp is a power of 2, taking the first `nt` rows of the
    order-`tp` Hadamard matrix produces exact row-orthogonal pilots:
        X @ X.conj().T = tp * I_{nt}  (unnormalized, entries in {+1, -1})
    or when normalized=True:
        X @ X.conj().T = (tp / nt) * I_{nt}  (entries in {+1/sqrt(nt), -1/sqrt(nt)}).

    When tp < nt and require_orthogonal=False (underdetermined pilot-overhead
    experiments), orthogonal rows are mathematically impossible because
    rank(X) <= tp < nt. In that case, we select `tp` orthogonal columns from an
    order-`next_pow2(nt)` Hadamard matrix so the pilot matrix retains full
    column rank `tp` with BPSK entries in {+1, -1}.
    """
    if nt <= 0 or tp <= 0:
        raise ValueError(f"nt and tp must be positive, got nt={nt}, tp={tp}")

    if tp < nt:
        if require_orthogonal:
            raise ValueError(
                f"Orthogonal pilot matrix X (shape {nt}x{tp}) satisfying "
                f"X @ X^H = Tp * I_{nt} is mathematically impossible when Tp < Nt "
                f"because rank(X) <= Tp = {tp} < Nt = {nt}."
            )
        order = 1 << (nt - 1).bit_length()
        h_mat = hadamard(order, dtype=np.float64)
        pilots = h_mat[:nt, :tp].astype(np.complex128)
    else:
        if not _is_power_of_two(tp):
            raise ValueError(
                f"Sylvester Hadamard construction requires tp to be a power of 2, got tp={tp}. "
                "Use pilot_type='dft' for arbitrary tp >= nt."
            )
        h_mat = hadamard(tp, dtype=np.float64)
        pilots = h_mat[:nt, :].astype(np.complex128)

    if normalized:
        pilots = pilots / np.sqrt(nt)

    return pilots


def generate_dft_pilots(
    nt: int,
    tp: int,
    normalized: bool = False,
    require_orthogonal: bool = True,
) -> np.ndarray:
    """Generate a complex DFT pilot matrix X of shape (nt, tp).

    For any tp >= nt, rows of the tp-point DFT matrix satisfy X @ X^H = tp * I_{nt}.
    """
    if nt <= 0 or tp <= 0:
        raise ValueError(f"nt and tp must be positive, got nt={nt}, tp={tp}")
    if tp < nt and require_orthogonal:
        raise ValueError(
            f"Orthogonal pilots require Tp >= Nt, got Nt={nt}, Tp={tp}."
        )

    m = np.arange(nt)[:, None]
    k = np.arange(tp)[None, :]
    order = max(nt, tp)
    pilots = np.exp(-1j * 2.0 * np.pi * m * k / order)

    if normalized:
        pilots = pilots / np.sqrt(nt)

    return pilots.astype(np.complex128)


def generate_pilots(
    nt: int,
    tp: int,
    method: str = "hadamard",
    normalized: bool = False,
    require_orthogonal: bool = True,
) -> np.ndarray:
    """Generate pilot matrix X of shape (nt, tp)."""
    method_lower = method.lower()
    if method_lower == "hadamard":
        if tp >= nt and not _is_power_of_two(tp):
            # Fall back to DFT if non-power-of-2 tp >= nt is explicitly requested via generic helper
            return generate_dft_pilots(nt, tp, normalized=normalized, require_orthogonal=require_orthogonal)
        return generate_hadamard_pilots(nt, tp, normalized=normalized, require_orthogonal=require_orthogonal)
    if method_lower == "dft":
        return generate_dft_pilots(nt, tp, normalized=normalized, require_orthogonal=require_orthogonal)
    raise ValueError(f"Unknown pilot method: {method}. Expected 'hadamard' or 'dft'.")


def verify_pilot_orthogonality(
    pilots: np.ndarray,
    atol: float = 1e-8,
) -> tuple[bool, float]:
    """Check whether X @ X^H is proportional to the identity matrix I_{Nt}.

    Returns:
        (is_orthogonal, max_abs_error_against_Tp_I)
    """
    nt, tp = pilots.shape
    gram = pilots @ pilots.conj().T
    expected = tp * np.eye(nt, dtype=gram.dtype)
    max_err = float(np.max(np.abs(gram - expected)))
    return max_err <= atol, max_err
