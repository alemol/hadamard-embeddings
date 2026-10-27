"""
Training-free sentence representation via exact Hadamard projection.

This module is a reference implementation written directly from the method
description in the manuscript.  Each function cites the section it implements,
so that the paper is the specification and this file is its executable form.

Sections referenced:
  3.1  Lexical Indexing and Sliding-Window Decomposition
  3.2  The Hadamard Basis
  3.3  Dimensional Compatibility
  3.4  Vocabulary Reduction by a Frequency-Coverage Criterion
  3.5  The Hadamard Projection
  3.6  From Token Projections to Sentence Vectors
  3.8  Properties of the Representation (Proposition 2)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

OOV = "<OOV>"
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------- Section 3.1

def tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens.

    Section 3.1.  No stopword removal: the frequency-coverage criterion of
    Section 3.4 already prunes high-frequency terms, and no stopword list
    exists for every language the method targets.
    """
    return _TOKEN_RE.findall(text.lower())


def sliding_windows(tokens: list[str], window: int = 3, step: int = 1) -> list[list[str]]:
    """Overlapping observation windows, Equation (1).

    Section 3.1.  A sentence shorter than the window yields no window and is
    excluded downstream.  With step 1 the count is m = L - window + 1.
    """
    if len(tokens) < window:
        return []
    return [tokens[i:i + window] for i in range(0, len(tokens) - window + 1, step)]


def positional_weights(length: int, window: int = 3) -> np.ndarray:
    """Weights c_i of Proposition 2, for step 1.

    c_i = min(i, lambda, L - lambda + 1, L - i + 1), one-based in the paper.
    """
    if length < window:
        return np.zeros(length, dtype=np.int64)
    return np.array(
        [min(i, window, length - window + 1, length - i + 1)
         for i in range(1, length + 1)],
        dtype=np.int64,
    )


# ---------------------------------------------------------------- Section 3.2

def hadamard(order: int) -> np.ndarray:
    """Sylvester construction, Definition 2.  Order must be a power of two."""
    if order < 1 or order & (order - 1):
        raise ValueError(f"order must be a positive power of two, got {order}")
    H = np.ones((1, 1), dtype=np.int8)
    while H.shape[0] < order:
        H = np.block([[H, H], [H, -H]])
    return H


def hadamard_entry(i: int, j: int) -> int:
    """Closed form of Remark 1: H[i,j] = (-1)^<i,j> over the bit expansions."""
    return -1 if bin(i & j).count("1") % 2 else 1


# ---------------------------------------------------------------- Section 3.4

@dataclass(frozen=True)
class Vocabulary:
    """Reduced vocabulary and the Hadamard order it induces."""

    terms: tuple[str, ...]          # retained terms, frequency-descending
    index: dict[str, int]           # Section 3.1, the bijection idx
    threshold: int                  # theta of Section 3.4
    size_before: int                # |V_0|
    order: int                      # n of Equation (4)

    @property
    def oov_index(self) -> int:
        return len(self.terms)

    @property
    def n_symbols(self) -> int:
        """Symbols the encoder can emit: retained terms plus the OOV symbol."""
        return len(self.terms) + 1


def build_vocabulary(token_lists, alpha: float = 0.6, reserve_oov: bool = True) -> Vocabulary:
    """Frequency-coverage reduction, Section 3.4, and the order n of Section 3.3.

    Terms are sorted in non-increasing order of frequency.  k* is the smallest
    rank whose cumulative share of occurrences reaches alpha; theta is the
    frequency at that rank; the retained vocabulary is every term with
    frequency at least theta.  Ties at theta are retained, so |V| >= k* and the
    result does not depend on the order in which equally frequent terms appear.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")

    freq = Counter(t for toks in token_lists for t in toks)
    if not freq:
        raise ValueError("empty corpus")

    ordered = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    total = sum(f for _, f in ordered)

    cumulative = 0
    threshold = ordered[-1][1]
    for _, f in ordered:
        cumulative += f
        if cumulative / total >= alpha:
            threshold = f
            break

    terms = tuple(t for t, f in ordered if f >= threshold)
    n_symbols = len(terms) + (1 if reserve_oov else 0)
    order = 1 << math.ceil(math.log2(n_symbols)) if n_symbols > 1 else 1

    return Vocabulary(
        terms=terms,
        index={t: i for i, t in enumerate(terms)},
        threshold=threshold,
        size_before=len(freq),
        order=order,
    )


# ---------------------------------------------------------------- Section 3.5

class Encoder:
    """Hadamard projection and sentence aggregation, Sections 3.5 and 3.6."""

    def __init__(self, vocab: Vocabulary, window: int = 3, step: int = 1):
        self.vocab = vocab
        self.window = window
        self.step = step
        self.H = hadamard(vocab.order)
        # Phi(b_t) = (1/sqrt(n)) H_n b_t, i.e. a scaled column of H_n.
        self.scale = 1.0 / math.sqrt(vocab.order)

    def code_index(self, token: str) -> int:
        """Index of a token, falling back to the reserved OOV symbol."""
        return self.vocab.index.get(token, self.vocab.oov_index)

    def project(self, token: str) -> np.ndarray:
        """Phi applied to the one-hot code of a token: a column of H_n."""
        return self.scale * self.H[:, self.code_index(token)].astype(np.float64)

    def encode(self, tokens: list[str]) -> np.ndarray | None:
        """Sentence vector of Section 3.6: a plain sum, within and across windows.

        Returns None when the sentence is shorter than the window and therefore
        yields no observation window.
        """
        windows = sliding_windows(tokens, self.window, self.step)
        if not windows:
            return None
        out = np.zeros(self.vocab.order, dtype=np.float64)
        for w in windows:
            for t in w:
                out += self.project(t)
        return out

    def weighted_counts(self, tokens: list[str]) -> np.ndarray:
        """Sum_i c_i b_{t_i}, the right-hand side of Proposition 2."""
        c = positional_weights(len(tokens), self.window)
        out = np.zeros(self.vocab.order, dtype=np.float64)
        for weight, token in zip(c, tokens):
            out[self.code_index(token)] += weight
        return out

    def project_vector(self, v: np.ndarray) -> np.ndarray:
        """Phi applied to an arbitrary coordinate vector."""
        return self.scale * (self.H.astype(np.float64) @ v)

    def invert(self, embedding: np.ndarray) -> np.ndarray:
        """Phi inverse, Section 2.2: n^{-1} H^T applied to the scaled vector."""
        return (self.H.astype(np.float64).T @ embedding) * self.scale


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))
