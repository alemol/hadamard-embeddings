"""
Executable verification of the manuscript's formal claims.

Every test corresponds to a numbered statement in the paper.  If this suite
passes, the implementation and the theory agree exactly.

Run:  python3 -I code/test_hadamard_text.py
"""

import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from hadamard_text import (  # noqa: E402
    Encoder, build_vocabulary, cosine, hadamard, hadamard_entry,
    positional_weights, sliding_windows, tokenize,
)

RNG = np.random.default_rng(20261026)
_results = []


def check(name, condition, detail=""):
    _results.append((name, bool(condition), detail))


# -------------------------------------------------- Definition 1 and Remark 1

def test_hadamard_properties():
    for k in range(0, 8):
        n = 1 << k
        H = hadamard(n).astype(np.float64)
        check(f"Def.1  H H^T = n I          (n={n})", np.allclose(H @ H.T, n * np.eye(n)))
        check(f"Def.1  H^T H = n I          (n={n})", np.allclose(H.T @ H, n * np.eye(n)))
        check(f"Def.1  entries in {{-1,+1}}   (n={n})", set(np.unique(H)) <= {-1.0, 1.0})
    n = 64
    H = hadamard(n)
    closed = np.array([[hadamard_entry(i, j) for j in range(n)] for i in range(n)])
    check("Rem.1  closed form H[i,j] = (-1)^<i,j>", np.array_equal(H, closed))


# ------------------------------------------------------------------ Theorem 1

def test_theorem_1_orthogonality():
    vocab = build_vocabulary([[f"w{i}" for i in range(50)]], alpha=1.0)
    enc = Encoder(vocab)
    cols = np.stack([enc.project(t) for t in vocab.terms])
    gram = cols @ cols.T
    off = gram - np.diag(np.diag(gram))
    check("Thm.1  distinct codes stay orthogonal", np.allclose(off, 0.0, atol=1e-12),
          f"max |off-diagonal| = {np.abs(off).max():.2e}")
    check("Thm.1  rescaled projections are unit norm",
          np.allclose(np.diag(gram), 1.0, atol=1e-12))


# ------------------------------------------------------ Section 3.3, OOV room

def test_order_leaves_room_for_oov():
    worst = []
    for size in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 592, 1017, 1033, 1024, 2048]:
        vocab = build_vocabulary([[f"w{i}" for i in range(size)]], alpha=1.0)
        ok = vocab.order >= vocab.n_symbols
        worst.append((size, len(vocab.terms), vocab.order, ok))
        check(f"S.3.3  n >= |V|+1            (|V|={len(vocab.terms)})", ok,
              f"n={vocab.order}, symbols={vocab.n_symbols}")
    exact_pow2 = [w for w in worst if w[1] in (1, 2, 4, 8, 16, 1024, 2048)]
    check("S.3.3  holds when |V| is an exact power of two",
          all(w[3] for w in exact_pow2))


# -------------------------------------------- Section 3.4, coverage criterion

def test_vocabulary_reduction():
    corpus = [["a"] * 60, ["b"] * 25, ["c"] * 10, ["d"] * 5]
    vocab = build_vocabulary(corpus, alpha=0.6, reserve_oov=False)
    check("S.3.4  alpha=0.6 retains the minimal prefix", vocab.terms == ("a",),
          f"terms={vocab.terms}, theta={vocab.threshold}")
    tied = [["x"] * 10, ["y"] * 10, ["z"] * 1]
    v2 = build_vocabulary(tied, alpha=0.5, reserve_oov=False)
    check("S.3.4  ties at theta are retained", set(v2.terms) == {"x", "y"},
          f"terms={v2.terms}")


# ------------------------------------------------ Section 3.7, worked example

def test_worked_example():
    """The paper's own numbers must come out of the code."""
    vocab = build_vocabulary([["the", "dog", "chased", "cat"]], alpha=1.0,
                             reserve_oov=False)
    # Fix the column order used in Section 3.7.
    order = ("the", "dog", "chased", "cat")
    vocab = type(vocab)(terms=order, index={t: i for i, t in enumerate(order)},
                        threshold=1, size_before=4, order=4)
    enc = Encoder(vocab, window=3)

    s1 = enc.encode("the dog chased the cat".split())
    s2 = enc.encode("the cat chased the dog".split())
    check("S.3.7  x_s1 = [4.5, 1.5, 0.5, -0.5]", np.allclose(s1, [4.5, 1.5, 0.5, -0.5]),
          f"got {np.round(s1, 4).tolist()}")
    check("S.3.7  x_s2 = [4.5, 1.5, -0.5, 0.5]", np.allclose(s2, [4.5, 1.5, -0.5, 0.5]),
          f"got {np.round(s2, 4).tolist()}")
    check("S.3.7  weights (1,2,3,2,1) for L=5, lambda=3",
          np.array_equal(positional_weights(5, 3), [1, 2, 3, 2, 1]))


# ----------------------------------------------------------- Proposition 2

def test_proposition_2():
    corpus = [[f"t{RNG.integers(0, 40)}" for _ in range(RNG.integers(4, 25))]
              for _ in range(400)]
    vocab = build_vocabulary(corpus, alpha=0.9)
    enc = Encoder(vocab)
    worst = 0.0
    tested = 0
    for toks in corpus:
        emb = enc.encode(toks)
        if emb is None:
            continue
        tested += 1
        worst = max(worst, float(np.abs(emb - enc.project_vector(enc.weighted_counts(toks))).max()))
    check("Prop.2  x_s = Phi(sum_i c_i b_ti)", worst < 1e-9,
          f"max deviation over {tested} sentences = {worst:.2e}")

    # Invariance: any permutation preserving the token-weight multiset.
    toks = list("abcde")
    emb = enc.encode(toks)
    mirrored = enc.encode(toks[::-1])          # weights are symmetric
    check("Prop.2  invariant under weight-preserving permutation",
          np.allclose(emb, mirrored), f"max diff = {np.abs(emb - mirrored).max():.2e}")


# ------------------------------------------------------------- Equation (9)

def test_equation_9_cosine_equivalence():
    corpus = [[f"t{RNG.integers(0, 60)}" for _ in range(RNG.integers(4, 30))]
              for _ in range(300)]
    vocab = build_vocabulary(corpus, alpha=0.9)
    enc = Encoder(vocab)
    pairs, worst = 0, 0.0
    for a, b in zip(corpus[::2], corpus[1::2]):
        ea, eb = enc.encode(a), enc.encode(b)
        if ea is None or eb is None:
            continue
        pairs += 1
        lhs = cosine(ea, eb)
        rhs = cosine(enc.weighted_counts(a), enc.weighted_counts(b))
        worst = max(worst, abs(lhs - rhs))
    check("Eq.(9)  cos(x_s,x_s') = cos(weighted BoW)", worst < 1e-9,
          f"max deviation over {pairs} pairs = {worst:.2e}")


# ------------------------------------------- Section 2.2, exact invertibility

def test_reversibility():
    corpus = [[f"t{RNG.integers(0, 30)}" for _ in range(RNG.integers(4, 20))]
              for _ in range(200)]
    vocab = build_vocabulary(corpus, alpha=0.9)
    enc = Encoder(vocab)
    worst, tested = 0.0, 0
    for toks in corpus:
        emb = enc.encode(toks)
        if emb is None:
            continue
        tested += 1
        worst = max(worst, float(np.abs(enc.invert(emb) - enc.weighted_counts(toks)).max()))
    check("S.2.2  Phi is exactly invertible", worst < 1e-9,
          f"max deviation over {tested} sentences = {worst:.2e}")


# --------------------------------------------------------- Section 3.1, misc

def test_windows_and_determinism():
    check("S.3.1  m = L - lambda + 1",
          all(len(sliding_windows(["x"] * L, 3)) == L - 2 for L in range(3, 30)))
    check("S.3.1  sentences shorter than lambda yield no window",
          sliding_windows(["a", "b"], 3) == [])
    vocab = build_vocabulary([tokenize("the quick brown fox jumps over the lazy dog")], alpha=1.0)
    enc = Encoder(vocab)
    toks = tokenize("the quick brown fox")
    check("S.3.8  encoding is bit-for-bit deterministic",
          np.array_equal(enc.encode(toks), enc.encode(toks)))


def main():
    for fn in (test_hadamard_properties, test_theorem_1_orthogonality,
               test_order_leaves_room_for_oov, test_vocabulary_reduction,
               test_worked_example, test_proposition_2,
               test_equation_9_cosine_equivalence, test_reversibility,
               test_windows_and_determinism):
        fn()

    width = max(len(n) for n, _, _ in _results)
    failed = 0
    for name, ok, detail in _results:
        if not ok:
            failed += 1
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {name:<{width}}"
        if detail and (not ok or "deviation" in detail or "off-diagonal" in detail):
            line += f"   {detail}"
        print(line)
    print(f"\n  {len(_results) - failed}/{len(_results)} verificaciones pasan")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
