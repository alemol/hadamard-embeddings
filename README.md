# `hadamard_text.py` — training-free sentence representation by exact Hadamard projection

Reference implementation of the method described in *A Hadamard-Based Training-Free
Sentence Representation for Efficient and Reproducible Text Processing*
(Aldana-Bobadilla & Molina-Villegas).

This repository contains four files: `hadamard_text.py`, `test_hadamard_text.py`,
`LICENSE` and this README. Releases are archived in Zenodo, where each one receives
its own DOI.

The module is written so that **the paper is the specification and this file is its
executable form**. Every function names the section it implements, and every numbered
claim of the paper has a corresponding check in the verification suite.

## Requirements

NumPy only. No training, no stored parameters, no model download, no network access.

To verify the implementation against the paper's formal claims:

```bash
python3 -I test_hadamard_text.py
```

## Quick start

```python
from hadamard_text import tokenize, build_vocabulary, Encoder, cosine

corpus = ["the dog chased the cat", "the cat chased the dog"]
toks   = [tokenize(t) for t in corpus]

vocab = build_vocabulary(toks, alpha=0.6)   # frequency-coverage reduction, Section 3.4
enc   = Encoder(vocab, window=3)            # lambda = 3

x1, x2 = enc.encode(toks[0]), enc.encode(toks[1])
print(cosine(x1, x2))
```

## API

### Tokenization and windows (Section 3.1)

| Function | Returns |
|---|---|
| `tokenize(text)` | lowercased `\w+` tokens. **No stopword removal** — the coverage criterion of Section 3.4 already prunes high-frequency terms, and no stopword list exists for every target language. |
| `sliding_windows(tokens, window=3, step=1)` | overlapping windows, Equation (1). A sentence shorter than `window` yields `[]`. With step 1 the count is `m = L - window + 1`. |
| `positional_weights(length, window=3)` | the weights `c_i = min(i, lambda, L-lambda+1, L-i+1)` of Proposition 2. |

### The Hadamard basis (Sections 3.2 and 3.3)

| Function | Returns |
|---|---|
| `hadamard(order)` | Sylvester construction, Definition 2. `order` must be a power of two. |
| `hadamard_entry(i, j)` | closed form of Remark 1, `(-1)^<i,j>` over the bit expansions — `O(n)` bit operations per token, no multiplications. |

### Vocabulary (Section 3.4)

`build_vocabulary(token_lists, alpha=0.6, reserve_oov=True) -> Vocabulary`

Terms are sorted by non-increasing frequency; `k*` is the smallest rank whose
cumulative share of occurrences reaches `alpha`; `theta` is the frequency at that
rank; every term with frequency at least `theta` is retained. **Ties at `theta` are
retained**, so `|V| >= k*` and the result does not depend on the order in which
equally frequent terms appear.

The returned `Vocabulary` carries `terms`, `index`, `threshold`, `size_before` (`|V_0|`)
and `order` (`n`), plus `oov_index` and `n_symbols`.

`reserve_oov=True` counts the OOV symbol when choosing `n`, so
`n = 2^ceil(log2(|V| + 1))` and `n >= |V| + 1` **including when `|V|` is an exact power
of two**. This is required for correctness: without the reserved slot, the OOV index
equals `n` and collides with term 0. Pass `reserve_oov=False` only to reproduce
illustrations that assume no OOV symbol, such as the worked example of Section 3.7.

### Encoding (Sections 3.5 and 3.6)

`Encoder(vocab, window=3, step=1)`

| Method | Description |
|---|---|
| `project(token)` | `Phi(b_t) = n^{-1/2} H_n b_t`, a scaled column of `H_n`. |
| `encode(tokens)` | sentence vector of Section 3.6: a **plain sum**, within and across windows. Returns `None` when the sentence is shorter than `window`. |
| `weighted_counts(tokens)` | `sum_i c_i b_{t_i}`, the right-hand side of Proposition 2. |
| `project_vector(v)` | `Phi` applied to an arbitrary coordinate vector. |
| `invert(embedding)` | `Phi^{-1} = n^{-1} H^T`, exact inversion (Section 2.2). |

`cosine(a, b)` returns 0.0 if either vector is zero.

The aggregation is a plain sum, not a mean and not an L2 normalization per window.
This matters: a nonlinear per-window aggregation breaks Proposition 2, Equation (9),
the worked example of Section 3.7 and the linearity property.

## Verified properties

`test_hadamard_text.py` runs 55 checks, all of which pass. Each one corresponds to a
numbered statement in the paper; the deviations below are the measured worst cases, in
double precision.

| Claim | Result |
|---|---|
| Proposition 2, `x_s = Phi(sum_i c_i b_{t_i})` | max deviation over 400 sentences = `0.00e+00` |
| Equation (9), `cos(x_s, x_s') = cos` of the weighted bags | max deviation over 150 pairs = `0.00e+00` |
| Section 3.7 worked example, `x_s1 = [4.5, 1.5, 0.5, -0.5]` | exact |
| Section 2.2, `Phi` is exactly invertible | max deviation = `4.40e-15` |
| Section 3.3, `n >= |V| + 1` | holds, including exact powers of two |
| Theorem 1, distinct codes stay orthogonal | max off-diagonal = `0.00e+00` |
| Definition 1, `H H^T = n I`, entries in `{-1,+1}` | holds for `n = 1 … 128` |
| Remark 1, closed form equals the Sylvester matrix | exact at `n = 64` |

The Section 3.7 test fixes the column order `("the", "dog", "chased", "cat")` and passes
`reserve_oov=False`, because that illustration takes `|V| = 4` and `n = 4`; with a
reserved OOV slot the order would be 8.

Encoding is deterministic: the same input yields bit-for-bit identical output across
runs, processes and machines, with no seed to set.

## On the choice of `lambda`

The default is `window=3`, matching the paper. Note that the sliding window with unit
step and additive aggregation is, by Proposition 2, **only a positional reweighting
that discounts tokens near the sentence boundaries**: `encode` is invariant under any
permutation preserving the multiset of `(token, weight)` pairs.

`window=1` makes every `c_i` equal to 1, reducing the representation to a projected
bag of words. It is strictly less discriminative on the toy example above — the two
sentences share a bag of words and so receive identical vectors — but on the
experiments reported in the paper it was the better choice at every `alpha` tested, and
it never discards a sentence for being shorter than the window. Set `window`
deliberately rather than relying on the default.

## License

MIT. See [`LICENSE`](LICENSE). You may use, modify and redistribute this code, including
commercially, provided the copyright notice and the permission notice are retained.

The licence covers this implementation only. The corpora used in the paper carry their
own terms, which are set out in the article's data availability statement.

## Citation

Please cite the article. If you refer to this implementation specifically, cite it
through the DOI of the corresponding Zenodo release.
