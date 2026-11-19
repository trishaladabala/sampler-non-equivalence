"""Hand-constructed tests for the perplexity estimators, masking, entropy and PPL@H interpolation.

Run: python -m pytest tests/test_ppl_estimators.py -v
Each test also documents a property the manuscript relies on.
"""
import math
import os
import sys
import numpy as np

try:
    import pytest
except ImportError:
    class _Approx:
        def __init__(self, val, abs=None, rel=1e-6):
            self.val = val
            self.abs = abs
            self.rel = rel
        def __eq__(self, other):
            if self.abs is not None:
                return math.isclose(self.val, other, abs_tol=self.abs)
            return math.isclose(self.val, other, rel_tol=self.rel)
        def __ne__(self, other):
            return not self.__eq__(other)
        def __repr__(self):
            return f"approx({self.val})"
    class _PytestFallback:
        approx = _Approx
        from contextlib import contextmanager
        @staticmethod
        @contextmanager
        def raises(exc):
            try:
                yield
            except exc:
                return
            raise AssertionError(f"Expected exception {exc} was not raised")
    pytest = _PytestFallback()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "analysis"))
import metrics_ref as M  # noqa: E402


# ---------------------------------------------------------------- Jensen direction

def test_jensen_mean_of_ppl_is_at_least_exp_of_mean_loss():
    S, T = [1.0 * 10, 3.0 * 10], [10, 10]          # per-token losses 1 and 3
    assert M.mean_seq_ppl(S, T) == pytest.approx((math.e + math.e ** 3) / 2)
    assert M.exp_mean_seq_ppl(S, T) == pytest.approx(math.e ** 2)
    assert M.mean_seq_ppl(S, T) > M.exp_mean_seq_ppl(S, T)   # averaging PPLs inflates, never deflates


def test_equal_lengths_corpus_equals_exp_mean():
    S, T = [5.0, 15.0, 9.0], [5, 5, 5]
    assert M.corpus_ppl(S, T) == pytest.approx(M.exp_mean_seq_ppl(S, T))


def test_length_weighting_can_move_corpus_ppl_either_way():
    # long low-loss sample + short high-loss sample: corpus < exp-mean
    S, T = [1.0 * 100, 4.0 * 2], [100, 2]
    assert M.corpus_ppl(S, T) < M.exp_mean_seq_ppl(S, T)
    # long high-loss sample + short low-loss sample: corpus > exp-mean
    S, T = [4.0 * 100, 1.0 * 2], [100, 2]
    assert M.corpus_ppl(S, T) > M.exp_mean_seq_ppl(S, T)


def test_corpus_can_exceed_mean_seq_ppl_with_unequal_lengths():
    # Shows that "corpus <= mean of sample PPLs" is NOT a theorem once lengths differ.
    S, T = [3.0 * 1000, 0.0 * 1], [1000, 1]
    assert M.corpus_ppl(S, T) > M.mean_seq_ppl(S, T) * 0.99  # ~e^3 vs (e^3+1)/2
    assert M.corpus_ppl(S, T) > M.mean_seq_ppl(S, T)


def test_empty_samples_rejected():
    with pytest.raises(ValueError):
        M.corpus_ppl([1.0, 0.0], [3, 0])


# ---------------------------------------------------------------- masking

def test_post_eos_positions_excluded_from_numerator_and_denominator():
    ids = [5, 6, 1, 7, 7, 7, 0, 0]          # eos=1 at position 2, pad=0
    nll = [1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 9.0, 9.0]
    m = M.valid_mask(ids, eos_id=1, pad_id=0, count_eos=True)
    assert m.tolist() == [True, True, True, False, False, False, False, False]
    S, T = M.sequence_stats(nll, m)
    assert (S, T) == (3.0, 3)


def test_unmasked_repetition_tail_lowers_ppl():
    # A cheap, repetitive post-EOS tail drags PPL down if it is not masked.
    ids = [5, 6, 1] + [7] * 50
    nll = [3.0, 3.0, 3.0] + [0.05] * 50
    good = M.sequence_stats(nll, M.valid_mask(ids, eos_id=1, pad_id=0))
    bad = M.sequence_stats(nll, np.ones(len(ids), bool))
    assert M.corpus_ppl([good[0]], [good[1]]) == pytest.approx(math.e ** 3)
    assert M.corpus_ppl([bad[0]], [bad[1]]) < 2.0


def test_pad_never_counted():
    m = M.valid_mask([4, 4, 0, 0], eos_id=1, pad_id=0)
    assert m.tolist() == [True, True, False, False]


# ---------------------------------------------------------------- entropy definitions

def test_per_sample_vs_pooled_entropy_differ():
    a, b = [1, 2, 3, 4], [5, 6, 7, 8]
    assert M.per_sample_unigram_entropy([a, b]) == pytest.approx(math.log(4))
    assert M.pooled_unigram_entropy([a, b]) == pytest.approx(math.log(8))


def test_per_sample_entropy_bounded_by_log_length():
    rng = np.random.default_rng(0)
    xs = [rng.integers(0, 50000, size=1024).tolist() for _ in range(4)]
    assert M.per_sample_unigram_entropy(xs) <= math.log(1024) + 1e-9


def test_rep4():
    assert M.rep_n([1, 2, 3, 4, 1, 2, 3, 4, 1], 4) == pytest.approx(1 - 4 / 6)
    assert M.rep_n([1, 2, 3, 4, 5], 4) == 0.0


# ---------------------------------------------------------------- PPL@H interpolation

PTS = [(5.30, 45.0), (5.20, 30.0), (5.10, 20.0)]


def test_interpolation_is_log_linear():
    v = M.frontier_ppl_at(PTS, 5.15)
    assert v == pytest.approx(math.sqrt(30.0 * 20.0))


def test_interpolation_hits_observed_points():
    assert M.frontier_ppl_at(PTS, 5.20) == pytest.approx(30.0)


def test_extrapolation_refused():
    with pytest.raises(M.ExtrapolationError):
        M.frontier_ppl_at(PTS, 5.35)
    with pytest.raises(M.ExtrapolationError):
        M.frontier_ppl_at(PTS, 5.00)


def test_monotonicity_check():
    assert M.is_monotone(PTS)
    assert not M.is_monotone([(5.0, 10.0), (5.1, 8.0)])


def test_frontier_residual_sign():
    # A point on the frontier has zero residual; below it is negative.
    assert M.frontier_residual(PTS, 5.20, 30.0) == pytest.approx(0.0)
    assert M.frontier_residual(PTS, 5.20, 27.0) < 0


def test_relative_change_formula():
    assert M.relative_change(0.1578, 0.0772) == pytest.approx(51.07, abs=0.01)
    assert M.relative_change(0.1706, 0.0724) == pytest.approx(57.56, abs=0.01)
    assert M.relative_change(0.1962, 0.0628) == pytest.approx(67.99, abs=0.01)


# ---------------------------------------------------------------- mutation checks
# These confirm the tests would catch common implementation errors.

def _buggy_corpus_mean_of_ppl(S, T):
    return M.mean_seq_ppl(S, T)


def test_mutation_mean_of_ppl_detected():
    S, T = [1.0 * 10, 3.0 * 10], [10, 10]
    assert _buggy_corpus_mean_of_ppl(S, T) != pytest.approx(M.corpus_ppl(S, T))


def test_mutation_missing_eos_mask_detected():
    ids = [5, 1, 7, 7]
    nll = [2.0, 2.0, 0.0, 0.0]
    masked = M.sequence_stats(nll, M.valid_mask(ids, 1, 0))
    unmasked = M.sequence_stats(nll, np.ones(4, bool))
    assert masked != unmasked


def test_jcd_ctw_invariants():
    from experiments import non_identifiability_suite as ni

    ref = [f"Natural language reference sentence {i} with diverse vocabulary." for i in range(30)]
    div = [f"Another diverse language sentence {i} with distinct phrasing." for i in range(30)]
    rep = ["the the the the the the the the the the the the" for _ in range(30)]

    r_self = ni.compute_jcd_ctw(ref, ref, depth=4)
    r_div = ni.compute_jcd_ctw(div, ref, depth=4)
    r_sym = ni.compute_jcd_ctw(ref, div, depth=4)
    r_rep = ni.compute_jcd_ctw(rep, ref, depth=4)

    assert r_self["jcd_bits_per_byte"] == pytest.approx(0.0, abs=1e-9)
    assert r_div["jcd_bits_per_byte"] == pytest.approx(r_sym["jcd_bits_per_byte"], abs=1e-9)
    assert r_rep["jcd_bits_per_byte"] > r_div["jcd_bits_per_byte"] * 2.0


def test_within_vs_cross_diversity_and_jsd():
    from experiments import non_identifiability_suite as ni

    identical = [list(range(1, 51)) for _ in range(20)]
    disjoint = [[i * 100 + (k % 25) for k in range(50)] for i in range(20)]

    da = ni.compute_within_and_cross_diversity(identical)
    db = ni.compute_within_and_cross_diversity(disjoint)

    assert da["within_distinct4"] == pytest.approx(1.0 - da["rep4"])
    assert db["within_distinct4"] == pytest.approx(1.0 - db["rep4"])
    assert da["jsd_inter_sample"] == pytest.approx(0.0, abs=1e-9)
    assert db["jsd_inter_sample"] > 2.5
    assert db["rep4"] > da["rep4"]
    assert db["cross_distinct2"] > da["cross_distinct2"] * 10.0


def test_self_bleu_and_position_entropy():
    from experiments import non_identifiability_suite as ni

    identical = [list(range(1, 33)) for _ in range(12)]
    disjoint = [[i * 100 + k for k in range(32)] for i in range(12)]

    sba = ni.compute_self_bleu(identical, max_eval_samples=12)
    sbb = ni.compute_self_bleu(disjoint, max_eval_samples=12)
    assert sba["self_bleu"] > 95.0
    assert sbb["self_bleu"] < 5.0

    pe = ni.compute_position_wise_entropy(disjoint, max_length=32)
    assert pe["quartiles"]["Q1"] == pytest.approx(math.log(12.0))
    assert pe["q4_minus_q1"] == pytest.approx(0.0, abs=1e-9)

