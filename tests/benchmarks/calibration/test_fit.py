# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import random

import pytest

from sonar.benchmarks.calibration.fit import FitError, PrefillSample, fit_prefill_cost

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)


def _samples(c: float, d: float, e: float, *, noise: float = 0.0, repeats: int = 1, seed: int = 0):
    rng = random.Random(seed)
    samples = []
    for length in LENGTHS:
        for _ in range(repeats):
            seconds = c + d * length + e * length * length / 2e5
            samples.append(PrefillSample(length, seconds * (1 + rng.gauss(0, noise))))
    return samples


def test_recovers_known_coefficients():
    fit = fit_prefill_cost(_samples(0.05, 1.862e-5, 3.33e-6), bootstrap=0)
    assert fit.intercept == pytest.approx(0.05, rel=1e-6)
    assert fit.per_token == pytest.approx(1.862e-5, rel=1e-6)
    assert fit.per_token_per_100k == pytest.approx(3.33e-6, rel=1e-6)
    assert fit.ratio == pytest.approx(3.33e-6 / 1.862e-5, rel=1e-6)
    assert fit.max_relative_error < 1e-9
    assert fit.predict(100000) == pytest.approx(0.05 + 1.862 + 3.33e-6 * 1e10 / 2e5, rel=1e-6)
    assert fit.coefficients(0.5) == pytest.approx((1.862e-5, 0.931e-5))


def test_sublinear_timings_clamp_the_quadratic_term_to_zero():
    # Prefill that gets cheaper per token at long context has no valid E > 0.
    samples = [PrefillSample(length, 1e-4 * length**0.9) for length in LENGTHS]
    fit = fit_prefill_cost(samples, bootstrap=0)
    assert fit.per_token_per_100k == 0.0
    assert fit.ratio == 0.0
    assert fit.per_token > 0


def test_negative_overhead_is_clamped_to_zero():
    samples = [PrefillSample(length, 2e-5 * length + 1e-6 * length**2 / 2e5 - 0.01) for length in LENGTHS]
    fit = fit_prefill_cost(samples, bootstrap=0)
    assert fit.intercept == 0.0
    assert fit.per_token > 0 and fit.per_token_per_100k > 0


def test_one_stalled_request_does_not_bend_the_fit():
    # GLM-5.3-Flash on two B300 GPUs; one 31k-token request stalled at 3.4 s.
    measured = {
        7897: [0.1627, 0.1508, 0.1511],
        15678: [0.3012, 0.2742, 0.2744],
        31126: [3.4, 0.5726, 0.5733],
        61796: [1.1291, 1.1220, 1.1254],
        122688: [2.2648, 2.2566, 2.2637],
    }
    samples = [PrefillSample(length, seconds) for length, values in measured.items() for seconds in values]
    fit = fit_prefill_cost(samples, bootstrap=0)
    assert fit.per_token == pytest.approx(1.73e-5, rel=0.05)
    assert fit.ratio < 0.25
    assert fit.max_relative_error < 0.05


def test_requires_three_distinct_lengths():
    samples = [PrefillSample(1024, 0.1), PrefillSample(1024, 0.11), PrefillSample(2048, 0.2)]
    with pytest.raises(FitError, match="3 or more distinct prompt lengths"):
        fit_prefill_cost(samples)


def test_rejects_timings_that_do_not_grow_with_length():
    samples = [PrefillSample(length, 1.0) for length in LENGTHS]
    with pytest.raises(FitError, match="does not increase with prompt length"):
        fit_prefill_cost(samples)


def test_rejects_non_positive_timings():
    samples = [PrefillSample(length, 0.1) for length in LENGTHS[:3]] + [PrefillSample(4096, 0.0)]
    with pytest.raises(FitError, match="Invalid prefill timing"):
        fit_prefill_cost(samples)


def test_bootstrap_interval_is_deterministic_and_covers_the_true_ratio():
    samples = _samples(0.02, 2e-5, 4e-6, noise=0.03, repeats=3, seed=7)
    first = fit_prefill_cost(samples, bootstrap=200, seed=3)
    second = fit_prefill_cost(samples, bootstrap=200, seed=3)
    assert first.ratio_interval == second.ratio_interval
    low, high = first.ratio_interval
    assert low <= 0.2 <= high
    assert math.isfinite(high)
