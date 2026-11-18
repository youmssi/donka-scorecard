import math

import numpy as np
import pytest

from donka_scorecard.metrics import numeric_population_stability, population_stability, woe_iv_by_bin


def test_independent_woe_iv_matches_a_worked_two_bin_example():
    result = woe_iv_by_bin(np.array([0, 0, 1, 0, 1, 1]), np.array([0, 0, 0, 1, 1, 1]))
    expected = math.log(2) / 3
    assert [row["woe"] for row in result] == [pytest.approx(math.log(2)), pytest.approx(-math.log(2))]
    assert sum(row["iv"] for row in result) == pytest.approx(2 * expected)


def test_population_stability_is_zero_for_unchanged_population():
    values = np.array([1, 1, 2, 2, 3])
    assert population_stability(values, values) == pytest.approx(0)
    assert numeric_population_stability(values, values) == pytest.approx(0)


