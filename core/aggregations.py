from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np


class Aggregation(ABC):
    """Abstract class for aggregation functions."""

    def __call__(self, utilities: NDArray) -> float:
        return self.forward(utilities)

    @abstractmethod
    def forward(self, utilities: NDArray) -> float: ...


class NSW(Aggregation):
    """Nash Social Welfare"""

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.epsilon = epsilon

    def forward(self, utilities: NDArray) -> float:
        return np.log(utilities + 1 + self.epsilon).sum()


class Utilitarian(Aggregation):
    """Utilitarian Welfare"""

    def forward(self, utilities: NDArray) -> float:
        return utilities.sum()


class Rawlsian(Aggregation):
    """Rawlsian Welfare"""

    def forward(self, utilities: NDArray) -> float:
        return utilities.min()


class Egalitarian(Aggregation):
    """Egalitarian Welfare"""

    def forward(self, utilities: NDArray) -> float:
        mean_reward = sum(utilities) / len(utilities)
        diff_sum = sum(abs(x - mean_reward) for x in utilities)
        return -diff_sum


class Gini(Aggregation):
    """Gini Coefficient based Social Welfare"""

    def __init__(self, epsilon: float = 1e-7) -> None:
        self.epsilon = epsilon

    def forward(self, utilities: NDArray) -> float:
        # Calculate the Gini coefficient of a list of values. Based on: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
        if np.amin(utilities) < 0:
            utilities -= np.amin(utilities)
        utilities += self.epsilon
        sorted_values = np.sort(utilities)
        index = np.arange(1, utilities.size + 1)
        n = utilities.size
        gini = (np.sum((2 * index - n - 1) * sorted_values)) / (
            n * np.sum(sorted_values)
        )

        # Calculate the Gini reward
        return 1 - gini


class RDP(Aggregation):
    """Relaxed Demographic Parity"""

    def forward(self, utilities: NDArray) -> float:
        assert len(utilities) == 2
        return -np.abs(utilities[0] - utilities[1])
