import math
import random
from typing import List


class Partitioner:
    """Class to randomly partition `n` integers into sets of size `m`
    """

    def __init__(self, n: int, m: int):
        """Constructor method

        Parameters
        ----------
        n : int
            Number of integers
        m : int
            Size of each set
        """
        if n <= 0 or m <= 0:
            raise ValueError(f"n ({n}) and m ({m}) must be positive integers")
        if m > n:
            raise ValueError(f"m ({m}) cannot be greater than n ({n})")
        self._n = n
        self._m = m

    def __call__(self) -> List[List[int]]:
        """Shuffles all integers from 0 to `n` - 1 and creates a partition of
        this list.

        Returns
        -------
        List[List[int]]
            The partitioned list
        """
        ints = list(range(self._n))
        random.shuffle(ints)
        output_list = []
        num_sets = math.ceil(self._n / self._m)
        for i in range(num_sets - 1):
            output_list.append(ints[i * self._m:(i + 1) * self._m])
        output_list.append(ints[(num_sets - 1) * self._m:])
        return output_list
