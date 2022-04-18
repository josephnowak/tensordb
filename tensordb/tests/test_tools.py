import pytest

from typing import List, Dict
from loguru import logger

from tensordb.utils.tools import groupby_chunks


class TestTools:

    def test_groupby_chunks(self):
        e = {'a': 0, 'b': 1, 'c': 0, 'd': 0, 'e': 0, 'm': 1, 'g': 2, 'l': 2}
        result = list(groupby_chunks(list(e), {0: 2, 1: 1}, lambda x: e[x], lambda x: (e[x], x)))
        assert result == [['a', 'c', 'b', 'g', 'l'], ['d', 'e', 'm']]

        result = list(groupby_chunks(list(e), {0: 2, 1: 2, 2: 1}, lambda x: e[x], lambda x: (e[x], x)))
        assert result == [['a', 'c', 'b', 'm', 'g'], ['d', 'e', 'l']]

        e['f'] = 1
        result = list(groupby_chunks(list(e), {0: 2, 1: 1, 2: 2}, lambda x: e[x], lambda x: (e[x], x)))
        assert result == [['a', 'c', 'b', 'g', 'l'], ['d', 'e', 'f'], ['m']]

        result = list(groupby_chunks(list(e), {0: 1, 1: 2, 2: 1}, lambda x: e[x], lambda x: (e[x], x)))
        assert result == [['a', 'b', 'f', 'g'], ['c', 'm', 'l'], ['d'], ['e']]

        result = list(groupby_chunks(list(e), {0: 3, 1: 2, 2: 1}, lambda x: e[x], lambda x: (e[x], x)))
        assert result == [['a', 'c', 'd', 'b', 'f', 'g'], ['e', 'm', 'l']]


if __name__ == "__main__":
    test = TestTools()
    test.test_groupby_chunks()
