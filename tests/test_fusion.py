import pytest

from mimirag.errors import MimiRAGError
from mimirag.evaluation import convex, rrf


def test_rrf_shared_disjoint_and_ties():
    assert rrf([['b', 'a'], ['a', 'c']], constant=1) == [('a', 1 / 3 + 1 / 2), ('b', 1 / 2), ('c', 1 / 3)]
    assert rrf([['b'], ['a']], constant=60)[0][0] == 'a'
    assert rrf([[], []], constant=60) == []


def test_convex_normalizes_each_channel_and_keeps_union():
    left = [('a', 4.0), ('b', 2.0)]
    right = [('b', 0.8), ('c', 0.4)]
    assert convex(left, right, weight=0.5) == [('a', 0.5), ('b', 0.5), ('c', 0.0)]
    assert convex(left, right, weight=0)[0] == ('a', 1.0)
    assert convex(left, right, weight=1)[0] == ('b', 1.0)
    assert convex([('b', 9), ('a', 9)], [], weight=0.5) == [('a', 0.5), ('b', 0.5)]
    assert convex([], [], weight=0.5) == []
    assert convex([('a', 1e308), ('b', -1e308)], [], weight=0) == [('a', 1.0), ('b', 0.0)]


@pytest.mark.parametrize('constant', [0, -1, float('nan'), float('inf')])
def test_invalid_rrf_constant(constant):
    with pytest.raises(MimiRAGError):
        rrf([], constant=constant)


@pytest.mark.parametrize('weight', [-0.1, 1.1, float('nan')])
def test_invalid_weight(weight):
    with pytest.raises(MimiRAGError):
        convex([], [], weight=weight)


def test_invalid_candidates():
    with pytest.raises(MimiRAGError, match='Duplicate'):
        rrf([['a', 'a']], constant=60)
    with pytest.raises(MimiRAGError, match='Duplicate'):
        convex([('a', 1), ('a', 2)], [], weight=0.5)
    with pytest.raises(MimiRAGError, match='finite'):
        convex([], [('a', float('inf'))], weight=0.5)
