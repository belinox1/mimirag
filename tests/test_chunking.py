from mimirag.chunking import units
from mimirag.datatypes import Unit


def test_units_merge_whole_blocks_up_to_the_target():
    table = '\n'.join(f'| row {i} | value |' for i in range(6))
    text = 'Intro\n\n' + table + '\n\nEnd'
    assert list(units(text, target=40)) == [Unit(1, 1, 'Intro'), Unit(3, 8, table), Unit(10, 10, 'End')]
    assert list(units('a\n\nb', target=40)) == [Unit(1, 3, 'a\n\nb')]
    assert list(units('')) == []


def test_a_long_block_is_one_oversized_unit():
    long = 'word ' * 50
    assert list(units(f'short\n\n{long}\n\ntail', target=40)) == [
        Unit(1, 1, 'short'),
        Unit(3, 3, long),
        Unit(5, 5, 'tail'),
    ]


def test_repeated_blank_lines_stay_with_the_next_block():
    assert list(units('a\n\n\nb', target=1)) == [Unit(1, 1, 'a'), Unit(3, 4, '\nb')]
