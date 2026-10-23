import pytest
from bs4 import BeautifulSoup

from src.insurerag_vlm.source_tables import EMPTY_CELL, render_reviewed_table


def table(html):
    return BeautifulSoup(html, 'html.parser').table


def test_grouped_headers_and_blank_cells_keep_their_original_columns():
    node = table('''<table><thead><tr><th rowspan="2">Provider</th>
    <th colspan="2">Options</th></tr><tr><th>Red</th><th>Blue</th></tr></thead>
    <tbody><tr><td>Alpha</td><td></td><td>X</td></tr>
    <tr><td>Beta</td><td>X</td><td>&nbsp;</td></tr></tbody></table>''')
    before = str(node)
    result = render_reviewed_table(node)
    assert result['header_line'] == 'Provider | Options / Red | Options / Blue'
    assert result['body_lines'] == [f'Alpha | {EMPTY_CELL} | X', f'Beta | X | {EMPTY_CELL}']
    assert [cell['column'] for cell in result['body_cells'] if cell['empty']] == [1, 2]
    assert result['column_paths'] == [['Provider'], ['Options', 'Red'], ['Options', 'Blue']]
    assert str(node) == before


@pytest.mark.parametrize('html', [
    '<table><thead><tr><th>A</th></tr></thead>'
    '<tbody><tr><td><img alt="covered" src="icon.svg"></td></tr></tbody></table>',
    '<table><thead><tr><th rowspan="3">A</th><th>B</th></tr></thead>'
    '<tbody><tr><td>x</td><td>y</td></tr></tbody></table>',
    '<table><thead><tr><th>A</th><th>B</th></tr><tr><th>C</th></tr></thead>'
    '<tbody><tr><td>x</td><td>y</td></tr></tbody></table>',
    '<table><thead><tr><th>A</th><th>B</th></tr></thead>'
    '<tbody><tr><td>x</td></tr></tbody></table>',
    '<table><thead><tr><th>A</th><th>B</th></tr></thead>'
    '<tbody><tr><td colspan="2">x</td><td>y</td></tr></tbody></table>',
    '<table><thead><tr><th>A</th><th>A</th></tr></thead>'
    '<tbody><tr><td>x</td><td>y</td></tr></tbody></table>',
    '<table><thead><tr><th>A</th></tr></thead>'
    '<tbody><tr><td><table><tr><td>x</td></tr></table></td></tr></tbody></table>',
    '<table><thead><tr><th colspan="2">A</th><th rowspan="2">B</th></tr>'
    '<tr><th colspan="3">C</th></tr></thead>'
    '<tbody><tr><td>x</td><td>y</td><td>z</td></tr></tbody></table>',
    '<table>Important qualification<thead><tr><th>A</th></tr></thead>'
    '<tbody><tr><td>x</td></tr></tbody></table>',
])
def test_ambiguous_or_incomplete_structure_cannot_be_silently_flattened(html):
    node = table(html)
    before = str(node)
    with pytest.raises(ValueError):
        render_reviewed_table(node)
    assert str(node) == before
