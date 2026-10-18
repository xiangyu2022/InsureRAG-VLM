"""Literal transcription of explicitly reviewed simple HTML data tables.

Only header spans are supported. Empty source cells remain explicit metadata;
they are never translated into a negative coverage or eligibility statement.
Callers must review provenance, licenses, contexts and canonical replacements.
"""
import re

from bs4 import Comment, NavigableString, Tag


EMPTY_CELL = '[empty source cell]'


def _structure(parent, allowed):
    for child in parent.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, Tag):
            if child.name not in allowed:
                raise ValueError('Unsupported table structural child')
        elif isinstance(child, NavigableString) and str(child).strip():
            raise ValueError('Unbound text outside table cells')


def _span(cell, name):
    value = str(cell.get(name, '1'))
    if not re.fullmatch(r'[1-9][0-9]*', value) or int(value) > 64:
        raise ValueError('Unsupported table span')
    return int(value)


def _text(cell):
    if cell.find(['table', 'script', 'style', 'math', 'img', 'svg', 'canvas', 'input']) is not None:
        raise ValueError('Nested table or unsupported cell content')
    value = ' '.join(cell.get_text(' ', strip=True).split())
    if '|' in value or EMPTY_CELL in value:
        raise ValueError('Ambiguous table transcription delimiter')
    return value


def render_reviewed_table(table):
    """Return text plus original cell coordinates without changing the DOM.

    Require thead/tbody, a complete rectangular header grid, distinct column
    paths, and unmerged body cells. Reject incomplete or unsupported structure
    rather than guessing column labels or flattening a merged body cell.
    """
    if table.name != 'table' or table.find('table') is not None:
        raise ValueError('One non-nested table required')
    if table.find('tfoot') is not None:
        raise ValueError('Table footer needs separate source review')
    _structure(table, {'caption', 'thead', 'tbody'})
    heads = table.find_all('thead', recursive=False)
    bodies = table.find_all('tbody', recursive=False)
    if len(heads) != 1 or len(bodies) != 1:
        raise ValueError('Exactly one thead and tbody required')
    _structure(heads[0], {'tr'})
    _structure(bodies[0], {'tr'})
    header_rows = heads[0].find_all('tr', recursive=False)
    body_rows = bodies[0].find_all('tr', recursive=False)
    if not header_rows or len(header_rows) > 16 or not body_rows:
        raise ValueError('Nonempty bounded header and body required')
    first = header_rows[0].find_all(['th', 'td'], recursive=False)
    width = sum(_span(cell, 'colspan') for cell in first)
    if not 1 <= width <= 64:
        raise ValueError('Unsupported table width')
    grid = [[None] * width for _ in header_rows]
    header_cells = []
    for row_index, row in enumerate(header_rows):
        _structure(row, {'th', 'td'})
        column = 0
        for cell in row.find_all(['th', 'td'], recursive=False):
            while column < width and grid[row_index][column] is not None:
                column += 1
            rows, columns = _span(cell, 'rowspan'), _span(cell, 'colspan')
            if row_index + rows > len(grid) or column + columns > width:
                raise ValueError('Header span exceeds table grid')
            positions = [(r, c) for r in range(row_index, row_index + rows)
                         for c in range(column, column + columns)]
            if any(grid[r][c] is not None for r, c in positions):
                raise ValueError('Overlapping header spans')
            value = _text(cell)
            index = len(header_cells)
            header_cells.append(dict(row=row_index, column=column, rowspan=rows,
                                     colspan=columns, text=value))
            for r, c in positions:
                grid[r][c] = index
            column += columns
        if any(index is None for index in grid[row_index]):
            raise ValueError('Incomplete header row')
    paths = []
    for column in range(width):
        indices = list(dict.fromkeys(row[column] for row in grid))
        path = [header_cells[index]['text'] for index in indices
                if header_cells[index]['text']]
        if not path:
            raise ValueError('Column lacks an explicit header')
        paths.append(path)
    labels = [' / '.join(path) for path in paths]
    if len(set(labels)) != width:
        raise ValueError('Ambiguous repeated column labels')
    values, body_cells = [], []
    for row_index, row in enumerate(body_rows):
        _structure(row, {'th', 'td'})
        cells = row.find_all(['th', 'td'], recursive=False)
        if len(cells) != width:
            raise ValueError('Body row width differs from header')
        row_values = []
        for column, cell in enumerate(cells):
            if _span(cell, 'rowspan') != 1 or _span(cell, 'colspan') != 1:
                raise ValueError('Merged body cells need separate source review')
            value = _text(cell)
            row_values.append(value)
            body_cells.append(dict(row=row_index, column=column, text=value,
                                   empty=(value == '')))
        values.append(row_values)
    captions = table.find_all('caption', recursive=False)
    if len(captions) > 1:
        raise ValueError('Multiple table captions')
    caption = _text(captions[0]) if captions else ''
    header_line = ' | '.join(labels)
    body_lines = [' | '.join(value if value else EMPTY_CELL for value in row)
                  for row in values]
    return dict(text='\n'.join(([caption] if caption else []) + [header_line] + body_lines),
                caption=caption, header_line=header_line, body_lines=body_lines,
                column_paths=paths, header_cells=header_cells, body_cells=body_cells,
                header_grid=grid, rows=len(values), columns=width,
                blank_cell_interpretation='Explicit source emptiness; no inferred negative.')
