import pytest
from bs4 import BeautifulSoup

from src.insurerag_vlm.source_revisions import transcribe_reviewed_deletions


def root(html):
    return BeautifulSoup(html, 'html.parser').div


def test_original_deletions_and_adjacent_replacement_are_preserved_without_mutation():
    node = root('<div>Historical revision: within <span style="color:red">'
                '<span style="text-decoration: line-through;">14</span>28</span>'
                ' days. <del>Old service</del> New service.</div>')
    before = str(node)
    result = transcribe_reviewed_deletions(node, ['span[style*="line-through"]', 'del'])
    assert result['text'] == 'Historical revision: within [deleted in source: 14] 28 days. [deleted in source: Old service] New service.'
    assert [n['text'] for n in result['deletions']] == ['14', 'Old service']
    assert str(node) == before


@pytest.mark.parametrize('html, selectors', [
    ('<div><span>14</span>28</div>', ['span']),
    ('<div><del>14</del>28</div>', []),
    ('<div><del>14</del>28</div>', ['.missing']),
    ('<div><del>14</del><s>Old</s>28</div>', ['del']),
    ('<div><del><strong>14</strong></del>28</div>', ['del']),
    ('<div><del></del>28</div>', ['del']),
    ('<div><del>14]</del>28</div>', ['del']),
    ('<div>[deleted in source: 14]<del>Old</del></div>', ['del']),
    ('<div><p style="text-decoration:line-through">14</p><del>Old</del></div>', ['del']),
    ('<div><span style="text-decoration:line-through;text-decoration:none">14</span></div>', ['span']),
])
def test_unsupported_or_incomplete_revision_selection_fails_without_mutation(html, selectors):
    node = root(html)
    before = str(node)
    with pytest.raises(ValueError):
        transcribe_reviewed_deletions(node, selectors)
    assert str(node) == before
