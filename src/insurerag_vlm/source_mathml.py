"""Conservative text rendering for presentation MathML fractions.

Plain get_text loses an mfrac's division operator. Unsupported structures fail
instead of turning a formula into a plausible but different source statement.
This renderer transcribes structure; it does not calculate or infer operators.
"""
import re
from decimal import Decimal

from bs4 import Comment, NavigableString, Tag


def _name(node):
    return (node.name or '').split(':')[-1].lower()


def _children(node):
    children = []
    for child in node.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, Tag):
            children.append(child)
        elif isinstance(child, NavigableString) and str(child).strip():
            raise ValueError('MathML structural element contains untagged text')
    return children


def render_mathml(node):
    """Render the supported fraction/token subset with explicit grouping.

    Reject barless fractions: those may denote binomial coefficients, not
    division. Reject unfamiliar presentation/content nodes rather than flatten
    exponents, roots, matrices, alternatives or hidden annotation branches.
    """
    name = _name(node)
    if name in {'mi', 'mn', 'mo', 'mtext'}:
        if node.find(True) is not None:
            raise ValueError('MathML token contains nested markup')
        value = ' '.join(node.get_text().split())
        if not value:
            raise ValueError('Empty MathML token')
        return value
    if name not in {'math', 'mrow', 'mfrac'}:
        raise ValueError('Unsupported MathML element: ' + name)
    children = _children(node)
    if name == 'mfrac':
        thickness = str(node.get('linethickness', '')).strip().lower()
        if thickness and thickness not in {'thin', 'medium', 'thick'}:
            length = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:[a-z]+|%)?', thickness)
            if length is None:
                raise ValueError('Unsupported MathML fraction line thickness')
            if Decimal(length.group(1)) <= 0:
                raise ValueError('Barless MathML fraction needs explicit source review')
        if len(children) != 2:
            raise ValueError('MathML fraction needs exactly two operands')
        numerator, denominator = (render_mathml(child) for child in children)
        return f'(({numerator}) / ({denominator}))'
    if not children:
        raise ValueError('Empty MathML expression')
    return ' '.join(render_mathml(child) for child in children)


def preserve_mathml(root):
    """Replace complete math trees only after every tree renders successfully."""
    nodes = ([root] if isinstance(root, Tag) and _name(root) == 'math'
             else root.find_all(lambda node: isinstance(node, Tag) and _name(node) == 'math'))
    nodes = [node for node in nodes
             if not any(isinstance(p, Tag) and _name(p) == 'math' for p in node.parents)]
    rendered = [(node, render_mathml(node)) for node in nodes]
    for node, value in rendered:
        if node is root:
            node.clear()
            node.name = 'span'
            node.append(NavigableString(value))
        else:
            node.replace_with(NavigableString(value))
    return len(rendered)
