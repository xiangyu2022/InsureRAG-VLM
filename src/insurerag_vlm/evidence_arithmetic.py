"""Bounded arithmetic over quoted source values; no eval and no semantic oracle.

Passing this check proves provenance of numeric inputs and arithmetic only. It
does not prove that the model selected the right year, segment or denominator.
"""
import re
from decimal import Decimal, InvalidOperation, localcontext

OPERATIONS = ('identity', 'sum', 'mean', 'difference', 'ratio', 'percent_ratio', 'percent_change')


def decimal_value(value):
    if not isinstance(value, str) or not re.fullmatch(r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?', value.strip()):
        raise ValueError('Expected a plain decimal string')
    try: number = Decimal(value.replace(',', '').strip())
    except InvalidOperation as exc: raise ValueError('Invalid numeric input') from exc
    if not number.is_finite() or abs(number) > Decimal('1e18'):
        raise ValueError('Numeric input exceeds the diagnostic bound')
    return number


def normalized(text):
    return ' '.join(str(text).casefold().split())


def verify_calculation(payload, sources, report_scope):
    if not isinstance(payload, dict) or not isinstance(payload.get('abstain'), bool):
        raise ValueError('Expected a structured abstention decision')
    if payload['abstain']:
        return {'status': 'model_abstained', 'semantics_verified': False}
    operation = payload.get('operation')
    operands = payload.get('operands')
    if operation not in OPERATIONS or not isinstance(operands, list) or not 1 <= len(operands) <= 8:
        raise ValueError('Invalid operation or operand count')
    required_count = {'identity': 1, 'difference': 2, 'ratio': 2, 'percent_ratio': 2, 'percent_change': 2}
    if operation in required_count and len(operands) != required_count[operation]:
        raise ValueError('Wrong operand count for the selected operation')
    values = []
    citations = []
    for operand in operands:
        if not isinstance(operand, dict) or operand.get('source') not in sources:
            raise ValueError('Unknown operand citation')
        source = sources[operand['source']]
        if source.get('source_group') != report_scope:
            raise ValueError('Operand report scope differs from requested report')
        quote = normalized(operand.get('quote', ''))
        if len(quote) < 12 or quote not in normalized(source['text']):
            raise ValueError('Operand quotation is not in the supplied evidence')
        value = decimal_value(operand.get('value'))
        printed = {decimal_value(token.rstrip(',')) for token in re.findall(r'(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?![\w.])', quote)}
        if value not in printed:
            raise ValueError('Operand value is not printed in its quotation')
        values.append(value)
        citations.append({'source': operand['source'], 'answer_id': source['answer_id'],
                          'source_page': source.get('source_page'), 'quote': operand['quote'], 'value': str(value)})
    with localcontext() as ctx:
        ctx.prec = 32
        if operation == 'identity': result = values[0]
        elif operation == 'sum': result = sum(values)
        elif operation == 'mean': result = sum(values) / len(values)
        elif operation == 'difference': result = values[0] - values[1]
        else:
            if values[1] == 0: raise ValueError('Zero denominator')
            if operation == 'ratio': result = values[0] / values[1]
            elif operation == 'percent_ratio': result = values[0] / values[1] * 100
            else: result = (values[0] - values[1]) / values[1] * 100
    proposed = decimal_value(payload.get('proposed_result'))
    return {'status': 'inputs_and_arithmetic_verified', 'semantics_verified': False,
            'operation': operation, 'operands': [str(v) for v in values], 'computed_result': str(result),
            'proposed_result': str(proposed), 'proposed_within_0_02': abs(proposed-result) <= Decimal('.02'),
            'unit': 'percent' if operation.startswith('percent_') else 'source_units', 'citations': citations}


CALCULATION_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'abstain': {'type': 'boolean'},
        'reason': {'type': 'string'},
        'operation': {'type': 'string', 'enum': list(OPERATIONS)},
        'operands': {'type': 'array', 'maxItems': 8, 'items': {
            'type': 'object', 'additionalProperties': False,
            'properties': {'value': {'type': 'string', 'pattern': r'^-?[0-9]+(\.[0-9]+)?$'},
                           'source': {'type': 'string'}, 'quote': {'type': 'string'}},
            'required': ['value', 'source', 'quote']}},
        'proposed_result': {'type': 'string', 'pattern': r'^-?[0-9]+(\.[0-9]+)?$'},
    },
    'required': ['abstain', 'reason', 'operation', 'operands', 'proposed_result'],
}
