import pytest

from lexer.lexer import tokenize
from parser.parser import (
    ParseError,
    ExprLiteral,
    ExprColRef,
    ExprBinop,
    CondCompare,
    CondIn,
    CondNull,
    CondBetween,
    CondContains,
    CondAnd,
    CondOr,
    AstFilter,
    _Parser,
    parse,
)


def _parser(source: str) -> _Parser:
    return _Parser(tokenize(source))


class TestExpressions:
    def test_literal_number(self):
        p = _parser("42")
        expr = p.parse_expr()
        assert expr == ExprLiteral("42", "integer", 1)

    def test_colref(self):
        p = _parser("amount")
        expr = p.parse_expr()
        assert expr == ExprColRef("amount", False, 1)

    def test_loopvar_ref(self):
        p = _parser("$col")
        expr = p.parse_expr()
        assert expr == ExprColRef("col", True, 1)

    def test_precedence(self):
        p = _parser("a + b * c")
        expr = p.parse_expr()
        assert isinstance(expr, ExprBinop)
        assert expr.op == "+"
        assert isinstance(expr.right, ExprBinop)
        assert expr.right.op == "*"

    def test_parentheses(self):
        p = _parser("(a + b) * c")
        expr = p.parse_expr()
        assert isinstance(expr, ExprBinop)
        assert expr.op == "*"


class TestConditions:
    def test_compare(self):
        node = parse(tokenize("FILTER WHERE amount >= 100"))[0]
        assert isinstance(node, AstFilter)
        assert isinstance(node.condition, CondCompare)

    def test_in(self):
        node = parse(tokenize("FILTER WHERE country IN 'MD', 'RO'"))[0]
        assert isinstance(node.condition, CondIn)
        assert node.condition.negate is False

    def test_not_in(self):
        node = parse(tokenize("FILTER WHERE country NOT IN 'XX', 'ZZ'"))[0]
        assert isinstance(node.condition, CondIn)
        assert node.condition.negate is True

    def test_is_empty(self):
        node = parse(tokenize("FILTER WHERE email IS EMPTY"))[0]
        assert isinstance(node.condition, CondNull)
        assert node.condition.is_not is False

    def test_is_not_empty(self):
        node = parse(tokenize("FILTER WHERE email IS NOT EMPTY"))[0]
        assert isinstance(node.condition, CondNull)
        assert node.condition.is_not is True

    def test_between(self):
        node = parse(tokenize("FILTER WHERE score BETWEEN 0 AND 100"))[0]
        assert isinstance(node.condition, CondBetween)

    def test_contains(self):
        node = parse(tokenize("FILTER WHERE email CONTAINS '@gmail'"))[0]
        assert isinstance(node.condition, CondContains)
        assert node.condition.negate is False

    def test_not_contains(self):
        node = parse(tokenize("FILTER WHERE notes NOT CONTAINS 'deleted'"))[0]
        assert isinstance(node.condition, CondContains)
        assert node.condition.negate is True

    def test_and_or_precedence(self):
        node = parse(tokenize("FILTER WHERE a == 1 OR b == 2 AND c == 3"))[0]
        assert isinstance(node.condition, CondOr)
        assert isinstance(node.condition.right, CondAnd)

    def test_condition_parentheses(self):
        node = parse(tokenize("FILTER WHERE (a == 1 OR b == 2) AND c == 3"))[0]
        assert isinstance(node.condition, CondAnd)


class TestFilterErrors:
    def test_missing_where(self):
        with pytest.raises(ParseError):
            parse(tokenize("FILTER amount > 0"))

    def test_bad_not_clause(self):
        with pytest.raises(ParseError):
            parse(tokenize("FILTER WHERE x NOT 1"))
