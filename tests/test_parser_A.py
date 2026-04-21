import pytest

from lexer.lexer import TokenType, tokenize
from parser.parser import (
    ParseError,
    ExprLiteral,
    ExprColRef,
    ExprBinop,
    CondCompare,
    CondAnd,
    CondOr,
    AstRead,
    AstSave,
    AstDropEmpty,
    AstIf,
    _Parser,
    parse,
)


def _parser(source: str) -> _Parser:
    return _Parser(tokenize(source))


class TestCoreNodes:
    def test_expr_nodes_are_frozen(self):
        n = ExprLiteral("1", "integer", 1)
        with pytest.raises((AttributeError, TypeError)):
            n.value = "2"  # type: ignore

    def test_expr_colref_loopvar(self):
        n = ExprColRef("col", True, 2)
        assert n.is_loopvar is True

    def test_expr_binop(self):
        n = ExprBinop("+", ExprLiteral("1", "integer", 1), ExprLiteral("2", "integer", 1), 1)
        assert n.op == "+"

    def test_condition_nodes(self):
        c1 = CondCompare(ExprColRef("a", False, 1), ">", ExprLiteral("0", "integer", 1), 1)
        c2 = CondCompare(ExprColRef("b", False, 1), "==", ExprLiteral('"x"', "string", 1), 1)
        assert isinstance(CondAnd(c1, c2, 1), CondAnd)
        assert isinstance(CondOr(c1, c2, 1), CondOr)


class TestParseError:
    def test_stores_message_and_line(self):
        err = ParseError("bad", 7)
        assert err.message == "bad"
        assert err.line == 7


class TestParserCursor:
    def test_peek_consume_at_end(self):
        p = _parser("READ 'x.csv'")
        assert p.peek().value == "READ"
        p.consume()
        assert not p.at_end()

    def test_expect_helpers(self):
        p = _parser("READ 'x.csv'")
        assert p.expect_kw("READ").value == "READ"
        assert p.expect_string().value == "'x.csv'"

    def test_expect_int(self):
        p = _parser("10")
        assert p.expect_int().value == "10"

    def test_expect_int_rejects_float(self):
        p = _parser("1.5")
        with pytest.raises(ParseError):
            p.expect_int()

    def test_next_is_helpers(self):
        p = _parser("READ")
        assert p.next_is_kw("READ")
        assert p.next_is(TokenType.KW)


class TestDispatcher:
    def test_parse_statement_read(self):
        p = _parser("READ 'x.csv'")
        node = p.parse_statement()
        assert isinstance(node, AstRead)

    def test_parse_statement_operation(self):
        p = _parser("SAVE 'out.csv'")
        node = p.parse_statement()
        assert isinstance(node, AstSave)

    def test_parse_statement_if(self):
        p = _parser("IF EXISTS sales : INFO END")
        node = p.parse_statement()
        assert isinstance(node, AstIf)

    def test_non_statement_token_raises(self):
        p = _parser("'x'")
        with pytest.raises(ParseError):
            p.parse_statement()


class TestParseAll:
    def test_empty_returns_empty(self):
        assert parse(tokenize("")) == []

    def test_multiple_statements(self):
        nodes = parse(tokenize("READ 'x.csv' SAVE 'y.csv' DROP EMPTY"))
        assert isinstance(nodes[0], AstRead)
        assert isinstance(nodes[1], AstSave)
        assert isinstance(nodes[2], AstDropEmpty)
