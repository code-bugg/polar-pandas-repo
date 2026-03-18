"""
tests/test_parser_B.py
======================
Unit tests for Member B's parser subtask:
  - I/O and inspection statements: LOAD, EXPORT, PREVIEW, INFO, DESCRIBE
  - Configuration statement: SET ENGINE
  - Shared helpers: parse_col_list(), parse_value_list(), parse_value()
"""

import pytest

from lexer.lexer import tokenize
from parser.parser import (
    ParseError,
    ExprLiteral,
    AstLoad, AstExport, AstPreview, AstInfo, AstDescribe, AstSetEngine,
    _Parser, parse,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parser(source: str) -> _Parser:
    return _Parser(tokenize(source))


# ─────────────────────────────────────────────────────────────────────────────
# Statement parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestIOAndInspectionStatements:

    def test_parse_load(self):
        ast = parse(tokenize('LOAD "data.csv" AS df'))
        assert ast == [AstLoad(file="data.csv", name="df", line=1)]

    def test_parse_export(self):
        ast = parse(tokenize('EXPORT df TO "out.csv"'))
        assert ast == [AstExport(name="df", file="out.csv", line=1)]

    def test_parse_preview_default_rows(self):
        ast = parse(tokenize("PREVIEW df"))
        assert ast == [AstPreview(name="df", rows=5, line=1)]

    def test_parse_preview_with_rows(self):
        ast = parse(tokenize("PREVIEW df ROWS 10"))
        assert ast == [AstPreview(name="df", rows=10, line=1)]

    def test_parse_info(self):
        ast = parse(tokenize("INFO df"))
        assert ast == [AstInfo(name="df", line=1)]

    def test_parse_describe(self):
        ast = parse(tokenize("DESCRIBE df"))
        assert ast == [AstDescribe(name="df", line=1)]

    def test_parse_set_engine(self):
        ast = parse(tokenize("SET ENGINE POLARS"))
        assert ast == [AstSetEngine(engine="polars", line=1)]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestSharedHelpers:

    def test_parse_col_list(self):
        p = _parser('[a, "b", c]')
        cols = p.parse_col_list()
        assert cols == ["a", "b", "c"]
        assert p.at_end()

    def test_parse_value(self):
        p = _parser('"x"')
        lit = p.parse_value()
        assert lit == ExprLiteral(value='"x"', kind="string", line=1)

    def test_parse_value_list(self):
        p = _parser('[1, 2.5, "x", true]')
        values = p.parse_value_list()
        assert values == [
            ExprLiteral(value="1", kind="integer", line=1),
            ExprLiteral(value="2.5", kind="float", line=1),
            ExprLiteral(value='"x"', kind="string", line=1),
            ExprLiteral(value="TRUE", kind="bool", line=1),
        ]
        assert p.at_end()

    def test_parse_value_rejects_non_literal(self):
        p = _parser("foo")
        with pytest.raises(ParseError):
            p.parse_value()
