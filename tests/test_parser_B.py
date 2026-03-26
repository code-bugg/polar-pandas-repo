"""
tests/test_parser_B.py
======================
Unit tests for Member B's parser subtask
"""

import pytest
from lexer.lexer import tokenize
from parser.parser import (
    ParseError, ExprLiteral,
    AstLoad, AstExport, AstPreview, AstInfo, AstDescribe, AstSetEngine,
    _Parser, parse,
)

def _parser(source: str) -> _Parser:
    return _Parser(tokenize(source))

class TestIOAndInspectionStatements:
    def test_parse_load(self):
        ast = parse(tokenize('LOAD "data.csv" AS df'))
        assert ast == [AstLoad(file="data.csv", name="df", engine=None, line=1)]

    def test_parse_export(self):
        ast = parse(tokenize('EXPORT df TO "out.csv"'))
        assert ast == [AstExport(name="df", file="out.csv", format=None, line=1)]

    def test_parse_preview_default_rows(self):
        ast = parse(tokenize("PREVIEW df"))
        assert ast == [AstPreview(name="df", rows=5, line=1)]

    def test_parse_set_engine(self):
        ast = parse(tokenize("SET ENGINE POLARS"))
        assert ast == [AstSetEngine(engine="polars", line=1)]

class TestSharedHelpers:
    def test_parse_col_list(self):
        p = _parser('[a, "b", c]')
        cols = p.parse_col_list()
        assert cols == ["a", "b", "c"]
        assert p.at_end()

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