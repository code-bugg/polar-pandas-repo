"""
tests/test_lexer_C.py
=====================
Unit tests for Member C's lexer subtask:
  - _consume_string()
  - _consume_number()
  - _consume_loopvar()
"""

import pytest
from lexer.lexer import LexError, Token, TokenType, _Lexer

class TestConsumeString:
    def test_simple_string(self):
        lex = _Lexer('"hello"')
        lex._consume_string()
        assert lex._tokens == [Token(TokenType.STRING, '"hello"', 1)]
        assert lex.at_end()

    def test_supported_escapes_are_resolved(self):
        lex = _Lexer('"a\\nb\\tc\\\\d\\\"e"')
        lex._consume_string()
        assert lex._tokens == [Token(TokenType.STRING, '"a\nb\tc\\d\"e"', 1)]

    def test_unterminated_string_at_eof_raises_lex_error(self):
        lex = _Lexer('"unterminated')
        with pytest.raises(LexError, match="Unterminated string literal"):
            lex._consume_string()

class TestConsumeNumber:
    def test_integer_token(self):
        lex = _Lexer("12345")
        lex._consume_number()
        assert lex._tokens == [Token(TokenType.INTEGER, "12345", 1)]

    def test_float_token(self):
        lex = _Lexer("123.45")
        lex._consume_number()
        assert lex._tokens == [Token(TokenType.FLOAT, "123.45", 1)]

    def test_integer_when_dot_not_followed_by_digit(self):
        lex = _Lexer("123.")
        lex._consume_number()
        assert lex._tokens == [Token(TokenType.INTEGER, "123", 1)]
        assert lex.peek() == "."

class TestConsumeLoopVar:
    def test_simple_loopvar(self):
        lex = _Lexer("$col")
        lex._consume_loopvar()
        assert lex._tokens == [Token(TokenType.LOOPVAR, "$col", 1)]
        assert lex.at_end()

    def test_bare_dollar_raises_lex_error(self):
        lex = _Lexer("$")
        with pytest.raises(LexError, match="must be followed by a letter or underscore"):
            lex._consume_loopvar()