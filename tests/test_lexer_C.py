"""
tests/test_lexer_C.py
=====================
Unit tests for Member C's lexer subtask:
  - _consume_string(): double-quoted strings with supported escapes
  - _consume_number(): INTEGER vs FLOAT with maximal munch
  - _consume_loopvar(): $identifier handling and validation
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

    def test_invalid_escape_raises_lex_error(self):
        lex = _Lexer('"bad\\q"')

        with pytest.raises(LexError, match="Invalid escape sequence"):
            lex._consume_string()

    def test_unterminated_string_at_eof_raises_lex_error(self):
        lex = _Lexer('"unterminated')

        with pytest.raises(LexError, match="Unterminated string literal"):
            lex._consume_string()

    def test_newline_before_closing_quote_raises_lex_error(self):
        lex = _Lexer('"unterminated\nnext')

        with pytest.raises(LexError, match="Unterminated string literal"):
            lex._consume_string()

    def test_string_line_number_uses_opening_quote_line(self):
        lex = _Lexer('\n"x"')
        lex.advance()
        lex._consume_string()

        assert lex._tokens == [Token(TokenType.STRING, '"x"', 2)]


class TestConsumeNumber:

    def test_integer_token(self):
        lex = _Lexer("12345")
        lex._consume_number()

        assert lex._tokens == [Token(TokenType.INTEGER, "12345", 1)]
        assert lex.at_end()

    def test_float_token(self):
        lex = _Lexer("123.45")
        lex._consume_number()

        assert lex._tokens == [Token(TokenType.FLOAT, "123.45", 1)]
        assert lex.at_end()

    def test_integer_when_dot_not_followed_by_digit(self):
        lex = _Lexer("123.")
        lex._consume_number()

        assert lex._tokens == [Token(TokenType.INTEGER, "123", 1)]
        assert lex.peek() == "."

    def test_integer_stops_before_non_digit_suffix(self):
        lex = _Lexer("123abc")
        lex._consume_number()

        assert lex._tokens == [Token(TokenType.INTEGER, "123", 1)]
        assert lex.peek() == "a"

    def test_float_consumes_longest_fractional_sequence(self):
        lex = _Lexer("12.345xyz")
        lex._consume_number()

        assert lex._tokens == [Token(TokenType.FLOAT, "12.345", 1)]
        assert lex.peek() == "x"

    def test_leading_zeroes_are_preserved(self):
        lex = _Lexer("0012")
        lex._consume_number()

        assert lex._tokens == [Token(TokenType.INTEGER, "0012", 1)]


class TestConsumeLoopVar:

    def test_simple_loopvar(self):
        lex = _Lexer("$col")
        lex._consume_loopvar()

        assert lex._tokens == [Token(TokenType.LOOPVAR, "$col", 1)]
        assert lex.at_end()

    def test_loopvar_allows_underscore_and_digits_after_start(self):
        lex = _Lexer("$_col2")
        lex._consume_loopvar()

        assert lex._tokens == [Token(TokenType.LOOPVAR, "$_col2", 1)]

    def test_loopvar_stops_before_non_identifier_character(self):
        lex = _Lexer("$item-name")
        lex._consume_loopvar()

        assert lex._tokens == [Token(TokenType.LOOPVAR, "$item", 1)]
        assert lex.peek() == "-"

    def test_bare_dollar_raises_lex_error(self):
        lex = _Lexer("$")

        with pytest.raises(LexError, match="must be followed by a letter or underscore"):
            lex._consume_loopvar()

    def test_dollar_followed_by_digit_raises_lex_error(self):
        lex = _Lexer("$1abc")

        with pytest.raises(LexError, match="must be followed by a letter or underscore"):
            lex._consume_loopvar()

    def test_loopvar_line_number_uses_dollar_line(self):
        lex = _Lexer("\n$item")
        lex.advance()
        lex._consume_loopvar()

        assert lex._tokens == [Token(TokenType.LOOPVAR, "$item", 2)]