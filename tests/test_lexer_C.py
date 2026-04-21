import pytest

from lexer.lexer import LexError, Token, TokenType, _Lexer


class TestConsumeString:
    def test_double_quoted_string(self):
        lex = _Lexer('"hello"')
        lex._consume_string()
        assert lex._tokens == [Token(TokenType.STRING, '"hello"', 1)]

    def test_single_quoted_string(self):
        lex = _Lexer("'hello'")
        lex._consume_string()
        assert lex._tokens == [Token(TokenType.STRING, "'hello'", 1)]

    def test_supported_escapes(self):
        lex = _Lexer('"a\\nb\\tc\\\\d\\\"e"')
        lex._consume_string()
        assert lex._tokens == [Token(TokenType.STRING, '"a\nb\tc\\d\"e"', 1)]

    def test_invalid_escape_raises(self):
        lex = _Lexer('"bad\\q"')
        with pytest.raises(LexError, match="Invalid escape sequence"):
            lex._consume_string()


class TestConsumeNumber:
    def test_integer_number_token(self):
        lex = _Lexer("123")
        lex._consume_number()
        assert lex._tokens == [Token(TokenType.NUMBER, "123", 1)]

    def test_float_number_token(self):
        lex = _Lexer("123.45")
        lex._consume_number()
        assert lex._tokens == [Token(TokenType.NUMBER, "123.45", 1)]

    def test_stops_before_non_digit_suffix(self):
        lex = _Lexer("42abc")
        lex._consume_number()
        assert lex._tokens == [Token(TokenType.NUMBER, "42", 1)]
        assert lex.peek() == "a"


class TestConsumeLoopVar:
    def test_valid_loopvar(self):
        lex = _Lexer("$col_2")
        lex._consume_loopvar()
        assert lex._tokens == [Token(TokenType.LOOPVAR, "$col_2", 1)]

    def test_bare_dollar_raises(self):
        with pytest.raises(LexError, match="must be followed by a letter or underscore"):
            _Lexer("$")._consume_loopvar()
