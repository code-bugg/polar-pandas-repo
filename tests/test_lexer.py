"""Unified unit tests for the lexer component.

Merged from: test_lexer_A.py, test_lexer_B.py, test_lexer_C.py, test_lexer_D.py.
"""
import pytest
from lexer.lexer import LexError, Token, TokenType, _Lexer
from lexer.lexer import TokenType, _Lexer, tokenize
from lexer.lexer import LexError, TokenType, tokenize


# ---------------------------------------------------------------------------
# Block from test_lexer_A.py
# ---------------------------------------------------------------------------
class TestTokenTypeEnum:
    def test_has_exactly_15_members(self):
        assert len(TokenType) == 15

    def test_all_expected_members_exist(self):
        expected = {
            "KW",
            "IDENT",
            "STRING",
            "NUMBER",
            "BOOL",
            "OP",
            "ARITH_OP",
            "LPAREN",
            "RPAREN",
            "COMMA",
            "ASSIGN",
            "PIPE",
            "COLON",
            "LOOPVAR",
            "EOF",
        }
        assert {m.name for m in TokenType} == expected


class TestTokenAndError:
    def test_token_frozen(self):
        t = Token(TokenType.KW, "READ", 1)
        with pytest.raises((AttributeError, TypeError)):
            t.value = "SAVE"  # type: ignore

    def test_lexerror_stores_line(self):
        err = LexError("bad", 9)
        assert err.line == 9
        assert "9" in str(err)


class TestCursor:
    def test_peek_and_advance(self):
        lex = _Lexer("ab")
        assert lex.peek() == "a"
        assert lex.advance() == "a"
        assert lex.peek() == "b"

    def test_advance_tracks_newline(self):
        lex = _Lexer("a\n")
        lex.advance()
        lex.advance()
        assert lex._line == 2


class TestDispatcherSingleChar:
    def test_paren_comma_colon(self):
        tokens = _Lexer("(),:")._tokenize_all()
        assert [t.type for t in tokens[:-1]] == [
            TokenType.LPAREN,
            TokenType.RPAREN,
            TokenType.COMMA,
            TokenType.COLON,
        ]

    def test_pipe_and_arith(self):
        tokens = _Lexer("->+-*/")._tokenize_all()
        assert [t.type for t in tokens[:-1]] == [
            TokenType.PIPE,
            TokenType.ARITH_OP,
            TokenType.ARITH_OP,
            TokenType.ARITH_OP,
            TokenType.ARITH_OP,
        ]

    def test_square_brackets_raise(self):
        with pytest.raises(LexError, match="Square brackets"):
            _Lexer("[")._tokenize_all()

    def test_eof_last(self):
        tokens = _Lexer("READ")._tokenize_all()
        assert tokens[-1].type == TokenType.EOF


# ---------------------------------------------------------------------------
# Block from test_lexer_B.py
# ---------------------------------------------------------------------------
def first(source: str):
    return tokenize(source)[0]


class TestKeywordsSet:
    def test_core_keywords_present(self):
        for kw in [
            "READ",
            "SAVE",
            "DROP",
            "CAST",
            "SET",
            "FILTER",
            "PLOT",
            "IF",
            "FOR",
            "ROWCOUNT",
            "ERROR",
            "STOP",
        ]:
            assert kw in _Lexer.KEYWORDS

    def test_values_not_reserved_keywords(self):
        # v5 spec: these are semantic values, not reserved words.
        for value_word in [
            "SUM",
            "AVERAGE",
            "COUNT",
            "MAX",
            "MIN",
            "INNER",
            "LEFT",
            "RIGHT",
            "OUTER",
            "mean",
            "median",
            "mode",
            "ffill",
            "bfill",
            "bar",
            "line",
            "scatter",
            "hist",
            "pie",
            "box",
        ]:
            assert value_word.upper() not in _Lexer.KEYWORDS


class TestConsumeWord:
    def test_keyword_is_case_insensitive_and_upper(self):
        tok = first("rEaD")
        assert tok.type == TokenType.KW
        assert tok.value == "READ"

    def test_boolean_lowercase_storage(self):
        tok_true = first("TRUE")
        tok_false = first("False")
        assert tok_true.type == TokenType.BOOL
        assert tok_true.value == "true"
        assert tok_false.type == TokenType.BOOL
        assert tok_false.value == "false"

    def test_identifier_preserves_original_case(self):
        tok = first("myDataFrame")
        assert tok.type == TokenType.IDENT
        assert tok.value == "myDataFrame"

    def test_non_reserved_values_are_ident(self):
        assert first("sum").type == TokenType.IDENT
        assert first("left").type == TokenType.IDENT
        assert first("bar").type == TokenType.IDENT

    def test_maximal_munch_word(self):
        tok = first("READING")
        assert tok.type == TokenType.IDENT
        assert tok.value == "READING"


# ---------------------------------------------------------------------------
# Block from test_lexer_C.py
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Block from test_lexer_D.py
# ---------------------------------------------------------------------------
def values_no_eof(source: str):
    return [t.value for t in tokenize(source) if t.type != TokenType.EOF]


class TestWhitespaceAndComments:
    def test_skips_whitespace(self):
        vals = values_no_eof("READ   'data.csv'\nSET col = 1")
        assert vals == ["READ", "'data.csv'", "SET", "col", "=", "1"]

    def test_skips_comment_to_eol(self):
        vals = values_no_eof("READ 'x.csv' # comment")
        assert vals == ["READ", "'x.csv'"]


class TestOperatorsAndPunctuation:
    def test_comparison_ops(self):
        toks = tokenize("a == b AND c != d AND e >= 1 AND f <= 2 AND g > 3 AND h < 4")
        ops = [t.value for t in toks if t.type == TokenType.OP]
        assert ops == ["==", "!=", ">=", "<=", ">", "<"]

    def test_assign_token(self):
        toks = tokenize("SET x = y")
        assert toks[2].type == TokenType.ASSIGN
        assert toks[2].value == "="

    def test_pipe_and_colon_tokens(self):
        toks = tokenize("READ 'x.csv' -> INFO :")
        types = [t.type for t in toks if t.type != TokenType.EOF]
        assert TokenType.PIPE in types
        assert TokenType.COLON in types


class TestErrorHandling:
    def test_bare_exclamation(self):
        with pytest.raises(LexError, match="did you mean '!='"):
            tokenize("! x")

    def test_bare_pipe_error(self):
        with pytest.raises(LexError, match="Use '->'"):
            tokenize("x | y")

    def test_ampersand_error(self):
        with pytest.raises(LexError, match="Use AND"):
            tokenize("x & y")

    def test_square_bracket_error(self):
        with pytest.raises(LexError, match="Square brackets"):
            tokenize("READ ['x.csv']")


class TestIntegration:
    def test_program_like_sequence(self):
        source = """
READ 'orders.csv'
-> FILTER WHERE amount >= 0
-> SET gross = amount + tax
IF ROWCOUNT orders > 1000 :
    SAVE 'big.csv'
END
"""
        toks = tokenize(source)
        assert toks[-1].type == TokenType.EOF
        assert any(t.type == TokenType.PIPE for t in toks)
        assert any(t.type == TokenType.ASSIGN for t in toks)
