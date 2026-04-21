import pytest

from lexer.lexer import LexError, TokenType, tokenize


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
