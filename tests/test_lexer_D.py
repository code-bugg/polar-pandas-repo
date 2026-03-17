import pytest
from lexer.lexer import tokenize, LexError, TokenType

class TestWhitespace:
    def test_skip_spaces(self):
        tokens = tokenize("LOAD   \"data.csv\"")
        assert tokens[0].value == "LOAD"
        assert tokens[1].value == '"data.csv"'

    def test_skip_tabs_and_newlines(self):
        tokens = tokenize("LOAD\t\"x.csv\"\nAS\tdf")
        assert [t.value for t in tokens if t.type != TokenType.EOF] == [
            "LOAD", '"x.csv"', "AS", "df"
        ]

    def test_carriage_return(self):
        # Windows line ending
        tokens = tokenize("LOAD\r\n\"data.csv\"")
        assert tokens[0].value == "LOAD"


class TestComments:
    def test_line_comment(self):
        tokens = tokenize("LOAD \"data.csv\" # Load the input file")
        assert tokens[0].value == "LOAD"
        assert tokens[2].type == TokenType.EOF

    def test_comment_at_eof(self):
        tokens = tokenize("df = data # end comment")
        assert tokens[-1].type == TokenType.EOF

    def test_comment_with_special_chars(self):
        tokens = tokenize("df = x # Comment with @#$%^&*()")
        # Should not raise
        assert tokens[-1].type == TokenType.EOF


class TestOperators:
    def test_equals_vs_double_equals(self):
        tokens1 = tokenize("a = b")
        tokens2 = tokenize("a == b")
        assert tokens1[1].value == "="
        assert tokens2[1].value == "=="

    def test_not_equals(self):
        tokens = tokenize("x != y")
        assert tokens[1].value == "!="

    def test_greater_equal(self):
        tokens = tokenize("x >= 5")
        assert tokens[1].value == ">="
        assert tokens[1].type == TokenType.OP

    def test_less_equal(self):
        tokens = tokenize("y <= 10")
        assert tokens[1].value == "<="

    def test_bare_greater_less(self):
        tokens1 = tokenize("x > 3")
        tokens2 = tokenize("y < 2")
        assert tokens1[1].value == ">"
        assert tokens2[1].value == "<"


class TestErrorHandling:
    def test_bare_exclamation(self):
        with pytest.raises(LexError, match="did you mean '!='"):
            tokenize("if ! x")

    def test_unknown_char_pipe(self):
        with pytest.raises(LexError):
            tokenize("LOAD x | FILTER")

    def test_unknown_char_ampersand(self):
        with pytest.raises(LexError, match="use AND"):
            tokenize("x & y")

    def test_single_quotes_error(self):
        with pytest.raises(LexError, match="double quotes"):
            tokenize("LOAD 'data.csv'")

    def test_backticks_error(self):
        with pytest.raises(LexError, match="backticks"):
            tokenize("SELECT `col` FROM df")

    def test_semicolon_error(self):
        with pytest.raises(LexError, match="semicolons"):
            tokenize("LOAD x;")

    def test_error_line_number(self):
        try:
            tokenize("LOAD x\nFILTER y\n@")
        except LexError as e:
            assert e.line == 3


class TestIntegration:
    def test_complex_statement_with_comments(self):
        source = """
        LOAD "data.csv" AS df  # Load data
        WHERE df.amount >= 100  # Filter by amount
        """
        tokens = tokenize(source)
        # Should parse without error
        assert tokens[-1].type == TokenType.EOF

    def test_all_operators_in_expression(self):
        tokens = tokenize("x == 1 AND y != 2 AND z >= 3 AND w <= 4")
        ops = [t.value for t in tokens if t.type == TokenType.OP]
        assert ops == ["==", "!=", ">=", "<="]