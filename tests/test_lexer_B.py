from lexer.lexer import TokenType, _Lexer, tokenize


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
