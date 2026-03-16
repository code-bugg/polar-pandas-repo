"""
tests/test_lexer_B.py
=====================
Unit tests for Member B's lexer subtask:
  - KEYWORDS set completeness (66+ reserved words, including v1.1 additions)
  - _consume_word(): keyword recognition (case-insensitive, stored UPPERCASE)
  - _consume_word(): boolean literals (true/false → BOOL token)
  - _consume_word(): identifiers (anything else → IDENT token)
  - Maximal munch: longest [a-zA-Z_][a-zA-Z0-9_]* sequence consumed

NOTE: Because Members C and D stubs are still unimplemented, tests here
      operate directly on _Lexer instances with stubs monkey-patched out.
"""

import pytest
from lexer.lexer import TokenType, Token, LexError, _Lexer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _patch_stubs(lex: _Lexer) -> None:
    """Monkeypatch Member C/D stubs so they don't raise NotImplementedError."""
    lex._skip_whitespace = lambda: _do_skip_ws(lex)       # type: ignore
    lex._skip_comment    = lambda: _do_skip_comment(lex)   # type: ignore
    lex._consume_string  = lambda: _do_skip_string(lex)    # type: ignore
    lex._consume_number  = lambda: _do_skip_number(lex)    # type: ignore
    lex._consume_loopvar = lambda: _do_skip_loopvar(lex)   # type: ignore
    lex._consume_operator = lambda: _do_skip_operator(lex) # type: ignore
    lex._unknown_char    = lambda: _do_unknown(lex)        # type: ignore


def _do_skip_ws(lex: _Lexer) -> None:
    while not lex.at_end() and lex.peek() in (" ", "\t", "\r", "\n"):
        lex.advance()


def _do_skip_comment(lex: _Lexer) -> None:
    while not lex.at_end() and lex.peek() != "\n":
        lex.advance()


def _do_skip_string(lex: _Lexer) -> None:
    lex.advance()  # opening "
    while not lex.at_end() and lex.peek() != '"':
        lex.advance()
    if not lex.at_end():
        lex.advance()  # closing "


def _do_skip_number(lex: _Lexer) -> None:
    while not lex.at_end() and (lex.peek() or "").isdigit():
        lex.advance()


def _do_skip_loopvar(lex: _Lexer) -> None:
    lex.advance()  # $
    while not lex.at_end() and (lex.peek() or "").isalnum():
        lex.advance()


def _do_skip_operator(lex: _Lexer) -> None:
    lex.advance()
    if not lex.at_end() and lex.peek() in ("=",):
        lex.advance()


def _do_unknown(lex: _Lexer) -> None:
    ch = lex.advance()
    raise LexError(f"Unexpected character '{ch}'", lex._line)


def _tokenize_patched(source: str) -> list[Token]:
    """Tokenize using a fully patched _Lexer (stubs replaced)."""
    lex = _Lexer(source)
    _patch_stubs(lex)
    return lex._tokenize_all()


# ─────────────────────────────────────────────────────────────────────────────
# KEYWORDS set
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywordsSet:

    def test_keywords_set_is_not_empty(self):
        assert len(_Lexer.KEYWORDS) > 0

    def test_keywords_at_least_66(self):
        """Spec requires 66 reserved words."""
        assert len(_Lexer.KEYWORDS) >= 66

    def test_all_keywords_are_uppercase(self):
        for kw in _Lexer.KEYWORDS:
            assert kw == kw.upper(), f"Keyword {kw!r} is not uppercase"

    def test_all_keywords_are_strings(self):
        for kw in _Lexer.KEYWORDS:
            assert isinstance(kw, str)

    def test_true_false_not_in_keywords(self):
        """true/false must be BOOL tokens, not keywords."""
        assert "TRUE" not in _Lexer.KEYWORDS
        assert "FALSE" not in _Lexer.KEYWORDS

    # ── v1.0 core keywords ───────────────────────────────────────────────

    @pytest.mark.parametrize("kw", [
        "LOAD", "EXPORT", "PREVIEW", "INFO", "DESCRIBE",
        "AS", "TO", "FROM", "IN",
        "SELECT", "COLUMNS", "FILTER", "WHERE",
        "DROP", "NULLS", "DUPLICATES", "FILL", "WITH",
        "CAST", "ADD", "COLUMN", "RENAME", "SORT", "BY",
        "GROUP", "COUNT", "JOIN",
        "ASC", "DESC",
        "PLOT",
        "SET", "ENGINE",
        "SUM", "MEAN", "MIN", "MAX", "MEDIAN",
        "INT", "FLOAT", "STR", "BOOL",
        "INNER", "LEFT", "RIGHT", "OUTER", "ON",
        "AND", "OR", "NOT",
    ])
    def test_core_keyword_present(self, kw):
        assert kw in _Lexer.KEYWORDS, f"Missing core keyword: {kw}"

    # ── v1.1 control-flow keywords ───────────────────────────────────────

    @pytest.mark.parametrize("kw", [
        "IF", "THEN", "ELSE", "END",
        "FOR", "EACH", "OVER", "DO",
        "EXISTS", "ROWCOUNT", "NULLCOUNT",
    ])
    def test_v11_keyword_present(self, kw):
        assert kw in _Lexer.KEYWORDS, f"Missing v1.1 keyword: {kw}"


# ─────────────────────────────────────────────────────────────────────────────
# _consume_word(): Keyword recognition
# ─────────────────────────────────────────────────────────────────────────────

class TestConsumeWordKeywords:

    def test_simple_keyword_uppercase(self):
        tokens = _tokenize_patched("LOAD")
        assert tokens[0] == Token(TokenType.KW, "LOAD", 1)

    def test_keyword_lowercase_normalised(self):
        """Keywords are case-insensitive; value stored as UPPERCASE."""
        tokens = _tokenize_patched("load")
        assert tokens[0] == Token(TokenType.KW, "LOAD", 1)

    def test_keyword_mixed_case(self):
        tokens = _tokenize_patched("LoAd")
        assert tokens[0] == Token(TokenType.KW, "LOAD", 1)

    @pytest.mark.parametrize("kw", [
        "LOAD", "EXPORT", "PREVIEW", "INFO", "DESCRIBE",
        "SELECT", "FILTER", "DROP", "FILL", "CAST",
        "ADD", "RENAME", "SORT", "GROUP", "COUNT",
        "JOIN", "PLOT", "SET",
    ])
    def test_statement_keywords_produce_kw_token(self, kw):
        tokens = _tokenize_patched(kw)
        assert tokens[0].type == TokenType.KW
        assert tokens[0].value == kw

    @pytest.mark.parametrize("kw", [
        "IF", "THEN", "ELSE", "END", "FOR", "EACH", "OVER", "DO",
    ])
    def test_v11_control_flow_keywords_produce_kw_token(self, kw):
        tokens = _tokenize_patched(kw)
        assert tokens[0].type == TokenType.KW
        assert tokens[0].value == kw

    @pytest.mark.parametrize("kw", ["EXISTS", "ROWCOUNT", "NULLCOUNT"])
    def test_v11_meta_condition_keywords_produce_kw_token(self, kw):
        tokens = _tokenize_patched(kw)
        assert tokens[0].type == TokenType.KW
        assert tokens[0].value == kw

    def test_keyword_stored_uppercase(self):
        tokens = _tokenize_patched("select")
        assert tokens[0].value == "SELECT"

    def test_keyword_line_number_tracked(self):
        tokens = _tokenize_patched("LOAD")
        assert tokens[0].line == 1


# ─────────────────────────────────────────────────────────────────────────────
# _consume_word(): Boolean literals
# ─────────────────────────────────────────────────────────────────────────────

class TestConsumeWordBooleans:

    def test_true_lowercase(self):
        tokens = _tokenize_patched("true")
        assert tokens[0] == Token(TokenType.BOOL, "TRUE", 1)

    def test_false_lowercase(self):
        tokens = _tokenize_patched("false")
        assert tokens[0] == Token(TokenType.BOOL, "FALSE", 1)

    def test_true_uppercase(self):
        tokens = _tokenize_patched("TRUE")
        assert tokens[0] == Token(TokenType.BOOL, "TRUE", 1)

    def test_false_uppercase(self):
        tokens = _tokenize_patched("FALSE")
        assert tokens[0] == Token(TokenType.BOOL, "FALSE", 1)

    def test_true_mixed_case(self):
        tokens = _tokenize_patched("TrUe")
        assert tokens[0] == Token(TokenType.BOOL, "TRUE", 1)

    def test_false_mixed_case(self):
        tokens = _tokenize_patched("fAlSe")
        assert tokens[0] == Token(TokenType.BOOL, "FALSE", 1)

    def test_bool_not_kw(self):
        """true/false must be BOOL, not KW."""
        for word in ("true", "false", "TRUE", "FALSE"):
            tokens = _tokenize_patched(word)
            assert tokens[0].type == TokenType.BOOL, \
                f"{word!r} should be BOOL, got {tokens[0].type}"

    def test_bool_not_ident(self):
        """true/false must be BOOL, not IDENT."""
        for word in ("true", "false"):
            tokens = _tokenize_patched(word)
            assert tokens[0].type != TokenType.IDENT


# ─────────────────────────────────────────────────────────────────────────────
# _consume_word(): Identifiers
# ─────────────────────────────────────────────────────────────────────────────

class TestConsumeWordIdentifiers:

    def test_simple_identifier(self):
        tokens = _tokenize_patched("df")
        assert tokens[0] == Token(TokenType.IDENT, "df", 1)

    def test_identifier_preserves_case(self):
        """Unlike keywords, identifiers keep their original casing."""
        tokens = _tokenize_patched("myDataFrame")
        assert tokens[0].value == "myDataFrame"

    def test_underscore_start(self):
        tokens = _tokenize_patched("_temp")
        assert tokens[0].type == TokenType.IDENT
        assert tokens[0].value == "_temp"

    def test_identifier_with_digits(self):
        tokens = _tokenize_patched("col2")
        assert tokens[0].type == TokenType.IDENT
        assert tokens[0].value == "col2"

    def test_identifier_with_underscores(self):
        tokens = _tokenize_patched("sales_df")
        assert tokens[0].type == TokenType.IDENT
        assert tokens[0].value == "sales_df"

    def test_single_letter_identifier(self):
        tokens = _tokenize_patched("x")
        # 'X' is a keyword, so lowercase 'x' maps to KW too
        # But a non-keyword single letter like 'z' should be IDENT
        tokens = _tokenize_patched("z")
        assert tokens[0].type == TokenType.IDENT
        assert tokens[0].value == "z"

    def test_long_identifier(self):
        tokens = _tokenize_patched("very_long_identifier_name_123")
        assert tokens[0].type == TokenType.IDENT
        assert tokens[0].value == "very_long_identifier_name_123"

    def test_identifier_not_keyword(self):
        """A word that is not in the keyword set should be IDENT."""
        tokens = _tokenize_patched("myvar")
        assert tokens[0].type == TokenType.IDENT

    def test_all_underscores(self):
        tokens = _tokenize_patched("___")
        assert tokens[0].type == TokenType.IDENT
        assert tokens[0].value == "___"


# ─────────────────────────────────────────────────────────────────────────────
# Maximal munch
# ─────────────────────────────────────────────────────────────────────────────

class TestMaximalMunch:

    def test_keyword_prefix_not_split(self):
        """'LOADING' is not the keyword 'LOAD' + 'ING'; it's one IDENT."""
        tokens = _tokenize_patched("LOADING")
        assert tokens[0].type == TokenType.IDENT
        assert tokens[0].value == "LOADING"

    def test_keyword_as_prefix_of_longer_word(self):
        """'SELECT' is a keyword, but 'SELECTED' is an identifier."""
        tokens = _tokenize_patched("SELECTED")
        assert tokens[0].type == TokenType.IDENT
        assert tokens[0].value == "SELECTED"

    def test_keyword_followed_by_bracket(self):
        """Word stops at non-alphanumeric character."""
        tokens = _tokenize_patched("DROP[")
        assert tokens[0] == Token(TokenType.KW, "DROP", 1)
        assert tokens[1] == Token(TokenType.LBRACKET, "[", 1)

    def test_word_stops_at_comma(self):
        tokens = _tokenize_patched("df,")
        assert tokens[0] == Token(TokenType.IDENT, "df", 1)
        assert tokens[1] == Token(TokenType.COMMA, ",", 1)

    def test_word_stops_at_star(self):
        tokens = _tokenize_patched("df*")
        assert tokens[0] == Token(TokenType.IDENT, "df", 1)
        assert tokens[1] == Token(TokenType.STAR, "*", 1)


# ─────────────────────────────────────────────────────────────────────────────
# Multiple words in sequence
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleWords:

    def test_two_keywords_with_space(self):
        tokens = _tokenize_patched("DROP NULLS")
        assert tokens[0] == Token(TokenType.KW, "DROP", 1)
        assert tokens[1] == Token(TokenType.KW, "NULLS", 1)

    def test_keyword_then_identifier(self):
        tokens = _tokenize_patched("LOAD df")
        assert tokens[0] == Token(TokenType.KW, "LOAD", 1)
        assert tokens[1] == Token(TokenType.IDENT, "df", 1)

    def test_keyword_bool_ident_sequence(self):
        tokens = _tokenize_patched("FILTER df WHERE true")
        assert tokens[0] == Token(TokenType.KW, "FILTER", 1)
        assert tokens[1] == Token(TokenType.IDENT, "df", 1)
        assert tokens[2] == Token(TokenType.KW, "WHERE", 1)
        assert tokens[3] == Token(TokenType.BOOL, "TRUE", 1)

    def test_eof_always_last(self):
        tokens = _tokenize_patched("LOAD df")
        assert tokens[-1].type == TokenType.EOF

    def test_case_insensitive_in_sequence(self):
        tokens = _tokenize_patched("load DF")
        assert tokens[0] == Token(TokenType.KW, "LOAD", 1)
        assert tokens[1] == Token(TokenType.IDENT, "DF", 1)


# ─────────────────────────────────────────────────────────────────────────────
# Line tracking across words
# ─────────────────────────────────────────────────────────────────────────────

class TestLineTracking:

    def test_word_on_second_line(self):
        tokens = _tokenize_patched("\nLOAD")
        assert tokens[0] == Token(TokenType.KW, "LOAD", 2)

    def test_words_on_multiple_lines(self):
        tokens = _tokenize_patched("LOAD\ndf")
        assert tokens[0].line == 1
        assert tokens[1].line == 2

    def test_many_newlines(self):
        tokens = _tokenize_patched("\n\n\nSELECT")
        assert tokens[0].line == 4


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_single_underscore_is_ident(self):
        tokens = _tokenize_patched("_")
        assert tokens[0].type == TokenType.IDENT
        assert tokens[0].value == "_"

    def test_keyword_bool_type_is_kw_not_bool(self):
        """'BOOL' as a type keyword (e.g. CAST ... TO BOOL) is KW, not BOOL."""
        tokens = _tokenize_patched("BOOL")
        assert tokens[0].type == TokenType.KW
        assert tokens[0].value == "BOOL"

    def test_keyword_float_type_is_kw(self):
        """'FLOAT' as a type keyword is KW (distinct from FLOAT literal)."""
        tokens = _tokenize_patched("FLOAT")
        assert tokens[0].type == TokenType.KW
        assert tokens[0].value == "FLOAT"

    def test_if_then_else_end_sequence(self):
        """v1.1 control flow block keywords."""
        tokens = _tokenize_patched("IF THEN ELSE END")
        types = [t.type for t in tokens[:-1]]  # exclude EOF
        assert types == [TokenType.KW] * 4
        values = [t.value for t in tokens[:-1]]
        assert values == ["IF", "THEN", "ELSE", "END"]

    def test_for_each_over_do_end_sequence(self):
        """v1.1 loop block keywords."""
        tokens = _tokenize_patched("FOR EACH OVER DO END")
        types = [t.type for t in tokens[:-1]]
        assert types == [TokenType.KW] * 5
        values = [t.value for t in tokens[:-1]]
        assert values == ["FOR", "EACH", "OVER", "DO", "END"]
