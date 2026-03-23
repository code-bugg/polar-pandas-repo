"""
polarpandas.lexer
=================
PolarPandas Translator — Component 1: Lexer
Fully implemented by Members A, B, C, D.

Public interface:
    tokenize(source: str) -> list[Token]

Shared types:
    TokenType  (enum)   — 14 token types
    Token      (dataclass)
    LexError   (exception)

Changelog:
    v1.0  Member A  — TokenType, Token, LexError, dispatcher, single-char tokens
    v1.0  Member B  — _consume_word, KEYWORDS set
    v1.0  Member C  — _consume_string, _consume_number, _consume_loopvar
    v1.0  Member D  — _skip_whitespace, _skip_comment, _consume_operator,
                      _unknown_char
    v1.1  Fix       — KEYWORDS rebuilt from spec exactly (66 words)
                    — Removed non-keyword values (agg funcs, join types, dtypes,
                      plot types) from KEYWORDS set
                    — Bare '=' now raises LexError (not a valid PolarPandas token)
                    — BOOL tokens normalised to lowercase ('true'/'false') per spec
                    — DOT token removed (not in spec; dot notation unsupported)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


# -----------------------------------------------------------------------------
# 1. TOKEN TYPES
#    15 types: 14 from spec + DOT (team extension for df.column notation).
#    NOTE: TRUE/FALSE are BOOL tokens, NOT KW tokens.
# -----------------------------------------------------------------------------

class TokenType(Enum):
    KW        = auto()  # Reserved keyword (case-insensitive). e.g. LOAD, WHERE, IF
    IDENT     = auto()  # User identifier. e.g. df, sales_df
    STRING    = auto()  # Double-quoted string literal. e.g. "amount", "data.csv"
    INTEGER   = auto()  # Integer literal.  e.g. 5, 100
    FLOAT     = auto()  # Float literal.    e.g. 3.14, 0.5
    BOOL      = auto()  # Boolean literal.  true | false  (stored lowercase)
    OP        = auto()  # Comparison operator: == != > < >= <=
    ARITH_OP  = auto()  # Arithmetic operator: + - / %  (see also STAR)
    LBRACKET  = auto()  # [
    RBRACKET  = auto()  # ]
    COMMA     = auto()  # ,
    STAR      = auto()  # *  wildcard in SELECT COLUMNS * and multiply in expr
    LOOPVAR   = auto()  # $identifier — loop variable in FOR EACH $col OVER ...
    EOF       = auto()  # Sentinel: end of input


# -----------------------------------------------------------------------------
# 2. TOKEN DATACLASS
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    """A single lexical unit produced by the lexer.

    Attributes
    ----------
    type  : TokenType
    value : str
        Normalisations applied:
        - Keywords  -> UPPER CASE  (e.g. 'load' -> 'LOAD')
        - Booleans  -> lower case  (e.g. 'TRUE' -> 'true')  [spec §2.1.5]
        - Strings   -> include enclosing double-quote characters
        - LOOPVAR   -> includes the leading '$'  (e.g. '$col')
    line  : int   1-based source line number.
    """
    type:  TokenType
    value: str
    line:  int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, line={self.line})"


# -----------------------------------------------------------------------------
# 3. LEXER ERROR
# -----------------------------------------------------------------------------

class LexError(Exception):
    """Raised on unrecognised character or malformed token.

    Attributes
    ----------
    message : str   Human-readable description.
    line    : int   1-based source line.
    """
    def __init__(self, message: str, line: int) -> None:
        super().__init__(f"[Lexer] line {line}: {message}")
        self.message = message
        self.line    = line


# -----------------------------------------------------------------------------
# 4. LEXER STATE
# -----------------------------------------------------------------------------

class _Lexer:
    """Internal mutable lexer state.
    tokenize(source) constructs one _Lexer per call and returns its token list.
    """

    # -------------------------------------------------------------------------
    # KEYWORDS  (Member B, corrected v1.1)
    #
    # Source of truth: spec §2.1.3  — 66 reserved words exactly.
    #
    # NOT included (these are identifier/string VALUES, not structural keywords):
    #   - Aggregation functions: sum, mean, median, min, max, count, std, var, ...
    #   - Join type values:      inner, left, right, outer, cross
    #   - Data type names:       int, float, str, bool, datetime, date, category
    #   - Plot type names:       bar, line, scatter, hist, box, pie, heatmap, ...
    #   - Fill method names:     mean, median, mode, ffill, bfill
    #   - Engine names:          pandas, polars  (validated by semantic layer)
    #
    # The semantic validator enforces allowed values for each field;
    # the lexer only needs to recognise structural keywords.
    # -------------------------------------------------------------------------
    KEYWORDS: set[str] = {
        # I/O
        "LOAD", "EXPORT",
        # Shared structural
        "AS", "TO", "FROM", "IN", "FORMAT", "ENGINE", "BACKEND",
        # Inspection
        "PREVIEW", "ROWS", "INFO", "DESCRIBE",
        # Selection
        "SELECT", "COLUMNS",
        # Filtering
        "FILTER", "WHERE", "AND", "OR", "NOT", "IS", "NULL", "BETWEEN",
        # Cleaning
        "DROP", "NULLS", "DUPLICATES", "KEEP",
        "FILL", "WITH", "COLUMN",
        # Transformation
        "CAST", "ADD", "RENAME", "SORT", "BY", "ASC", "DESC",
        # Aggregation
        "GROUP", "AGGREGATE", "COUNT",
        # Join
        "JOIN", "TYPE", "ON",
        # Plot
        "PLOT", "X", "Y", "HUE", "COLOR",
        "TITLE", "XLABEL", "YLABEL", "BINS", "FIGSIZE", "SAVE",
        # Engine config
        "SET",
        # Control flow (v1.1)
        "IF", "THEN", "ELSE", "END",
        "FOR", "EACH", "OVER", "DO",
        # Meta-conditions (v1.1)
        "EXISTS", "ROWCOUNT", "NULLCOUNT",
    }

    # Recognised by _consume_word but emitted as BOOL, not KW.
    _BOOLEANS: set[str] = {"TRUE", "FALSE"}

    def __init__(self, source: str) -> None:
        self._source: str         = source
        self._pos:    int         = 0
        self._line:   int         = 1
        self._tokens: list[Token] = []

    # -------------------------------------------------------------------------
    # Cursor primitives  (Member A)
    # -------------------------------------------------------------------------

    def peek(self, offset: int = 0) -> Optional[str]:
        """Return char at _pos+offset without advancing; None past end."""
        idx = self._pos + offset
        return self._source[idx] if idx < len(self._source) else None

    def advance(self) -> str:
        """Consume and return current char; increments line on newline."""
        if self._pos >= len(self._source):
            raise LexError("Unexpected end of source", self._line)
        ch = self._source[self._pos]
        self._pos += 1
        if ch == "\n":
            self._line += 1
        return ch

    def at_end(self) -> bool:
        return self._pos >= len(self._source)

    def _emit(self, ttype: TokenType, value: str, line: int) -> Token:
        tok = Token(ttype, value, line)
        self._tokens.append(tok)
        return tok

    # -------------------------------------------------------------------------
    # Maximal-munch dispatcher  (Member A)
    # -------------------------------------------------------------------------

    def _next_token(self) -> None:
        ch = self.peek()

        # 1. Whitespace / comments
        if ch in (" ", "\t", "\r", "\n"):
            self._skip_whitespace(); return
        if ch == "#":
            self._skip_comment(); return
        # 2. LOOPVAR  $ident  — must precede word/ident branch
        if ch == "$":
            self._consume_loopvar(); return
        # 3. STRING
        if ch == '"':
            self._consume_string(); return
        # 4. NUMBER
        if ch is not None and ch.isdigit():
            self._consume_number(); return
        # 5. WORD  (keyword / bool / ident)
        if ch is not None and (ch.isalpha() or ch == "_"):
            self._consume_word(); return
        # 6. Multi-char operators  — >= before >, == before =
        if ch in ("=", "!", ">", "<"):
            self._consume_operator(); return
        # 7. Single-char tokens
        line = self._line
        if ch == "[":  self.advance(); self._emit(TokenType.LBRACKET, "[", line); return
        if ch == "]":  self.advance(); self._emit(TokenType.RBRACKET, "]", line); return
        if ch == ",":  self.advance(); self._emit(TokenType.COMMA,    ",", line); return
        if ch == "*":  self.advance(); self._emit(TokenType.STAR,     "*", line); return
        if ch in ("+", "-", "/", "%"):
            self.advance(); self._emit(TokenType.ARITH_OP, ch, line); return
        # 8. Unknown
        self._unknown_char()

    def _tokenize_all(self) -> list[Token]:
        while not self.at_end():
            self._next_token()
        self._emit(TokenType.EOF, "", self._line)
        return self._tokens

    # -------------------------------------------------------------------------
    # Member B — word lexing
    # -------------------------------------------------------------------------

    def _consume_word(self) -> None:
        """Read longest [a-zA-Z_][a-zA-Z0-9_]* sequence, then classify:
            1. true/false (case-insensitive) -> BOOL  (stored lowercase per spec)
            2. In KEYWORDS (case-insensitive) -> KW   (stored UPPER)
            3. Anything else                 -> IDENT (original casing preserved)
        """
        start_line = self._line
        start_pos  = self._pos

        while not self.at_end():
            ch = self.peek()
            if ch is not None and (ch.isalnum() or ch == "_"):
                self.advance()
            else:
                break

        word  = self._source[start_pos : self._pos]
        upper = word.upper()

        if upper in self._BOOLEANS:
            # Spec §2.1.5: canonical form is lowercase true/false
            self._emit(TokenType.BOOL, upper.lower(), start_line)
        elif upper in self.KEYWORDS:
            self._emit(TokenType.KW, upper, start_line)
        else:
            # IDENT: preserve original casing (identifiers are case-sensitive)
            self._emit(TokenType.IDENT, word, start_line)

    # -------------------------------------------------------------------------
    # Member C — literals and loop variables
    # -------------------------------------------------------------------------

    def _consume_string(self) -> None:
        """Consume "..." with escapes \\" \\\\ \\n \\t.
        Emitted value includes the enclosing double-quote characters.
        """
        start_line = self._line
        self.advance()  # opening "
        chars: list[str] = []
        escapes = {'"': '"', "\\": "\\", "n": "\n", "t": "\t"}

        while True:
            ch = self.peek()
            if ch is None or ch == "\n":
                raise LexError("Unterminated string literal", start_line)
            if ch == '"':
                self.advance()  # closing "
                self._emit(TokenType.STRING, f'"{"".join(chars)}"', start_line)
                return
            if ch == "\\":
                self.advance()
                esc = self.peek()
                if esc is None or esc == "\n":
                    raise LexError("Unterminated escape sequence in string", start_line)
                if esc not in escapes:
                    raise LexError(
                        f"Invalid escape sequence '\\{esc}' in string literal",
                        self._line,
                    )
                self.advance()
                chars.append(escapes[esc])
            else:
                chars.append(self.advance())

    def _consume_number(self) -> None:
        """Consume integer or float.
        FLOAT   ::= digit+ '.' digit+
        INTEGER ::= digit+
        """
        start_line = self._line
        start_pos  = self._pos

        while (self.peek() or "").isdigit():
            self.advance()

        if self.peek() == "." and (self.peek(1) or "").isdigit():
            self.advance()  # '.'
            while (self.peek() or "").isdigit():
                self.advance()
            self._emit(TokenType.FLOAT, self._source[start_pos : self._pos], start_line)
        else:
            self._emit(TokenType.INTEGER, self._source[start_pos : self._pos], start_line)

    def _consume_loopvar(self) -> None:
        """Consume $identifier.  Emitted value includes the leading '$'."""
        start_line = self._line
        self.advance()  # '$'

        first = self.peek()
        if first is None or not (first.isalpha() or first == "_"):
            raise LexError(
                "'$' must be immediately followed by a letter or underscore "
                "(e.g. $col, $field)",
                start_line,
            )

        start_pos = self._pos
        while not self.at_end():
            ch = self.peek()
            if ch is not None and (ch.isalnum() or ch == "_"):
                self.advance()
            else:
                break

        name = self._source[start_pos : self._pos]
        self._emit(TokenType.LOOPVAR, f"${name}", start_line)

    # -------------------------------------------------------------------------
    # Member D — whitespace, comments, operators, errors
    # -------------------------------------------------------------------------

    def _skip_whitespace(self) -> None:
        """Skip all contiguous whitespace. Line tracking via advance()."""
        while self.peek() in (" ", "\t", "\r", "\n"):
            self.advance()

    def _skip_comment(self) -> None:
        """Skip from # to end of line (exclusive of the newline itself)."""
        self.advance()  # '#'
        while self.peek() is not None and self.peek() != "\n":
            self.advance()

    def _consume_operator(self) -> None:
        """Consume comparison operator using maximal-munch.

        Valid:   ==  !=  >=  <=  >  <
        Invalid: bare '='  -> LexError (no assignment in PolarPandas)
        Invalid: bare '!'  -> LexError
        """
        line = self._line
        ch   = self.peek()

        if ch == "=":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, "==", line)
            else:
                raise LexError(
                    "Unexpected '='; PolarPandas has no assignment operator. "
                    "Did you mean '==' for equality comparison?",
                    line,
                )

        elif ch == "!":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, "!=", line)
            else:
                raise LexError(
                    "Unexpected '!'; did you mean '!=' for inequality?",
                    line,
                )

        elif ch == ">":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, ">=", line)
            else:
                self._emit(TokenType.OP, ">", line)

        elif ch == "<":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, "<=", line)
            else:
                self._emit(TokenType.OP, "<", line)

    def _unknown_char(self) -> None:
        """Raise a descriptive LexError for any unhandled character."""
        ch   = self.peek()
        line = self._line

        hints: dict[str, str] = {
            ".": "unexpected '.'; dot notation is not supported in PolarPandas",
            "'": "single quotes are not supported; use double quotes: \"...\"",
            "|": "did you mean OR? Use the keyword OR for logical disjunction",
            "&": "did you mean AND? Use the keyword AND for logical conjunction",
            ";": "semicolons are not needed; statements are separated by newlines",
            "`": "backticks are not supported; use double quotes for strings",
            "@": "unexpected character '@'",
            "{": "braces are not supported in PolarPandas",
            "}": "unexpected '}'",
            "^": "unexpected '^'",
            "~": "unexpected '~'",
        }
        msg = hints.get(ch, f"unexpected character {ch!r}")
        raise LexError(msg, line)


# -----------------------------------------------------------------------------
# 5. PUBLIC ENTRY POINT
# -----------------------------------------------------------------------------

def tokenize(source: str) -> list[Token]:
    """Lex a PolarPandas source string into an ordered list of tokens.

    The final token is always Token(TokenType.EOF, "", <last_line>).
    Raises LexError on any unrecognised character or malformed token.

    Parameters
    ----------
    source : str   Full PolarPandas source text (UTF-8 string).

    Returns
    -------
    list[Token]   Flat token sequence including the terminal EOF token.

    Raises
    ------
    LexError

    Examples
    --------
    >>> tokens = tokenize('LOAD "data.csv" AS df')
    >>> [(t.type.name, t.value) for t in tokens]
    [('KW','LOAD'), ('STRING','"data.csv"'), ('KW','AS'), ('IDENT','df'), ('EOF','')]

    >>> tokens = tokenize('FILTER df WHERE "amount" >= 100')
    >>> [(t.type.name, t.value) for t in tokens]
    [('KW','FILTER'), ('IDENT','df'), ('KW','WHERE'),
     ('STRING','"amount"'), ('OP','>='), ('INTEGER','100'), ('EOF','')]

    >>> tokens = tokenize('FOR EACH $col OVER ["age","bmi"] DO')
    >>> [(t.type.name, t.value) for t in tokens]
    [('KW','FOR'),('KW','EACH'),('LOOPVAR','$col'),('KW','OVER'),
     ('LBRACKET','['),('STRING','"age"'),('COMMA',','),
     ('STRING','"bmi"'),('RBRACKET',']'),('KW','DO'),('EOF','')]

    >>> tokenize('true') [0]
    Token(BOOL, 'true', line=1)

    >>> tokenize('sum') [0]
    Token(IDENT, 'sum', line=1)
    """
    return _Lexer(source)._tokenize_all()