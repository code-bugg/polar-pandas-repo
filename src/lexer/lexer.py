"""
polarpandas.lexer
=================
PolarPandas Translator — Component 1: Lexer  (v2.0)

Public interface:
    tokenize(source: str) -> list[Token]

Changelog:
    v1.0  Members A/B/C/D — initial implementation
    v1.1  Fix  — KEYWORDS rebuilt from spec; BOOL lowercase; '=' is LexError;
                 DOT removed
    v2.0  Simplification phase —
            PIPE token added for '->' pipeline operator (15 token types total)
            '-' moved from single-char block into _consume_operator to enable
            maximal-munch: '->' emits PIPE, bare '-' emits ARITH_OP
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# 1. TOKEN TYPES  (15 total)
# ---------------------------------------------------------------------------

class TokenType(Enum):
    KW        = auto()  # Reserved keyword (case-insensitive)
    IDENT     = auto()  # User identifier: df, sales_df, my_col
    STRING    = auto()  # Double-quoted string literal: "amount", "data.csv"
    INTEGER   = auto()  # Integer literal: 5, 100
    FLOAT     = auto()  # Float literal: 3.14, 0.5
    BOOL      = auto()  # Boolean literal: true | false  (stored lowercase)
    OP        = auto()  # Comparison operator: == != > < >= <=
    ARITH_OP  = auto()  # Arithmetic operator: + - / %  (also see STAR)
    LBRACKET  = auto()  # [
    RBRACKET  = auto()  # ]
    COMMA     = auto()  # ,
    STAR      = auto()  # *  wildcard in SELECT COLUMNS * and multiply in expr
    LOOPVAR   = auto()  # $identifier — loop variable in FOR EACH $col OVER ...
    PIPE      = auto()  # ->  pipeline chaining operator  (v2.0)
    EOF       = auto()  # Sentinel: end of input


# ---------------------------------------------------------------------------
# 2. TOKEN DATACLASS
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    """A single lexical unit.

    Attributes
    ----------
    type  : TokenType
    value : str
        Normalisations:
        - Keywords  -> UPPER CASE
        - Booleans  -> lower case  ('true' / 'false')
        - Strings   -> include enclosing double-quote characters
        - LOOPVAR   -> includes the leading '$'
        - PIPE      -> literal '->'
    line  : int   1-based source line number.
    """
    type:  TokenType
    value: str
    line:  int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, line={self.line})"


# ---------------------------------------------------------------------------
# 3. LEXER ERROR
# ---------------------------------------------------------------------------

class LexError(Exception):
    """Raised on unrecognised character or malformed token.

    Attributes
    ----------
    message : str
    line    : int   1-based source line.
    """
    def __init__(self, message: str, line: int) -> None:
        super().__init__(f"[Lexer] line {line}: {message}")
        self.message = message
        self.line    = line


# ---------------------------------------------------------------------------
# 4. LEXER STATE
# ---------------------------------------------------------------------------

class _Lexer:

    # -----------------------------------------------------------------------
    # KEYWORDS — 66 reserved words from spec §2.1.3
    #
    # NOT included (these are identifier/string VALUES validated by semantic
    # layer): agg functions, join types, dtypes, plot types, fill methods,
    # engine names.
    # -----------------------------------------------------------------------
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

    _BOOLEANS: set[str] = {"TRUE", "FALSE"}

    def __init__(self, source: str) -> None:
        self._source: str         = source
        self._pos:    int         = 0
        self._line:   int         = 1
        self._tokens: list[Token] = []

    # -----------------------------------------------------------------------
    # Cursor primitives
    # -----------------------------------------------------------------------

    def peek(self, offset: int = 0) -> Optional[str]:
        idx = self._pos + offset
        return self._source[idx] if idx < len(self._source) else None

    def advance(self) -> str:
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

    # -----------------------------------------------------------------------
    # Maximal-munch dispatcher
    # -----------------------------------------------------------------------

    def _next_token(self) -> None:
        """Classify and consume the next token.

        Dispatch order:
            1. Whitespace / comments     — skip silently
            2. LOOPVAR  $ident           — before IDENT
            3. STRING   "..."
            4. NUMBER   digit...
            5. WORD     alpha...         — KW / BOOL / IDENT
            6. OPERATOR = ! > < - chars  — maximal-munch (-> before -)
            7. Single-char tokens        — [ ] , * + / %
            8. Unknown character         — LexError
        """
        ch = self.peek()

        if ch in (" ", "\t", "\r", "\n"):
            self._skip_whitespace(); return
        if ch == "#":
            self._skip_comment(); return
        if ch == "$":
            self._consume_loopvar(); return
        if ch == '"':
            self._consume_string(); return
        if ch is not None and ch.isdigit():
            self._consume_number(); return
        if ch is not None and (ch.isalpha() or ch == "_"):
            self._consume_word(); return

        # Multi-char operators + '->' PIPE — all handled together
        if ch in ("=", "!", ">", "<", "-"):
            self._consume_operator(); return

        # Single-char tokens
        line = self._line
        if ch == "[":  self.advance(); self._emit(TokenType.LBRACKET, "[", line); return
        if ch == "]":  self.advance(); self._emit(TokenType.RBRACKET, "]", line); return
        if ch == ",":  self.advance(); self._emit(TokenType.COMMA,    ",", line); return
        if ch == "*":  self.advance(); self._emit(TokenType.STAR,     "*", line); return
        if ch in ("+", "/", "%"):
            self.advance(); self._emit(TokenType.ARITH_OP, ch, line); return

        self._unknown_char()

    def _tokenize_all(self) -> list[Token]:
        while not self.at_end():
            self._next_token()
        self._emit(TokenType.EOF, "", self._line)
        return self._tokens

    # -----------------------------------------------------------------------
    # Member B — word lexing
    # -----------------------------------------------------------------------

    def _consume_word(self) -> None:
        """Read longest [a-zA-Z_][a-zA-Z0-9_]* and classify:
            1. true/false  -> BOOL  (lowercase)
            2. In KEYWORDS -> KW    (UPPER)
            3. else        -> IDENT (original casing)
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
            self._emit(TokenType.BOOL, upper, start_line)
        elif upper in self.KEYWORDS:
            self._emit(TokenType.KW, upper, start_line)
        else:
            self._emit(TokenType.IDENT, word, start_line)

    # -----------------------------------------------------------------------
    # Member C — literals and loop variables
    # -----------------------------------------------------------------------

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
                self.advance()
                self._emit(TokenType.STRING, f'"{"".join(chars)}"', start_line)
                return
            if ch == "\\":
                self.advance()
                esc = self.peek()
                if esc is None or esc == "\n":
                    raise LexError("Unterminated escape sequence in string", start_line)
                if esc not in escapes:
                    raise LexError(f"Invalid escape sequence '\\{esc}'", self._line)
                self.advance()
                chars.append(escapes[esc])
            else:
                chars.append(self.advance())

    def _consume_number(self) -> None:
        """FLOAT ::= digit+ '.' digit+   INTEGER ::= digit+"""
        start_line = self._line
        start_pos  = self._pos
        while (self.peek() or "").isdigit():
            self.advance()
        if self.peek() == "." and (self.peek(1) or "").isdigit():
            self.advance()
            while (self.peek() or "").isdigit():
                self.advance()
            self._emit(TokenType.FLOAT, self._source[start_pos:self._pos], start_line)
        else:
            self._emit(TokenType.INTEGER, self._source[start_pos:self._pos], start_line)

    def _consume_loopvar(self) -> None:
        """Consume $identifier. Emitted value includes the leading '$'."""
        start_line = self._line
        self.advance()  # '$'
        first = self.peek()
        if first is None or not (first.isalpha() or first == "_"):
            raise LexError(
                "'$' must be followed by a letter or underscore",
                start_line,
            )
        start_pos = self._pos
        while not self.at_end():
            ch = self.peek()
            if ch is not None and (ch.isalnum() or ch == "_"):
                self.advance()
            else:
                break
        self._emit(TokenType.LOOPVAR, f"${self._source[start_pos:self._pos]}", start_line)

    # -----------------------------------------------------------------------
    # Member D — whitespace, comments, operators, errors
    # -----------------------------------------------------------------------

    def _skip_whitespace(self) -> None:
        while self.peek() in (" ", "\t", "\r", "\n"):
            self.advance()

    def _skip_comment(self) -> None:
        self.advance()  # '#'
        while self.peek() is not None and self.peek() != "\n":
            self.advance()

    def _consume_operator(self) -> None:
        """Consume comparison/arithmetic operators using maximal-munch.

        Handles:
            ->   PIPE       (v2.0 pipeline operator)
            ==   OP
            !=   OP
            >=   OP
            <=   OP
            >    OP
            <    OP
            -    ARITH_OP   (when not followed by >)

        Bare '=' raises LexError (no assignment in PolarPandas).
        Bare '!' raises LexError.
        """
        line = self._line
        ch   = self.peek()

        # '->' pipeline operator  OR  '-' arithmetic subtraction
        if ch == "-":
            self.advance()
            if self.peek() == ">":
                self.advance()
                self._emit(TokenType.PIPE, "->", line)
            else:
                self._emit(TokenType.ARITH_OP, "-", line)
            return

        if ch == "=":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, "==", line)
            else:
                raise LexError(
                    "Unexpected '='; did you mean '==' for equality comparison?",
                    line,
                )
            return

        if ch == "!":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, "!=", line)
            else:
                raise LexError("Unexpected '!'; did you mean '!='?", line)
            return

        if ch == ">":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, ">=", line)
            else:
                self._emit(TokenType.OP, ">", line)
            return

        if ch == "<":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, "<=", line)
            else:
                self._emit(TokenType.OP, "<", line)
            return

    def _unknown_char(self) -> None:
        ch   = self.peek()
        line = self._line
        hints: dict[str, str] = {
            ".": "dot notation is not supported; column names are plain identifiers",
            "'": "single quotes not supported; use double quotes: \"...\"",
            "|": "did you mean OR? Use the keyword OR",
            "&": "did you mean AND? Use the keyword AND",
            ";": "semicolons not needed; statements are separated by newlines",
            "`": "backticks not supported; use double quotes for strings",
            "@": "unexpected character '@'",
            "{": "braces not supported in PolarPandas",
            "}": "unexpected '}'",
        }
        raise LexError(hints.get(ch, f"unexpected character {ch!r}"), line)


# ---------------------------------------------------------------------------
# 5. PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def tokenize(source: str) -> list[Token]:
    """Lex a PolarPandas source string into an ordered list of tokens.

    The final token is always Token(TokenType.EOF, "", <last_line>).

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
    >>> tokenize('LOAD "sales.csv" AS df')[0]
    Token(KW, 'LOAD', line=1)

    >>> tokenize('df -> FILTER WHERE amount > 0')
    [Token(IDENT,'df',1), Token(PIPE,'->',1), Token(KW,'FILTER',1),
     Token(KW,'WHERE',1), Token(IDENT,'amount',1), Token(OP,'>',1),
     Token(INTEGER,'0',1), Token(EOF,'',1)]

    >>> tokenize('true')[0]
    Token(BOOL, 'true', line=1)

    >>> tokenize('sum')[0]
    Token(IDENT, 'sum', line=1)
    """
    return _Lexer(source)._tokenize_all()