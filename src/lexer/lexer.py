"""
polarpandas.lexer
=================
PolarPandas Translator — Component 1: Lexer
Member A subtask: TokenType enum, Token dataclass, tokenize() entry point,
                  and the maximal-munch character dispatcher.

Subtasks B, C, D have added their consume_*() helpers into this file.
The public interface is:

    tokenize(source: str) -> list[Token]

and the shared types:

    TokenType  (enum)
    Token      (dataclass)
    LexError   (exception)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. TOKEN TYPES
#    Spec reference: §2.1, §5.2 (LOOPVAR added in v1.1)
#    14 structural token types total.
# ─────────────────────────────────────────────────────────────────────────────

class TokenType(Enum):
    # ── Lexical categories ────────────────────────────────────────────────
    KW        = auto()   # Reserved keyword (case-insensitive). e.g. LOAD, WHERE, IF
    IDENT     = auto()   # User identifier (dataframe variable name). e.g. df, sales_df
    STRING    = auto()   # Double-quoted string literal. e.g. "amount", "data.csv"
    INTEGER   = auto()   # Integer literal.  e.g. 5, 100
    FLOAT     = auto()   # Float literal.    e.g. 3.14, 0.5
    BOOL      = auto()   # Boolean literal.  true | false  (case-insensitive)
    OP        = auto()   # Comparison operator: == != > < >= <=
    ARITH_OP  = auto()   # Arithmetic operator: + - * / %  (also see STAR)
    LBRACKET  = auto()   # [
    RBRACKET  = auto()   # ]
    COMMA     = auto()   # ,
    STAR      = auto()   # *  used as wildcard in SELECT … COLUMNS *
    LOOPVAR   = auto()   # $identifier  loop variable in FOR EACH $col OVER … (v1.1)
    DOT       = auto()   # .  member access separator, e.g. df.amount
    EOF       = auto()   # Sentinel: end of input


# ─────────────────────────────────────────────────────────────────────────────
# 2. TOKEN DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Token:
    """A single lexical unit produced by the lexer.

    Attributes
    ----------
    type  : TokenType
        The category of the token.
    value : str
        The raw source text for this token.
        Keywords are stored in UPPER CASE (normalised by Member B).
        String literals include the enclosing double-quote characters.
        LOOPVAR tokens include the leading ``$``.
    line  : int
        1-based source line number where the token starts.
    """
    type:  TokenType
    value: str
    line:  int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, line={self.line})"


# ─────────────────────────────────────────────────────────────────────────────
# 3. LEXER ERROR
# ─────────────────────────────────────────────────────────────────────────────

class LexError(Exception):
    """Raised when the lexer encounters an unrecognised character or
    malformed token (e.g. unterminated string, bare ``$`` not followed
    by an identifier).

    Attributes
    ----------
    message : str
        Human-readable description of the problem.
    line    : int
        1-based source line where the error occurred.
    """
    def __init__(self, message: str, line: int) -> None:
        super().__init__(f"[Lexer] line {line}: {message}")
        self.message = message
        self.line    = line


# ─────────────────────────────────────────────────────────────────────────────
# 4. LEXER STATE
# ─────────────────────────────────────────────────────────────────────────────

class _Lexer:
    """Internal mutable lexer state.

    The public entry point is ``tokenize(source)`` which constructs one
    ``_Lexer`` instance per call and returns its token list.
    """

    def __init__(self, source: str) -> None:
        self._source: str         = source
        self._pos:    int         = 0       # current character index
        self._line:   int         = 1       # current 1-based line number
        self._tokens: list[Token] = []

    # ── Cursor primitives ─────────────────────────────────────────────────

    def peek(self, offset: int = 0) -> Optional[str]:
        """Return the character at ``_pos + offset`` without advancing.
        Returns None if past end-of-source.
        """
        idx = self._pos + offset
        return self._source[idx] if idx < len(self._source) else None

    def advance(self) -> str:
        """Consume and return the current character, updating the line
        counter when a newline is consumed.
        """
        if self._pos >= len(self._source):
            raise LexError("Unexpected end of source", self._line)
        ch = self._source[self._pos]
        self._pos += 1
        if ch == "\n":
            self._line += 1
        return ch

    def at_end(self) -> bool:
        """True when the cursor is at or past the end of the source."""
        return self._pos >= len(self._source)

    # ── Token emission helper ─────────────────────────────────────────────

    def _emit(self, ttype: TokenType, value: str, line: int) -> Token:
        """Construct a Token and append it to the internal list."""
        tok = Token(ttype, value, line)
        self._tokens.append(tok)
        return tok

    # ── Maximal-munch dispatcher (Member A) ───────────────────────────────

    def _next_token(self) -> None:
        """Classify and consume the next token from the source.

        Dispatch order matters for maximal-munch correctness:
            1. Whitespace / comments  (Member D) — skip silently
            2. LOOPVAR  $ident        (Member C) — must check before IDENT
            3. STRING   "…"           (Member C)
            4. NUMBER   digit…        (Member C) — FLOAT before INTEGER
            5. WORD     alpha…        (Member B) — KW / BOOL / IDENT
            6. OPERATOR multi-char    (Member D) — >= before >, == before =
            7. Single-char tokens     (Member A) — [ ] , * and ARITH_OP
            8. Unknown character      (Member D) — LexError
        """
        ch = self.peek()

        # ── 1. Whitespace and comments ────────────────────────────────────
        if ch in (" ", "\t", "\r", "\n"):
            self._skip_whitespace()
            return
        if ch == "#":
            self._skip_comment()
            return

        # ── 2. LOOPVAR  $ident ────────────────────────────────────────────
        if ch == "$":
            self._consume_loopvar()
            return

        # ── 3. STRING literal ─────────────────────────────────────────────
        if ch == '"':
            self._consume_string()
            return

        # ── 4. NUMBER literal ─────────────────────────────────────────────
        if ch is not None and ch.isdigit():
            self._consume_number()
            return

        # ── 5. WORD: keyword, boolean, or identifier ──────────────────────
        if ch is not None and (ch.isalpha() or ch == "_"):
            self._consume_word()
            return

        # ── 6. Multi-character operators ──────────────────────────────────
        if ch in ("=", "!", ">", "<"):
            self._consume_operator()
            return

        # ── 7. Single-character tokens (Member A) ─────────────────────────
        line = self._line
        if ch == "[":
            self.advance()
            self._emit(TokenType.LBRACKET, "[", line)
            return
        if ch == "]":
            self.advance()
            self._emit(TokenType.RBRACKET, "]", line)
            return
        if ch == ",":
            self.advance()
            self._emit(TokenType.COMMA, ",", line)
            return
        if ch == "*":
            # Contextual disambiguation (STAR vs ARITH_OP) is resolved by
            # the parser, not the lexer. The lexer always emits STAR here;
            # the parser interprets it as ARITH_OP inside <expr> context.
            # See spec §3.4.1.
            self.advance()
            self._emit(TokenType.STAR, "*", line)
            return
        if ch in ("+", "-", "/", "%"):
            self.advance()
            self._emit(TokenType.ARITH_OP, ch, line)
            return
        if ch == ".":
            self.advance()
            self._emit(TokenType.DOT, ".", line)
            return

        # ── 8. Unknown character ───────────────────────────────────────────
        self._unknown_char()

    # ── Main tokenisation loop ────────────────────────────────────────────

    def _tokenize_all(self) -> list[Token]:
        """Drive the dispatcher until EOF, then append the EOF sentinel."""
        while not self.at_end():
            self._next_token()
        self._emit(TokenType.EOF, "", self._line)
        return self._tokens

    # ── KEYWORDS set (Member B) ───────────────────────────────────────────
    # 66 reserved words (all stored UPPERCASE). Includes v1.1 additions:
    # IF, THEN, ELSE, END, FOR, EACH, OVER, DO, EXISTS, ROWCOUNT, NULLCOUNT.

    KEYWORDS: set[str] = {
        # ── I/O & inspection ──────────────────────────────────────────────
        "LOAD", "EXPORT", "PREVIEW", "INFO", "DESCRIBE",
        # ── Data source / target ──────────────────────────────────────────
        "AS", "TO", "FROM", "IN", "INTO",
        # ── Selection & filtering ─────────────────────────────────────────
        "SELECT", "COLUMNS", "FILTER", "WHERE",
        # ── Cleaning ──────────────────────────────────────────────────────
        "DROP", "NULLS", "DUPLICATES", "FILL", "WITH",
        # ── Column operations ─────────────────────────────────────────────
        "CAST", "ADD", "COLUMN", "RENAME", "SORT", "BY",
        # ── Aggregation ───────────────────────────────────────────────────
        "GROUP", "COUNT", "JOIN",
        # ── Sorting modifiers ─────────────────────────────────────────────
        "ASC", "DESC",
        # ── Plot ──────────────────────────────────────────────────────────
        "PLOT", "TYPE", "X", "Y", "TITLE", "SAVE",
        # ── Engine ────────────────────────────────────────────────────────
        "SET", "ENGINE",
        # ── Aggregation functions ─────────────────────────────────────────
        "SUM", "MEAN", "MIN", "MAX", "MEDIAN",
        # ── Data types ────────────────────────────────────────────────────
        "INT", "FLOAT", "STR", "BOOL",
        # ── Join types ────────────────────────────────────────────────────
        "INNER", "LEFT", "RIGHT", "OUTER", "ON",
        # ── Logical operators (in conditions) ─────────────────────────────
        "AND", "OR", "NOT",
        # ── Control flow (v1.1) ───────────────────────────────────────────
        "IF", "THEN", "ELSE", "END",
        "FOR", "EACH", "OVER", "DO",
        # ── Meta-condition keywords (v1.1) ────────────────────────────────
        "EXISTS", "ROWCOUNT", "NULLCOUNT",
        # ── Plot types ────────────────────────────────────────────────────
        "BAR", "LINE", "SCATTER", "HIST",
        # ── Miscellaneous ─────────────────────────────────────────────────
        "USING", "ROWS",
    }

    # ── Member B: words ───────────────────────────────────────────────────

    def _consume_word(self) -> None:
        """Read the longest [a-zA-Z_][a-zA-Z0-9_]* sequence, then classify:
        - ``true`` / ``false`` (case-insensitive) → BOOL token
        - word found in KEYWORDS (case-insensitive) → KW token (stored UPPER)
        - anything else → IDENT token (original casing preserved)
        """
        start_line = self._line
        start_pos  = self._pos

        while not self.at_end():
            ch = self.peek()
            if ch is not None and (ch.isalnum() or ch == "_"):
                self.advance()
            else:
                break

        word  = self._source[start_pos:self._pos]
        upper = word.upper()

        if upper in ("TRUE", "FALSE"):
            self._emit(TokenType.BOOL, upper, start_line)
        elif upper in self.KEYWORDS:
            self._emit(TokenType.KW, upper, start_line)
        else:
            self._emit(TokenType.IDENT, word, start_line)

    # ── Member C: strings, numbers, loop variables ────────────────────────

    def _consume_string(self) -> None:
        start_line = self._line
        if self.peek() != '"':
            raise LexError("String literal must start with double quote", start_line)

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
                    raise LexError("Unterminated string literal", start_line)
                if esc not in escapes:
                    raise LexError(f"Invalid escape sequence '\\{esc}'", self._line)
                self.advance()
                chars.append(escapes[esc])
                continue
            chars.append(self.advance())

    def _consume_number(self) -> None:
        start_line = self._line
        start_pos  = self._pos

        while (self.peek() or "").isdigit():
            self.advance()

        # Check for decimal point followed by more digits → FLOAT
        if self.peek() == "." and (self.peek(1) or "").isdigit():
            self.advance()  # consume "."
            while (self.peek() or "").isdigit():
                self.advance()
            self._emit(TokenType.FLOAT, self._source[start_pos:self._pos], start_line)
            return

        self._emit(TokenType.INTEGER, self._source[start_pos:self._pos], start_line)

    def _consume_loopvar(self) -> None:
        start_line = self._line
        if self.peek() != "$":
            raise LexError("Loop variable must start with '$'", start_line)

        self.advance()  # consume "$"
        first = self.peek()
        if first is None or not (first.isalpha() or first == "_"):
            raise LexError("'$' must be followed by a letter or underscore", start_line)

        start_pos = self._pos
        while True:
            ch = self.peek()
            if ch is not None and (ch.isalnum() or ch == "_"):
                self.advance()
            else:
                break

        self._emit(TokenType.LOOPVAR, f"${self._source[start_pos:self._pos]}", start_line)

    # ── Member D: whitespace, comments, operators, errors ─────────────────

    def _skip_whitespace(self) -> None:
        """Consume and skip horizontal/vertical whitespace.

        Handles: space (U+0020), tab (U+0009), CR (U+000D), LF (U+000A).
        Line tracking is automatic via advance() when consuming \\n.
        """
        while self.peek() in (" ", "\t", "\r", "\n"):
            self.advance()

    def _skip_comment(self) -> None:
        """Consume a line comment from # to end-of-line (exclusive).

        The newline itself is NOT consumed here; _skip_whitespace() handles
        it on the next dispatch so the line counter is updated correctly.
        """
        self.advance()  # consume "#"
        while self.peek() is not None and self.peek() != "\n":
            self.advance()

    def _consume_operator(self) -> None:
        """Consume a comparison operator using maximal-munch.

        Two-character forms: == != >= <=
        Single-character forms: = > <
        Bare ! (not followed by =) raises LexError.
        """
        line = self._line
        ch   = self.peek()

        if ch == "=":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, "==", line)
            else:
                self._emit(TokenType.OP, "=", line)

        elif ch == "!":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, "!=", line)
            else:
                raise LexError("Unexpected character '!'; did you mean '!='?", line)

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
        """Raise a LexError for any character not handled by the dispatcher."""
        ch   = self.peek()
        line = self._line

        suggestions = {
            "|": "chaining without | (use keywords like WHERE, FILTER)",
            "&": "use AND for logical conjunction",
            "^": "unexpected character; did you mean ^ operator?",
            "~": "unexpected character",
            "`": "backticks are not supported; use double quotes for strings",
            "'": "single quotes not supported; use double quotes \"...\" for strings",
            "@": "unexpected character (decorators not supported)",
            "{": "braces not supported (use parentheses instead: (...))",
            "}": "unexpected }; did you mean to close with )?",
            ";": "semicolons not needed; use newlines or keywords to separate statements",
        }

        msg = suggestions.get(ch, f"unexpected character {ch!r}")
        raise LexError(msg, line)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(source: str) -> list[Token]:
    """Lex a PolarPandas source string into an ordered list of tokens.

    The final token is always ``Token(TokenType.EOF, "", <last_line>)``.
    Raises ``LexError`` on any unrecognised character or malformed token.

    Parameters
    ----------
    source : str
        Full PolarPandas source text (UTF-8 string).

    Returns
    -------
    list[Token]
        Flat token sequence including the terminal EOF token.

    Raises
    ------
    LexError
        On any lexical error.

    Example
    -------
    # >>> from polarpandas.lexer import tokenize
    # >>> tokens = tokenize('LOAD "data.csv" AS df')
    # >>> [(t.type.name, t.value) for t in tokens]
    [('KW', 'LOAD'), ('STRING', '"data.csv"'), ('KW', 'AS'),
     ('IDENT', 'df'), ('EOF', '')]
    """
    return _Lexer(source)._tokenize_all()