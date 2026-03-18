"""
polarpandas.parser
==================
PolarPandas Translator — Component 2: Parser
Member A subtask: AST dataclasses, Parser class infrastructure, and the
                  top-level parse() entry point.

Subtasks B, C, D add their parse_*() methods directly to the _Parser class
in this file. The public interface they must respect is:

    parse(tokens: list[Token]) -> list[ASTNode]

and the shared types defined here (all in this module):

    AST node dataclasses   (Member A)
    Condition nodes        (Member A)
    Expression nodes       (Member A)
    ParseError             (Member A)
    _Parser helpers        (Member A)

Member B adds: parse_load, parse_export, parse_preview, parse_info,
               parse_describe, parse_set_engine,
               parse_col_list, parse_value_list, parse_value

Member C adds: parse_drop, parse_fill_nulls, parse_cast, parse_add_col,
               parse_rename, parse_sort, parse_select,
               parse_expr, parse_expr_atom

Member D adds: parse_group, parse_count, parse_join, parse_plot,
               parse_plot_opts, parse_condition, parse_cond_chain,
               parse_meta_cond, parse_if, parse_for
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

# Import shared lexer types — the Parser operates on the token stream
# produced by Component 1.
from lexer.lexer import Token, TokenType


# ─────────────────────────────────────────────────────────────────────────────
# 1. PARSE ERROR
# ─────────────────────────────────────────────────────────────────────────────

class ParseError(Exception):
    """Raised when the parser encounters an unexpected token or structure.

    Attributes
    ----------
    message : str
        Human-readable description of what went wrong.
    line    : int
        1-based source line where the error occurred.
    """
    def __init__(self, message: str, line: int) -> None:
        super().__init__(f"[Parser] line {line}: {message}")
        self.message = message
        self.line    = line


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPRESSION NODES
#    These are the leaf and binary nodes used inside statement fields.
#    Member C owns parse_expr / parse_expr_atom which produce these.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExprLiteral:
    """A bare literal value: integer, float, string, or boolean.

    Attributes
    ----------
    value   : str   Raw token value (e.g. "42", "3.14", '"hello"', "TRUE")
    kind    : str   One of: "integer", "float", "string", "bool"
    line    : int
    """
    value : str
    kind  : str   # "integer" | "float" | "string" | "bool"
    line  : int


@dataclass(frozen=True)
class ExprColRef:
    """A column reference, either bare (``amount``) or qualified (``df.amount``).

    Attributes
    ----------
    name        : str   Column name (raw identifier text)
    df          : str | None   DataFrame name if a dot-access form was used
    is_loopvar  : bool  True if this reference came from a $ident LOOPVAR token
    line        : int
    """
    name       : str
    df         : Optional[str]
    is_loopvar : bool
    line       : int


@dataclass(frozen=True)
class ExprBinop:
    """A binary arithmetic expression: left OP right.

    Attributes
    ----------
    op    : str          Operator string: "+", "-", "*", "/", "%"
    left  : Expr
    right : Expr
    line  : int
    """
    op    : str
    left  : "Expr"
    right : "Expr"
    line  : int


# Type alias for any expression node
Expr = Union[ExprLiteral, ExprColRef, ExprBinop]


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONDITION NODES
#    Used in FILTER/WHERE clauses and IF guards.
#    Member D owns parse_condition / parse_cond_chain which produce these.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CondCompare:
    """A simple comparison: <expr> <op> <expr>.

    Examples: ``df.amount >= 100``, ``status == "active"``

    Attributes
    ----------
    left  : Expr
    op    : str    One of: "==", "!=", ">", "<", ">=", "<="
    right : Expr
    line  : int
    """
    left  : Expr
    op    : str
    right : Expr
    line  : int


@dataclass(frozen=True)
class CondIn:
    """Membership test: <col> IN [v1, v2, ...].

    Attributes
    ----------
    col    : ExprColRef
    values : list[ExprLiteral]
    line   : int
    """
    col    : ExprColRef
    values : list
    line   : int


@dataclass(frozen=True)
class CondNull:
    """Null check: <col> IS NULL | IS NOT NULL.

    Attributes
    ----------
    col    : ExprColRef
    is_not : bool   True → IS NOT NULL
    line   : int
    """
    col    : ExprColRef
    is_not : bool
    line   : int


@dataclass(frozen=True)
class CondBetween:
    """Range check: <col> BETWEEN <low> AND <high>.

    Attributes
    ----------
    col  : ExprColRef
    low  : ExprLiteral
    high : ExprLiteral
    line : int
    """
    col  : ExprColRef
    low  : ExprLiteral
    high : ExprLiteral
    line : int


@dataclass(frozen=True)
class CondAnd:
    """Logical conjunction: left AND right."""
    left  : "Condition"
    right : "Condition"
    line  : int


@dataclass(frozen=True)
class CondOr:
    """Logical disjunction: left OR right."""
    left  : "Condition"
    right : "Condition"
    line  : int


# Type alias for any condition node
Condition = Union[CondCompare, CondIn, CondNull, CondBetween, CondAnd, CondOr]


# ─────────────────────────────────────────────────────────────────────────────
# 4. STATEMENT AST NODES
#    One dataclass per PolarPandas statement type (22 total).
#    Fields use snake_case. Optional fields default to None.
#    All nodes are frozen (immutable after construction).
# ─────────────────────────────────────────────────────────────────────────────

# ── I/O and Inspection (Member B) ────────────────────────────────────────────

@dataclass(frozen=True)
class AstLoad:
    """LOAD <file> AS <name>

    Attributes
    ----------
    file   : str   Path/filename string (raw, including quotes stripped by parser)
    name   : str   Identifier to bind the loaded dataframe to
    line   : int
    """
    file : str
    name : str
    line : int


@dataclass(frozen=True)
class AstExport:
    """EXPORT <name> TO <file>

    Attributes
    ----------
    name   : str   Dataframe identifier to export
    file   : str   Destination file path
    line   : int
    """
    name : str
    file : str
    line : int


@dataclass(frozen=True)
class AstPreview:
    """PREVIEW <name> [ROWS <n>]

    Attributes
    ----------
    name  : str
    rows  : int   Number of rows; defaults to 5 if omitted
    line  : int
    """
    name : str
    rows : int
    line : int


@dataclass(frozen=True)
class AstInfo:
    """INFO <name>

    Attributes
    ----------
    name : str
    line : int
    """
    name : str
    line : int


@dataclass(frozen=True)
class AstDescribe:
    """DESCRIBE <name>

    Attributes
    ----------
    name : str
    line : int
    """
    name : str
    line : int


@dataclass(frozen=True)
class AstSetEngine:
    """SET ENGINE <engine>

    Attributes
    ----------
    engine : str   "pandas" or "polars"
    line   : int
    """
    engine : str
    line   : int


# ── Cleaning and Transformation (Member C) ───────────────────────────────────

@dataclass(frozen=True)
class AstSelect:
    """SELECT COLUMNS [col, ...] FROM <name>

    Attributes
    ----------
    name    : str          Source dataframe identifier
    columns : list[str]    Column names to keep; empty list means * (all)
    line    : int
    """
    name    : str
    columns : list
    line    : int


@dataclass(frozen=True)
class AstFilter:
    """FILTER <name> WHERE <condition>

    Attributes
    ----------
    name      : str
    condition : Condition
    line      : int
    """
    name      : str
    condition : Condition
    line      : int


@dataclass(frozen=True)
class AstDropNulls:
    """DROP NULLS FROM <name> [COLUMNS [col, ...]]

    Attributes
    ----------
    name    : str
    columns : list[str]   Empty → drop rows with any null
    line    : int
    """
    name    : str
    columns : list
    line    : int


@dataclass(frozen=True)
class AstDropDups:
    """DROP DUPLICATES FROM <name> [COLUMNS [col, ...]]

    Attributes
    ----------
    name    : str
    columns : list[str]   Empty → consider all columns
    line    : int
    """
    name    : str
    columns : list
    line    : int


@dataclass(frozen=True)
class AstDropCol:
    """DROP COLUMN <col> FROM <name>

    Attributes
    ----------
    name   : str   Dataframe identifier
    column : str   Column name to drop
    line   : int
    """
    name   : str
    column : str
    line   : int


@dataclass(frozen=True)
class AstFillNulls:
    """FILL NULLS IN <name> [COLUMNS [col, ...]] WITH <value|method>

    Attributes
    ----------
    name    : str
    columns : list[str]    Empty → fill all columns
    fill    : Expr         The fill value (literal) or method keyword
    line    : int
    """
    name    : str
    columns : list
    fill    : Expr
    line    : int


@dataclass(frozen=True)
class AstCast:
    """CAST <col> IN <name> TO <type>

    Attributes
    ----------
    name   : str   Dataframe identifier
    column : str   Column to cast
    to     : str   Target type string: "INT", "FLOAT", "STR", "BOOL", etc.
    line   : int
    """
    name   : str
    column : str
    to     : str
    line   : int


@dataclass(frozen=True)
class AstAddCol:
    """ADD COLUMN <col> TO <name> AS <expr>

    Attributes
    ----------
    name   : str
    column : str   New column name
    expr   : Expr  Expression defining the column values
    line   : int
    """
    name   : str
    column : str
    expr   : Expr
    line   : int


@dataclass(frozen=True)
class AstRename:
    """RENAME <old> TO <new> IN <name>

    Attributes
    ----------
    name   : str   Dataframe identifier
    old    : str   Old column name
    new    : str   New column name
    line   : int
    """
    name : str
    old  : str
    new  : str
    line : int


@dataclass(frozen=True)
class AstSort:
    """SORT <name> BY <col> [ASC|DESC]

    Attributes
    ----------
    name      : str
    column    : str
    direction : str   "ASC" or "DESC"; default "ASC"
    line      : int
    """
    name      : str
    column    : str
    direction : str
    line      : int


# ── Aggregation, Join, Plot (Member D) ───────────────────────────────────────

@dataclass(frozen=True)
class AstGroup:
    """GROUP <name> BY <col> USING <agg> ON <col> [AS <result>]

    Attributes
    ----------
    name   : str
    by     : str    Column to group by
    agg    : str    Aggregation function name
    on     : str    Column to aggregate
    result : str | None   Optional result dataframe name
    line   : int
    """
    name   : str
    by     : str
    agg    : str
    on     : str
    result : Optional[str]
    line   : int


@dataclass(frozen=True)
class AstCount:
    """COUNT ROWS IN <name> [AS <result>]

    Attributes
    ----------
    name   : str
    result : str | None
    line   : int
    """
    name   : str
    result : Optional[str]
    line   : int


@dataclass(frozen=True)
class AstJoin:
    """JOIN <left> WITH <right> ON <col> [<type>] [AS <result>]

    Attributes
    ----------
    left      : str
    right     : str
    on        : str    Join key column
    how       : str    Join type: "INNER", "LEFT", "RIGHT", "OUTER"; default "INNER"
    result    : str | None
    line      : int
    """
    left   : str
    right  : str
    on     : str
    how    : str
    result : Optional[str]
    line   : int


@dataclass(frozen=True)
class AstPlot:
    """PLOT <name> TYPE <type> X <col> Y <col> [TITLE <str>] [SAVE <file>]

    Attributes
    ----------
    name    : str
    kind    : str           Plot type: "BAR", "LINE", "SCATTER", "HIST", ...
    x       : str | None    X-axis column
    y       : str | None    Y-axis column
    title   : str | None
    save    : str | None    Output file path
    line    : int
    """
    name  : str
    kind  : str
    x     : Optional[str]
    y     : Optional[str]
    title : Optional[str]
    save  : Optional[str]
    line  : int


# ── Control Flow (Member D) ───────────────────────────────────────────────────

@dataclass(frozen=True)
class AstIf:
    """IF <meta_cond> THEN <body> [ELSE <else_body>] END

    Attributes
    ----------
    condition : Condition    The meta-condition guard expression
    then_body : list[ASTNode]
    else_body : list[ASTNode]   Empty list if no ELSE branch
    line      : int
    """
    condition : Condition
    then_body : list
    else_body : list
    line      : int


@dataclass(frozen=True)
class AstFor:
    """FOR EACH $var OVER [col, ...] DO <body> END

    Attributes
    ----------
    var     : str             Loop variable name (without leading $)
    columns : list[str]       Static column list to iterate over
    body    : list[ASTNode]
    line    : int
    """
    var     : str
    columns : list
    body    : list
    line    : int


# Type alias covering all statement node types
ASTNode = Union[
    AstLoad, AstExport, AstPreview, AstInfo, AstDescribe, AstSetEngine,
    AstSelect, AstFilter, AstDropNulls, AstDropDups, AstDropCol,
    AstFillNulls, AstCast, AstAddCol, AstRename, AstSort,
    AstGroup, AstCount, AstJoin, AstPlot,
    AstIf, AstFor,
]


# ─────────────────────────────────────────────────────────────────────────────
# 5. PARSER CLASS
#    Member A owns:
#      - __init__  (cursor initialisation)
#      - peek()    (non-consuming lookahead)
#      - consume() (consuming one token)
#      - expect_kw(), expect_ident(), expect_string(), expect_int()
#      - next_is_kw()
#      - parse_statement()  (top-level dispatcher)
#      - _parse_all()       (the main loop)
#
#    Members B, C, D add their parse_*() methods directly to this class.
# ─────────────────────────────────────────────────────────────────────────────

class _Parser:
    """Internal mutable parser state.

    The public entry point ``parse(tokens)`` constructs one ``_Parser``
    instance per call and returns its AST node list.
    """

    def __init__(self, tokens: list[Token]) -> None:
        # EOF is always the last token; we rely on this as a sentinel.
        self._tokens : list[Token] = tokens
        self._pos    : int         = 0

    # ── Cursor primitives ─────────────────────────────────────────────────

    def peek(self, offset: int = 0) -> Token:
        """Return the token at ``_pos + offset`` without consuming.

        Always safe: returns the EOF token when past the end of the stream.
        """
        idx = self._pos + offset
        if idx < len(self._tokens):
            return self._tokens[idx]
        # Return the EOF sentinel regardless of how far past the end we are
        return self._tokens[-1]

    def consume(self) -> Token:
        """Consume and return the current token, advancing the cursor.

        Raises
        ------
        ParseError
            If called when the current token is EOF.
        """
        tok = self.peek()
        if tok.type is TokenType.EOF:
            raise ParseError("Unexpected end of input", tok.line)
        self._pos += 1
        return tok

    def at_end(self) -> bool:
        """True when the next token is EOF."""
        return self.peek().type is TokenType.EOF

    # ── Typed consume helpers ─────────────────────────────────────────────

    def expect_kw(self, *keywords: str) -> Token:
        """Consume and return the next token if it is a KW matching one of
        ``keywords`` (comparison is case-insensitive against stored UPPER values).

        Raises
        ------
        ParseError
            If the next token is not the expected keyword.
        """
        tok = self.peek()
        upper_kws = [kw.upper() for kw in keywords]
        if tok.type is TokenType.KW and tok.value in upper_kws:
            return self.consume()
        expected = " or ".join(f"'{k}'" for k in upper_kws)
        raise ParseError(
            f"Expected keyword {expected}, got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    def expect_ident(self) -> Token:
        """Consume and return the next token if it is an IDENT.

        Raises
        ------
        ParseError
            If the next token is not an identifier.
        """
        tok = self.peek()
        if tok.type is TokenType.IDENT:
            return self.consume()
        raise ParseError(
            f"Expected identifier, got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    def expect_string(self) -> Token:
        """Consume and return the next token if it is a STRING.

        Raises
        ------
        ParseError
            If the next token is not a string literal.
        """
        tok = self.peek()
        if tok.type is TokenType.STRING:
            return self.consume()
        raise ParseError(
            f"Expected string literal, got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    def expect_int(self) -> Token:
        """Consume and return the next token if it is an INTEGER.

        Raises
        ------
        ParseError
            If the next token is not an integer literal.
        """
        tok = self.peek()
        if tok.type is TokenType.INTEGER:
            return self.consume()
        raise ParseError(
            f"Expected integer literal, got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    # ── Lookahead predicates ──────────────────────────────────────────────

    def next_is_kw(self, *keywords: str) -> bool:
        """Return True if the next token is a KW matching any of ``keywords``."""
        tok = self.peek()
        return tok.type is TokenType.KW and tok.value in {kw.upper() for kw in keywords}

    def next_is(self, ttype: TokenType) -> bool:
        """Return True if the next token has the given type."""
        return self.peek().type is ttype

    # ── Top-level statement dispatcher ───────────────────────────────────

    def parse_statement(self) -> ASTNode:
        """Dispatch to the correct parse_*() method based on the leading keyword.

        Raises
        ------
        ParseError
            If the leading token is not a recognised statement keyword.
        """
        tok = self.peek()

        if tok.type is not TokenType.KW:
            raise ParseError(
                f"Expected a statement keyword, got {tok.type.name} {tok.value!r}",
                tok.line,
            )

        kw = tok.value  # already UPPER-cased by the lexer

        # ── I/O and Inspection (Member B) ─────────────────────────────────
        if kw == "LOAD":
            return self.parse_load()
        if kw == "EXPORT":
            return self.parse_export()
        if kw == "PREVIEW":
            return self.parse_preview()
        if kw == "INFO":
            return self.parse_info()
        if kw == "DESCRIBE":
            return self.parse_describe()
        if kw == "SET":
            return self.parse_set_engine()

        # ── Cleaning and Transformation (Member C) ────────────────────────
        if kw == "SELECT":
            return self.parse_select()
        if kw == "FILTER":
            return self.parse_filter()
        if kw == "DROP":
            return self.parse_drop()
        if kw == "FILL":
            return self.parse_fill_nulls()
        if kw == "CAST":
            return self.parse_cast()
        if kw == "ADD":
            return self.parse_add_col()
        if kw == "RENAME":
            return self.parse_rename()
        if kw == "SORT":
            return self.parse_sort()

        # ── Aggregation, Join, Plot, Control Flow (Member D) ──────────────
        if kw == "GROUP":
            return self.parse_group()
        if kw == "COUNT":
            return self.parse_count()
        if kw == "JOIN":
            return self.parse_join()
        if kw == "PLOT":
            return self.parse_plot()
        if kw == "IF":
            return self.parse_if()
        if kw == "FOR":
            return self.parse_for()

        raise ParseError(f"Unknown statement keyword {kw!r}", tok.line)

    # ── Main parse loop ───────────────────────────────────────────────────

    def _parse_all(self) -> list[ASTNode]:
        """Parse all statements until EOF and return the AST node list."""
        nodes: list[ASTNode] = []
        while not self.at_end():
            nodes.append(self.parse_statement())
        return nodes

    # ── Stubs for Members B, C, D ─────────────────────────────────────────
    # Replace each stub with the real implementation when merging.
    # Do NOT call these in production until implemented.

    # Member B
    def parse_load(self)        -> AstLoad:
        start = self.expect_kw("LOAD")
        file_tok = self.expect_string()
        self.expect_kw("AS")
        name_tok = self.expect_ident()

        raw = file_tok.value
        file_value = raw[1:-1] if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"' else raw
        return AstLoad(file=file_value, name=name_tok.value, line=start.line)

    def parse_export(self)      -> AstExport:
        start = self.expect_kw("EXPORT")
        name_tok = self.expect_ident()
        self.expect_kw("TO")
        file_tok = self.expect_string()

        raw = file_tok.value
        file_value = raw[1:-1] if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"' else raw
        return AstExport(name=name_tok.value, file=file_value, line=start.line)

    def parse_preview(self)     -> AstPreview:
        start = self.expect_kw("PREVIEW")
        name_tok = self.expect_ident()
        rows = 5
        if self.next_is_kw("ROWS"):
            self.consume()
            rows_tok = self.expect_int()
            rows = int(rows_tok.value)
        return AstPreview(name=name_tok.value, rows=rows, line=start.line)

    def parse_info(self)        -> AstInfo:
        start = self.expect_kw("INFO")
        name_tok = self.expect_ident()
        return AstInfo(name=name_tok.value, line=start.line)

    def parse_describe(self)    -> AstDescribe:
        start = self.expect_kw("DESCRIBE")
        name_tok = self.expect_ident()
        return AstDescribe(name=name_tok.value, line=start.line)

    def parse_set_engine(self)  -> AstSetEngine:
        start = self.expect_kw("SET")
        self.expect_kw("ENGINE")
        tok = self.peek()
        if tok.type is TokenType.IDENT or (
            tok.type is TokenType.KW and tok.value in {"PANDAS", "POLARS"}
        ):
            self.consume()
            engine = tok.value.lower()
            return AstSetEngine(engine=engine, line=start.line)
        raise ParseError(
            f"Expected engine name, got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    def parse_col_list(self)    -> list:
        tok = self.peek()
        if tok.type is not TokenType.LBRACKET:
            raise ParseError(
                f"Expected '[' to start column list, got {tok.type.name} {tok.value!r}",
                tok.line,
            )
        self.consume()  # [

        if self.peek().type is TokenType.RBRACKET:
            tok = self.peek()
            raise ParseError("Expected column name, got ']'", tok.line)

        cols: list[str] = []
        while True:
            tok = self.peek()
            if tok.type is TokenType.IDENT:
                cols.append(self.consume().value)
            elif tok.type is TokenType.STRING:
                raw = self.consume().value
                cols.append(raw[1:-1] if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"' else raw)
            else:
                raise ParseError(
                    f"Expected column name, got {tok.type.name} {tok.value!r}",
                    tok.line,
                )

            tok = self.peek()
            if tok.type is TokenType.COMMA:
                self.consume()
                continue
            if tok.type is TokenType.RBRACKET:
                self.consume()
                break
            raise ParseError(
                f"Expected ',' or ']', got {tok.type.name} {tok.value!r}",
                tok.line,
            )

        return cols

    def parse_value_list(self)  -> list:
        tok = self.peek()
        if tok.type is not TokenType.LBRACKET:
            raise ParseError(
                f"Expected '[' to start value list, got {tok.type.name} {tok.value!r}",
                tok.line,
            )
        self.consume()  # [

        if self.peek().type is TokenType.RBRACKET:
            tok = self.peek()
            raise ParseError("Expected value, got ']'", tok.line)

        values: list[ExprLiteral] = []
        while True:
            values.append(self.parse_value())
            tok = self.peek()
            if tok.type is TokenType.COMMA:
                self.consume()
                continue
            if tok.type is TokenType.RBRACKET:
                self.consume()
                break
            raise ParseError(
                f"Expected ',' or ']', got {tok.type.name} {tok.value!r}",
                tok.line,
            )
        return values

    def parse_value(self)       -> ExprLiteral:
        tok = self.peek()
        if tok.type is TokenType.STRING:
            self.consume()
            return ExprLiteral(value=tok.value, kind="string", line=tok.line)
        if tok.type is TokenType.INTEGER:
            self.consume()
            return ExprLiteral(value=tok.value, kind="integer", line=tok.line)
        if tok.type is TokenType.FLOAT:
            self.consume()
            return ExprLiteral(value=tok.value, kind="float", line=tok.line)
        if tok.type is TokenType.BOOL:
            self.consume()
            return ExprLiteral(value=tok.value, kind="bool", line=tok.line)
        raise ParseError(
            f"Expected literal value, got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    # Member C

    def parse_select(self) -> AstSelect:
        """SELECT COLUMNS [col, ...] FROM <name>
        
        Also supports: SELECT COLUMNS * FROM <name> (all columns)
        """
        start = self.expect_kw("SELECT")
        self.expect_kw("COLUMNS")
        
        columns: list[str] = []
        
        # Check for wildcard *
        if self.next_is(TokenType.STAR):
            self.consume()
            # Empty list means all columns
        else:
            columns = self.parse_col_list()
        
        self.expect_kw("FROM")
        name_tok = self.expect_ident()
        
        return AstSelect(name=name_tok.value, columns=columns, line=start.line)

    def parse_filter(self) -> AstFilter:
        """FILTER <name> WHERE <condition>
        
        Note: parse_condition is Member D, so this will call that method.
        """
        start = self.expect_kw("FILTER")
        name_tok = self.expect_ident()
        self.expect_kw("WHERE")
        condition = self.parse_condition()
        
        return AstFilter(name=name_tok.value, condition=condition, line=start.line)

    def parse_drop(self) -> ASTNode:
        """DROP NULLS FROM <name> [COLUMNS [col, ...]]
           DROP DUPLICATES FROM <name> [COLUMNS [col, ...]]
           DROP COLUMN <col> FROM <name>
        """
        start = self.expect_kw("DROP")
        
        # Determine which variant
        tok = self.peek()
        if tok.type is not TokenType.KW:
            raise ParseError(
                f"Expected keyword after DROP, got {tok.type.name} {tok.value!r}",
                tok.line,
            )
        
        variant = tok.value  # NULLS, DUPLICATES, or COLUMN
        
        if variant == "NULLS":
            self.consume()
            self.expect_kw("FROM")
            name_tok = self.expect_ident()
            
            columns: list[str] = []
            if self.next_is_kw("COLUMNS"):
                self.consume()
                columns = self.parse_col_list()
            
            return AstDropNulls(name=name_tok.value, columns=columns, line=start.line)
        
        elif variant == "DUPLICATES":
            self.consume()
            self.expect_kw("FROM")
            name_tok = self.expect_ident()
            
            columns: list[str] = []
            if self.next_is_kw("COLUMNS"):
                self.consume()
                columns = self.parse_col_list()
            
            return AstDropDups(name=name_tok.value, columns=columns, line=start.line)
        
        elif variant == "COLUMN":
            self.consume()
            col_tok = self.expect_ident()
            self.expect_kw("FROM")
            name_tok = self.expect_ident()
            
            return AstDropCol(name=name_tok.value, column=col_tok.value, line=start.line)
        
        else:
            raise ParseError(
                f"Expected NULLS, DUPLICATES, or COLUMN after DROP, got {tok.value!r}",
                tok.line,
            )

    def parse_fill_nulls(self) -> AstFillNulls:
        """FILL NULLS IN <name> [COLUMNS [col, ...]] WITH <value|method>"""
        start = self.expect_kw("FILL")
        self.expect_kw("NULLS")
        self.expect_kw("IN")
        name_tok = self.expect_ident()
        
        columns: list[str] = []
        if self.next_is_kw("COLUMNS"):
            self.consume()
            columns = self.parse_col_list()
        
        self.expect_kw("WITH")
        fill_value = self.parse_expr()
        
        return AstFillNulls(name=name_tok.value, columns=columns, fill=fill_value, line=start.line)

    def parse_cast(self) -> AstCast:
        """CAST <col> IN <name> TO <type>"""
        start = self.expect_kw("CAST")
        col_tok = self.expect_ident()
        self.expect_kw("IN")
        name_tok = self.expect_ident()
        self.expect_kw("TO")
        
        # Type should be an identifier or keyword (INT, FLOAT, STR, BOOL, etc.)
        type_tok = self.peek()
        if type_tok.type is TokenType.IDENT or type_tok.type is TokenType.KW:
            self.consume()
            type_str = type_tok.value
        else:
            raise ParseError(
                f"Expected type name, got {type_tok.type.name} {type_tok.value!r}",
                type_tok.line,
            )
        
        return AstCast(name=name_tok.value, column=col_tok.value, to=type_str, line=start.line)

    def parse_add_col(self) -> AstAddCol:
        """ADD COLUMN <col> TO <name> AS <expr>"""
        start = self.expect_kw("ADD")
        self.expect_kw("COLUMN")
        col_tok = self.expect_ident()
        self.expect_kw("TO")
        name_tok = self.expect_ident()
        self.expect_kw("AS")
        expr = self.parse_expr()
        
        return AstAddCol(name=name_tok.value, column=col_tok.value, expr=expr, line=start.line)

    def parse_rename(self) -> AstRename:
        """RENAME <old> TO <new> IN <name>"""
        start = self.expect_kw("RENAME")
        old_tok = self.expect_ident()
        self.expect_kw("TO")
        new_tok = self.expect_ident()
        self.expect_kw("IN")
        name_tok = self.expect_ident()
        
        return AstRename(name=name_tok.value, old=old_tok.value, new=new_tok.value, line=start.line)

    def parse_sort(self) -> AstSort:
        """SORT <name> BY <col> [ASC|DESC]"""
        start = self.expect_kw("SORT")
        name_tok = self.expect_ident()
        self.expect_kw("BY")
        col_tok = self.expect_ident()
        
        direction = "ASC"  # default
        if self.next_is_kw("ASC"):
            self.consume()
            direction = "ASC"
        elif self.next_is_kw("DESC"):
            self.consume()
            direction = "DESC"
        
        return AstSort(name=name_tok.value, column=col_tok.value, direction=direction, line=start.line)

    def parse_expr(self) -> Expr:
        """Parse an expression with binary operators.
        
        Uses recursive descent with proper precedence:
        - parse_expr: handles +, - (lowest precedence)
        - parse_expr_term: handles *, /, % (higher precedence)
        - parse_expr_atom: handles literals, column refs, parentheses
        """
        return self._parse_expr_additive()

    def _parse_expr_additive(self) -> Expr:
        """Parse addition/subtraction (lowest precedence)."""
        left = self._parse_expr_multiplicative()
        
        while self.next_is(TokenType.ARITH_OP):
            tok = self.peek()
            if tok.value not in {"+", "-"}:
                break
            op_tok = self.consume()
            right = self._parse_expr_multiplicative()
            left = ExprBinop(op=op_tok.value, left=left, right=right, line=op_tok.line)
        
        return left

    def _parse_expr_multiplicative(self) -> Expr:
        """Parse multiplication/division/modulo (higher precedence)."""
        left = self.parse_expr_atom()
        
        while self.next_is(TokenType.ARITH_OP) or self.next_is(TokenType.STAR):
            tok = self.peek()
            # For STAR, only treat as operator in expression context (not in SELECT COLUMNS *)
            if tok.type is TokenType.STAR:
                # In expression context, * is multiplication
                op_tok = self.consume()
                right = self.parse_expr_atom()
                left = ExprBinop(op="*", left=left, right=right, line=op_tok.line)
            elif tok.type is TokenType.ARITH_OP and tok.value in {"*", "/", "%"}:
                op_tok = self.consume()
                right = self.parse_expr_atom()
                left = ExprBinop(op=op_tok.value, left=left, right=right, line=op_tok.line)
            else:
                break
        
        return left

    def parse_expr_atom(self) -> Expr:
        """Parse atomic expressions: literals, column references, parentheses."""
        tok = self.peek()
        
        # Numeric literal
        if tok.type is TokenType.INTEGER:
            self.consume()
            return ExprLiteral(value=tok.value, kind="integer", line=tok.line)
        
        if tok.type is TokenType.FLOAT:
            self.consume()
            return ExprLiteral(value=tok.value, kind="float", line=tok.line)
        
        # String literal
        if tok.type is TokenType.STRING:
            self.consume()
            return ExprLiteral(value=tok.value, kind="string", line=tok.line)
        
        # Boolean literal
        if tok.type is TokenType.BOOL:
            self.consume()
            return ExprLiteral(value=tok.value, kind="bool", line=tok.line)
        
        # Column reference: ident or df.ident or $ident (loopvar)
        if tok.type is TokenType.IDENT:
            name_tok = self.consume()
            df = None
            
            # Check for df.column notation
            if self.next_is(TokenType.DOT):
                self.consume()
                col_tok = self.expect_ident()
                return ExprColRef(name=col_tok.value, df=name_tok.value, is_loopvar=False, line=name_tok.line)
            
            return ExprColRef(name=name_tok.value, df=None, is_loopvar=False, line=name_tok.line)
        
        # Loop variable: $ident
        if tok.type is TokenType.LOOPVAR:
            loopvar_tok = self.consume()
            # Remove leading $ from the value
            name = loopvar_tok.value[1:] if loopvar_tok.value.startswith("$") else loopvar_tok.value
            return ExprColRef(name=name, df=None, is_loopvar=True, line=loopvar_tok.line)
        
        # Parenthesized expression
        if tok.type is TokenType.LBRACKET:
            # In expression context, [ starts a parenthesized expression? Or is it just for lists?
            # For now, treat [ as an error in expression context
            raise ParseError(
                f"Unexpected '[' in expression, got {tok.type.name} {tok.value!r}",
                tok.line,
            )
        
        raise ParseError(
            f"Expected literal, column reference, or loop variable, got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    # Member D
    def parse_group(self)       -> AstGroup:       raise NotImplementedError("Member D: parse_group")
    def parse_count(self)       -> AstCount:       raise NotImplementedError("Member D: parse_count")
    def parse_join(self)        -> AstJoin:        raise NotImplementedError("Member D: parse_join")
    def parse_plot(self)        -> AstPlot:        raise NotImplementedError("Member D: parse_plot")
    def parse_plot_opts(self)   -> dict:           raise NotImplementedError("Member D: parse_plot_opts")
    def parse_condition(self)   -> Condition:      raise NotImplementedError("Member D: parse_condition")
    def parse_cond_chain(self)  -> Condition:      raise NotImplementedError("Member D: parse_cond_chain")
    def parse_meta_cond(self)   -> Condition:      raise NotImplementedError("Member D: parse_meta_cond")
    def parse_if(self)          -> AstIf:          raise NotImplementedError("Member D: parse_if")
    def parse_for(self)         -> AstFor:         raise NotImplementedError("Member D: parse_for")


# ─────────────────────────────────────────────────────────────────────────────
# 6. PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse(tokens: list[Token]) -> list[ASTNode]:
    """Parse a PolarPandas token stream into an ordered AST node list.

    Parameters
    ----------
    tokens : list[Token]
        The flat token sequence produced by ``lexer.tokenize()``.
        Must include the terminal EOF token.

    Returns
    -------
    list[ASTNode]
        One ASTNode per top-level statement in the source program.

    Raises
    ------
    ParseError
        On any structural or syntactic error.

    Example
    -------
    # >>> from lexer.lexer import tokenize
    # >>> from parser.parser import parse
    # >>> ast = parse(tokenize('LOAD "data.csv" AS df'))
    # >>> ast
    [AstLoad(file='data.csv', name='df', line=1)]
    """
    return _Parser(tokens)._parse_all()
