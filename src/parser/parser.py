"""
polarpandas.parser
==================
PolarPandas Translator — Component 2: Parser  (v2.0)

Public interface:
    parse(tokens: list[Token]) -> list[ASTNode]

Changelog:
    v1.0  Members A/B/C/D — initial implementation
    v1.1  Fix  — all statement syntaxes corrected to spec; meta-cond parser;
                 DOT removed
    v2.0  Simplification phase —
            AstPipeline node wraps df -> op -> op -> ... chains
            Pipeline syntax: df -> VERB [args...]  /  -> VERB [args...]
            Unquoted column names: IDENT accepted everywhere STRING was for cols
            Compressed GROUP: GROUP BY col SUM col  (AGGREGATE keyword optional)
            Shortened pipeline-mode statements:
              FILL col WITH value|method      (NULLS IN ... COLUMN implicit)
              DROP NULLS                       (FROM df implicit)
              DROP DUPLICATES [KEEP ...]       (FROM df implicit)
              FILTER WHERE cond                (df implicit)
              CAST COLUMN col TO dtype         (df implicit)
              RENAME COLUMN old TO new         (df implicit)
              SORT BY col [ASC|DESC]           (df implicit)
              SELECT COLUMNS col-list          (df implicit)
              GROUP BY col FUNC col            (df implicit)
              COUNT GROUP BY col               (df implicit)
              EXPORT TO file [FORMAT fmt]      (df implicit)
              PLOT TYPE ptype [opts]           (df implicit)
            All old verbose syntax still valid (backward compatible).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from lexer.lexer import Token, TokenType


# ---------------------------------------------------------------------------
# 1. PARSE ERROR
# ---------------------------------------------------------------------------

class ParseError(Exception):
    def __init__(self, message: str, line: int) -> None:
        super().__init__(f"[Parser] line {line}: {message}")
        self.message = message
        self.line    = line


# ---------------------------------------------------------------------------
# 2. EXPRESSION NODES
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExprLiteral:
    """Scalar literal: integer, float, string, or boolean.
    value: raw token value.  kind: 'integer'|'float'|'string'|'bool'
    """
    value : str
    kind  : str
    line  : int


@dataclass(frozen=True)
class ExprColRef:
    """Column reference inside an expression.
    name       : column name (bare or from quoted string, quotes stripped)
    is_loopvar : True if sourced from $col LOOPVAR token
    """
    name       : str
    is_loopvar : bool
    line       : int


@dataclass(frozen=True)
class ExprBinop:
    """Binary arithmetic: left OP right.  op: '+' '-' '*' '/' '%'"""
    op    : str
    left  : "Expr"
    right : "Expr"
    line  : int


Expr = Union[ExprLiteral, ExprColRef, ExprBinop]


# ---------------------------------------------------------------------------
# 3. CONDITION NODES
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CondCompare:
    left  : Expr
    op    : str
    right : Expr
    line  : int

@dataclass(frozen=True)
class CondIn:
    col    : ExprColRef
    values : list
    negate : bool
    line   : int

@dataclass(frozen=True)
class CondNull:
    col    : ExprColRef
    is_not : bool
    line   : int

@dataclass(frozen=True)
class CondBetween:
    col  : ExprColRef
    low  : ExprLiteral
    high : ExprLiteral
    line : int

@dataclass(frozen=True)
class CondAnd:
    left  : "Condition"
    right : "Condition"
    line  : int

@dataclass(frozen=True)
class CondOr:
    left  : "Condition"
    right : "Condition"
    line  : int

Condition = Union[CondCompare, CondIn, CondNull, CondBetween, CondAnd, CondOr]


# ---------------------------------------------------------------------------
# 4. STATEMENT AST NODES
# ---------------------------------------------------------------------------

# -- I/O -------------------------------------------------------------------

@dataclass(frozen=True)
class AstLoad:
    """LOAD "<file>" AS <ident> [ENGINE pandas|polars]"""
    file   : str
    name   : str
    engine : Optional[str]
    line   : int

@dataclass(frozen=True)
class AstExport:
    """EXPORT [<ident>] TO "<file>" [FORMAT fmt]
    name is None in pipeline context (subject is implicit).
    """
    name   : Optional[str]
    file   : str
    format : Optional[str]
    line   : int

# -- Inspection ------------------------------------------------------------

@dataclass(frozen=True)
class AstPreview:
    name : Optional[str]
    rows : int
    line : int

@dataclass(frozen=True)
class AstInfo:
    name : Optional[str]
    line : int

@dataclass(frozen=True)
class AstDescribe:
    name : Optional[str]
    line : int

@dataclass(frozen=True)
class AstSetEngine:
    engine : str
    line   : int

# -- Selection and Filtering -----------------------------------------------

@dataclass(frozen=True)
class AstSelect:
    """SELECT [<ident>] COLUMNS <col-list> | *
    columns = [] means wildcard (*).
    """
    name    : Optional[str]
    columns : list
    line    : int

@dataclass(frozen=True)
class AstFilter:
    """FILTER [<ident>] WHERE <condition>"""
    name      : Optional[str]
    condition : Condition
    line      : int

# -- Cleaning --------------------------------------------------------------

@dataclass(frozen=True)
class AstDropNulls:
    """DROP NULLS [FROM <ident>] [IN COLUMNS <col-list>]"""
    name    : Optional[str]
    columns : list
    line    : int

@dataclass(frozen=True)
class AstDropDups:
    """DROP DUPLICATES [FROM <ident>] [KEEP first|last|none]"""
    name : Optional[str]
    keep : str
    line : int

@dataclass(frozen=True)
class AstDropCol:
    """DROP COLUMN "<col>" [FROM <ident>]"""
    name   : Optional[str]
    column : str
    line   : int

@dataclass(frozen=True)
class AstFillNulls:
    """FILL [NULLS IN <ident>] COLUMN? <col> WITH <value|method>
    Long form:  FILL NULLS IN df COLUMN "age" WITH mean
    Short form: FILL age WITH mean            (pipeline context)
    """
    name   : Optional[str]
    column : str
    fill   : Expr
    line   : int

@dataclass(frozen=True)
class AstCast:
    """CAST [<ident>] COLUMN <col> TO <dtype>"""
    name   : Optional[str]
    column : str
    dtype  : str
    line   : int

# -- Transformation --------------------------------------------------------

@dataclass(frozen=True)
class AstAddCol:
    """ADD COLUMN <col> TO [<ident>] AS <expr>"""
    name   : Optional[str]
    column : str
    expr   : Expr
    line   : int

@dataclass(frozen=True)
class AstRename:
    """RENAME [<ident>] COLUMN <old> TO <new>"""
    name : Optional[str]
    old  : str
    new  : str
    line : int

@dataclass(frozen=True)
class AstSort:
    """SORT [<ident>] BY <col> [ASC|DESC]"""
    name      : Optional[str]
    column    : str
    direction : str
    line      : int

# -- Aggregation and Join --------------------------------------------------

@dataclass(frozen=True)
class AstGroup:
    """GROUP [<ident>] BY <col-or-list> AGGREGATE "<col>" AS <agg>
       or short: GROUP [<ident>] BY <col> <agg> <col>
    """
    name   : Optional[str]
    by     : list
    column : str
    agg    : str
    line   : int

@dataclass(frozen=True)
class AstCount:
    """COUNT [<ident>] GROUP BY <col-or-list>"""
    name : Optional[str]
    by   : list
    line : int

@dataclass(frozen=True)
class AstJoin:
    """JOIN <ident> WITH <ident> ON <col> [TYPE <jtype>] [AS <ident>]"""
    left   : str
    right  : str
    on     : str
    how    : str
    result : Optional[str]
    line   : int

# -- Visualisation ---------------------------------------------------------

@dataclass(frozen=True)
class AstPlot:
    """PLOT [<ident>] TYPE <ptype> <plot-opt>*"""
    name  : Optional[str]
    kind  : str
    x     : Optional[str]
    y     : Optional[str]
    hue   : Optional[str]
    color : Optional[str]
    title : Optional[str]
    xlabel: Optional[str]
    ylabel: Optional[str]
    bins  : Optional[int]
    save  : Optional[str]
    column: Optional[str]
    line  : int

# -- Control Flow ----------------------------------------------------------

@dataclass(frozen=True)
class AstIf:
    """IF <meta-cond> THEN <body> [ELSE <body>] END"""
    condition : Condition
    then_body : list
    else_body : list
    line      : int

@dataclass(frozen=True)
class AstFor:
    """FOR EACH $<var> OVER <col-list> DO <body> END"""
    var     : str
    columns : list
    body    : list
    line    : int

# -- Pipeline (v2.0) -------------------------------------------------------

@dataclass(frozen=True)
class AstPipeline:
    """A chained sequence of operations on a single dataframe.

    df -> FILTER WHERE amount > 0
       -> SORT BY age DESC
       -> EXPORT TO "out.csv"

    Attributes
    ----------
    name  : str             The source dataframe name.
    steps : list[ASTNode]   Each step has name=None (subject is implicit).
    line  : int             Line of the opening 'df ->' expression.
    """
    name  : str
    steps : list
    line  : int


ASTNode = Union[
    AstLoad, AstExport, AstPreview, AstInfo, AstDescribe, AstSetEngine,
    AstSelect, AstFilter,
    AstDropNulls, AstDropDups, AstDropCol, AstFillNulls, AstCast,
    AstAddCol, AstRename, AstSort,
    AstGroup, AstCount, AstJoin, AstPlot,
    AstIf, AstFor,
    AstPipeline,
]


# ---------------------------------------------------------------------------
# 5. PARSER CLASS
# ---------------------------------------------------------------------------

class _Parser:

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens: list[Token] = tokens
        self._pos:    int         = 0

    # -----------------------------------------------------------------------
    # Cursor primitives
    # -----------------------------------------------------------------------

    def peek(self, offset: int = 0) -> Token:
        idx = self._pos + offset
        return self._tokens[idx] if idx < len(self._tokens) else self._tokens[-1]

    def consume(self) -> Token:
        tok = self.peek()
        if tok.type is TokenType.EOF:
            raise ParseError("Unexpected end of input", tok.line)
        self._pos += 1
        return tok

    def at_end(self) -> bool:
        return self.peek().type is TokenType.EOF

    # -----------------------------------------------------------------------
    # Typed consume helpers
    # -----------------------------------------------------------------------

    def expect_kw(self, *keywords: str) -> Token:
        tok = self.peek()
        upper_set = {kw.upper() for kw in keywords}
        if tok.type is TokenType.KW and tok.value in upper_set:
            return self.consume()
        expected = " or ".join(f"'{k}'" for k in keywords)
        raise ParseError(
            f"Expected keyword {expected}, got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    def expect_ident(self) -> Token:
        tok = self.peek()
        if tok.type is TokenType.IDENT:
            return self.consume()
        raise ParseError(
            f"Expected identifier, got {tok.type.name} {tok.value!r}", tok.line)

    def expect_string(self) -> Token:
        tok = self.peek()
        if tok.type is TokenType.STRING:
            return self.consume()
        raise ParseError(
            f"Expected string literal, got {tok.type.name} {tok.value!r}", tok.line)

    def expect_int(self) -> Token:
        tok = self.peek()
        if tok.type is TokenType.INTEGER:
            return self.consume()
        raise ParseError(
            f"Expected integer, got {tok.type.name} {tok.value!r}", tok.line)

    def expect_col_name(self) -> str:
        """Accept a column name as either a quoted string or a bare identifier.

        This is the v2.0 helper that replaces expect_string() in all
        column-name positions, enabling unquoted column names throughout.

        Returns the column name as a plain string (quotes stripped if present).
        """
        tok = self.peek()
        if tok.type is TokenType.STRING:
            return self._strip_quotes(self.consume().value)
        if tok.type is TokenType.IDENT:
            return self.consume().value
        if tok.type is TokenType.LOOPVAR:
            raw = self.consume().value
            return raw[1:] if raw.startswith("$") else raw
        raise ParseError(
            f"Expected column name (quoted or unquoted), "
            f"got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    # -----------------------------------------------------------------------
    # Lookahead predicates
    # -----------------------------------------------------------------------

    def next_is_kw(self, *keywords: str) -> bool:
        tok = self.peek()
        return tok.type is TokenType.KW and tok.value in {kw.upper() for kw in keywords}

    def next_is(self, ttype: TokenType) -> bool:
        return self.peek().type is ttype

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _strip_quotes(s: str) -> str:
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            return s[1:-1]
        return s

    def parse_col_list(self) -> list:
        """Parse [col, col, ...] — each col may be quoted or unquoted."""
        self.expect_kw_or_punct(TokenType.LBRACKET, "[")
        cols: list[str] = []
        while not self.next_is(TokenType.RBRACKET):
            cols.append(self.expect_col_name())
            if self.next_is(TokenType.COMMA):
                self.consume()
            elif not self.next_is(TokenType.RBRACKET):
                raise ParseError(
                    f"Expected ',' or ']', got "
                    f"{self.peek().type.name} {self.peek().value!r}",
                    self.peek().line,
                )
        self.expect_kw_or_punct(TokenType.RBRACKET, "]")
        return cols

    def parse_value_list(self) -> list:
        self.expect_kw_or_punct(TokenType.LBRACKET, "[")
        values: list[ExprLiteral] = []
        while not self.next_is(TokenType.RBRACKET):
            values.append(self.parse_value())
            if self.next_is(TokenType.COMMA):
                self.consume()
            elif not self.next_is(TokenType.RBRACKET):
                raise ParseError(
                    f"Expected ',' or ']', got "
                    f"{self.peek().type.name} {self.peek().value!r}",
                    self.peek().line,
                )
        self.expect_kw_or_punct(TokenType.RBRACKET, "]")
        return values

    def parse_value(self) -> ExprLiteral:
        tok = self.peek()
        if tok.type is TokenType.STRING:
            return ExprLiteral(self.consume().value, "string", tok.line)
        if tok.type is TokenType.INTEGER:
            return ExprLiteral(self.consume().value, "integer", tok.line)
        if tok.type is TokenType.FLOAT:
            return ExprLiteral(self.consume().value, "float", tok.line)
        if tok.type is TokenType.BOOL:
            return ExprLiteral(self.consume().value, "bool", tok.line)
        raise ParseError(
            f"Expected literal value, got {tok.type.name} {tok.value!r}", tok.line)

    def expect_kw_or_punct(self, ttype: TokenType, display: str) -> Token:
        tok = self.peek()
        if tok.type is ttype:
            return self.consume()
        raise ParseError(
            f"Expected '{display}', got {tok.type.name} {tok.value!r}", tok.line)

    def parse_col_or_list(self) -> list:
        """Single col name OR [col, col, ...] → always returns list[str]."""
        if self.next_is(TokenType.LBRACKET):
            return self.parse_col_list()
        tok = self.peek()
        if tok.type in (TokenType.STRING, TokenType.IDENT):
            return [self.expect_col_name()]
        raise ParseError(
            f"Expected column name or column list, "
            f"got {tok.type.name} {tok.value!r}", tok.line)

    # -----------------------------------------------------------------------
    # Known aggregation function names (for compressed GROUP syntax)
    # -----------------------------------------------------------------------
    _AGG_FUNCS = {
        "sum", "mean", "median", "min", "max",
        "count", "std", "var", "first", "last",
    }

    def _next_is_agg_func(self) -> bool:
        """True if the next token is a bare IDENT that names an agg function."""
        tok = self.peek()
        return tok.type is TokenType.IDENT and tok.value.lower() in self._AGG_FUNCS

    # -----------------------------------------------------------------------
    # Optional name consumer
    # -----------------------------------------------------------------------

    def _maybe_consume_name(self) -> Optional[str]:
        """In standalone statements, consume the dataframe name (IDENT).
        In pipeline steps (name already known), returns None without consuming.
        Called only by standalone parsers that need to decide.
        """
        if self.next_is(TokenType.IDENT):
            return self.consume().value
        return None

    # -----------------------------------------------------------------------
    # Statement dispatcher
    # -----------------------------------------------------------------------

    def parse_statement(self, pipeline_subject: Optional[str] = None) -> ASTNode:
        """Parse one statement.

        Parameters
        ----------
        pipeline_subject : str | None
            If set, this statement is a pipeline step — the named dataframe
            is the implicit subject and statements that normally require an
            explicit df name will use this value instead.
        """
        tok = self.peek()

        # In pipeline mode, check for nested PIPE chains too
        if tok.type is TokenType.KW:
            kw = tok.value
            dispatch = {
                "LOAD":     lambda: self.parse_load(),
                "EXPORT":   lambda: self.parse_export(pipeline_subject),
                "PREVIEW":  lambda: self.parse_preview(pipeline_subject),
                "INFO":     lambda: self.parse_info(pipeline_subject),
                "DESCRIBE": lambda: self.parse_describe(pipeline_subject),
                "SET":      lambda: self.parse_set_engine(),
                "SELECT":   lambda: self.parse_select(pipeline_subject),
                "FILTER":   lambda: self.parse_filter(pipeline_subject),
                "DROP":     lambda: self.parse_drop(pipeline_subject),
                "FILL":     lambda: self.parse_fill_nulls(pipeline_subject),
                "CAST":     lambda: self.parse_cast(pipeline_subject),
                "ADD":      lambda: self.parse_add_col(pipeline_subject),
                "RENAME":   lambda: self.parse_rename(pipeline_subject),
                "SORT":     lambda: self.parse_sort(pipeline_subject),
                "GROUP":    lambda: self.parse_group(pipeline_subject),
                "COUNT":    lambda: self.parse_count(pipeline_subject),
                "JOIN":     lambda: self.parse_join(),
                "PLOT":     lambda: self.parse_plot(pipeline_subject),
                "IF":       lambda: self.parse_if(),
                "FOR":      lambda: self.parse_for(),
            }
            if kw in dispatch:
                return dispatch[kw]()
            raise ParseError(f"Unknown statement keyword {kw!r}", tok.line)

        # IDENT at top level — could be the start of a pipeline: df -> ...
        if tok.type is TokenType.IDENT and pipeline_subject is None:
            if self.peek(1).type is TokenType.PIPE:
                return self.parse_pipeline()

        raise ParseError(
            f"Expected statement keyword or pipeline expression, "
            f"got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    def _parse_all(self) -> list[ASTNode]:
        nodes: list[ASTNode] = []
        while not self.at_end():
            nodes.append(self.parse_statement())
        return nodes

    # -----------------------------------------------------------------------
    # Pipeline parser  (v2.0)
    # -----------------------------------------------------------------------

    def parse_pipeline(self) -> AstPipeline:
        """Parse:  df -> step -> step -> ...

        The first '->' may be on the same line or the next line.
        Continuation lines begin with '->' (PIPE token).

        Grammar:
            pipeline ::= IDENT PIPE pipeline-step (PIPE pipeline-step)*
            pipeline-step ::= statement parsed with implicit subject
        """
        name_tok = self.expect_ident()
        name     = name_tok.value
        line     = name_tok.line

        steps: list[ASTNode] = []

        # Consume first '->' and parse first step
        self.expect_kw_or_punct(TokenType.PIPE, "->")
        steps.append(self.parse_statement(pipeline_subject=name))

        # Consume any continuation steps that start with '->'
        while self.next_is(TokenType.PIPE):
            self.consume()  # '->'
            steps.append(self.parse_statement(pipeline_subject=name))

        return AstPipeline(name=name, steps=steps, line=line)

    # =======================================================================
    # STATEMENT PARSERS
    # =======================================================================

    # -----------------------------------------------------------------------
    # I/O
    # -----------------------------------------------------------------------

    def parse_load(self) -> AstLoad:
        """LOAD "<file>" AS <ident> [ENGINE pandas|polars]"""
        start  = self.expect_kw("LOAD")
        file   = self._strip_quotes(self.expect_string().value)
        self.expect_kw("AS")
        name   = self.expect_ident().value
        engine = None
        if self.next_is_kw("ENGINE"):
            self.consume()
            tok = self.peek()
            if tok.type in (TokenType.IDENT, TokenType.KW):
                engine = self.consume().value.lower()
            else:
                raise ParseError(
                    f"Expected engine name after ENGINE, "
                    f"got {tok.type.name} {tok.value!r}", tok.line)
        return AstLoad(file=file, name=name, engine=engine, line=start.line)

    def parse_export(self, subject: Optional[str] = None) -> AstExport:
        """EXPORT [<ident>] TO "<file>" [FORMAT fmt]

        Standalone: EXPORT df TO "out.csv"
        Pipeline:   -> EXPORT TO "out.csv"
        """
        start = self.expect_kw("EXPORT")
        # In standalone mode with no subject, consume df name
        if subject is None and self.next_is(TokenType.IDENT):
            subject = self.consume().value
        self.expect_kw("TO")
        file = self._strip_quotes(self.expect_string().value)
        fmt  = None
        if self.next_is_kw("FORMAT"):
            self.consume()
            tok = self.peek()
            if tok.type in (TokenType.IDENT, TokenType.KW):
                fmt = self.consume().value.lower()
            else:
                raise ParseError(
                    f"Expected format name, got {tok.type.name} {tok.value!r}", tok.line)
        return AstExport(name=subject, file=file, format=fmt, line=start.line)

    def parse_preview(self, subject: Optional[str] = None) -> AstPreview:
        """PREVIEW [<ident>] [ROWS <int>]"""
        start = self.expect_kw("PREVIEW")
        if subject is None and self.next_is(TokenType.IDENT):
            subject = self.consume().value
        rows = 5
        if self.next_is_kw("ROWS"):
            self.consume()
            rows = int(self.expect_int().value)
        return AstPreview(name=subject, rows=rows, line=start.line)

    def parse_info(self, subject: Optional[str] = None) -> AstInfo:
        start = self.expect_kw("INFO")
        if subject is None and self.next_is(TokenType.IDENT):
            subject = self.consume().value
        return AstInfo(name=subject, line=start.line)

    def parse_describe(self, subject: Optional[str] = None) -> AstDescribe:
        start = self.expect_kw("DESCRIBE")
        if subject is None and self.next_is(TokenType.IDENT):
            subject = self.consume().value
        return AstDescribe(name=subject, line=start.line)

    def parse_set_engine(self) -> AstSetEngine:
        start = self.expect_kw("SET")
        self.expect_kw("ENGINE")
        tok = self.peek()
        if tok.type in (TokenType.IDENT, TokenType.KW):
            return AstSetEngine(engine=self.consume().value.lower(), line=start.line)
        raise ParseError(
            f"Expected engine name, got {tok.type.name} {tok.value!r}", tok.line)

    # -----------------------------------------------------------------------
    # Selection and Filtering
    # -----------------------------------------------------------------------

    def parse_select(self, subject: Optional[str] = None) -> AstSelect:
        """SELECT [<ident>] COLUMNS <col-list> | *"""
        start = self.expect_kw("SELECT")
        if subject is None and self.next_is(TokenType.IDENT):
            subject = self.consume().value
        self.expect_kw("COLUMNS")
        if self.next_is(TokenType.STAR):
            self.consume()
            return AstSelect(name=subject, columns=[], line=start.line)
        return AstSelect(name=subject, columns=self.parse_col_list(), line=start.line)

    def parse_filter(self, subject: Optional[str] = None) -> AstFilter:
        """FILTER [<ident>] WHERE <condition>"""
        start = self.expect_kw("FILTER")
        if subject is None and self.next_is(TokenType.IDENT):
            subject = self.consume().value
        self.expect_kw("WHERE")
        return AstFilter(name=subject, condition=self.parse_condition(), line=start.line)

    # -----------------------------------------------------------------------
    # Cleaning
    # -----------------------------------------------------------------------

    def parse_drop(self, subject: Optional[str] = None) -> ASTNode:
        """DROP NULLS [FROM <ident>] [IN COLUMNS <col-list>]
           DROP DUPLICATES [FROM <ident>] [KEEP first|last|none]
           DROP COLUMN <col> [FROM <ident>]
        """
        start = self.expect_kw("DROP")
        tok   = self.peek()
        if tok.type is not TokenType.KW:
            raise ParseError(
                f"Expected NULLS, DUPLICATES, or COLUMN after DROP, "
                f"got {tok.type.name} {tok.value!r}", tok.line)
        variant = tok.value

        if variant == "NULLS":
            self.consume()
            # Optional FROM <ident> in standalone mode
            name = subject
            if name is None and self.next_is_kw("FROM"):
                self.consume()
                name = self.expect_ident().value
            columns: list[str] = []
            if self.next_is_kw("IN"):
                self.consume()
                self.expect_kw("COLUMNS")
                columns = self.parse_col_list()
            return AstDropNulls(name=name, columns=columns, line=start.line)

        if variant == "DUPLICATES":
            self.consume()
            name = subject
            if name is None and self.next_is_kw("FROM"):
                self.consume()
                name = self.expect_ident().value
            keep = "first"
            if self.next_is_kw("KEEP"):
                self.consume()
                tok2 = self.peek()
                if tok2.type in (TokenType.IDENT, TokenType.KW):
                    keep = self.consume().value.lower()
                else:
                    raise ParseError(
                        f"Expected first|last|none after KEEP, "
                        f"got {tok2.type.name} {tok2.value!r}", tok2.line)
            return AstDropDups(name=name, keep=keep, line=start.line)

        if variant == "COLUMN":
            self.consume()
            col = self.expect_col_name()
            name = subject
            if name is None and self.next_is_kw("FROM"):
                self.consume()
                name = self.expect_ident().value
            return AstDropCol(name=name, column=col, line=start.line)

        raise ParseError(
            f"Expected NULLS, DUPLICATES, or COLUMN after DROP, got {variant!r}",
            tok.line)

    def parse_fill_nulls(self, subject: Optional[str] = None) -> AstFillNulls:
        """Two forms:

        Long (standalone):
            FILL NULLS IN <ident> COLUMN <col> WITH <value|method>

        Short (pipeline):
            -> FILL <col> WITH <value|method>
        """
        start = self.expect_kw("FILL")

        # Detect form: if next token is NULLS → long form
        if self.next_is_kw("NULLS"):
            self.consume()   # NULLS
            self.expect_kw("IN")
            name = subject
            if name is None:
                name = self.expect_ident().value
            self.expect_kw("COLUMN")
            column = self.expect_col_name()
        else:
            # Short form: FILL <col> WITH ...
            name   = subject
            column = self.expect_col_name()

        self.expect_kw("WITH")
        fill = self._parse_fill_value()
        return AstFillNulls(name=name, column=column, fill=fill, line=start.line)

    def _parse_fill_value(self) -> Expr:
        tok = self.peek()
        if tok.type in (TokenType.INTEGER, TokenType.FLOAT,
                        TokenType.STRING, TokenType.BOOL):
            return self.parse_value()
        if tok.type is TokenType.IDENT:
            self.consume()
            return ExprColRef(name=tok.value, is_loopvar=False, line=tok.line)
        raise ParseError(
            f"Expected fill value or method name, "
            f"got {tok.type.name} {tok.value!r}", tok.line)

    def parse_cast(self, subject: Optional[str] = None) -> AstCast:
        """CAST [<ident>] COLUMN <col> TO <dtype>"""
        start = self.expect_kw("CAST")
        name  = subject
        if name is None and self.next_is(TokenType.IDENT):
            # Peek: if next-next is COLUMN keyword, this IDENT is the df name
            if self.peek(1).type is TokenType.KW and self.peek(1).value == "COLUMN":
                name = self.consume().value
        self.expect_kw("COLUMN")
        column = self.expect_col_name()
        self.expect_kw("TO")
        tok = self.peek()
        if tok.type in (TokenType.IDENT, TokenType.KW):
            dtype = self.consume().value.lower()
        else:
            raise ParseError(
                f"Expected dtype after TO, got {tok.type.name} {tok.value!r}", tok.line)
        return AstCast(name=name, column=column, dtype=dtype, line=start.line)

    # -----------------------------------------------------------------------
    # Transformation
    # -----------------------------------------------------------------------

    def parse_add_col(self, subject: Optional[str] = None) -> AstAddCol:
        """ADD COLUMN <col> TO [<ident>] AS <expr>"""
        start  = self.expect_kw("ADD")
        self.expect_kw("COLUMN")
        column = self.expect_col_name()
        self.expect_kw("TO")
        name   = subject
        if name is None and self.next_is(TokenType.IDENT):
            name = self.consume().value
        self.expect_kw("AS")
        expr = self.parse_expr()
        return AstAddCol(name=name, column=column, expr=expr, line=start.line)

    def parse_rename(self, subject: Optional[str] = None) -> AstRename:
        """RENAME [<ident>] COLUMN <old> TO <new>"""
        start = self.expect_kw("RENAME")
        name  = subject
        if name is None and self.next_is(TokenType.IDENT):
            if self.peek(1).type is TokenType.KW and self.peek(1).value == "COLUMN":
                name = self.consume().value
        self.expect_kw("COLUMN")
        old = self.expect_col_name()
        self.expect_kw("TO")
        new = self.expect_col_name()
        return AstRename(name=name, old=old, new=new, line=start.line)

    def parse_sort(self, subject: Optional[str] = None) -> AstSort:
        """SORT [<ident>] BY <col> [ASC|DESC]"""
        start  = self.expect_kw("SORT")
        name   = subject
        if name is None and self.next_is(TokenType.IDENT):
            name = self.consume().value
        self.expect_kw("BY")
        column    = self.expect_col_name()
        direction = "ASC"
        if self.next_is_kw("ASC"):
            self.consume(); direction = "ASC"
        elif self.next_is_kw("DESC"):
            self.consume(); direction = "DESC"
        return AstSort(name=name, column=column, direction=direction, line=start.line)

    # -----------------------------------------------------------------------
    # Aggregation and Join
    # -----------------------------------------------------------------------

    def parse_group(self, subject: Optional[str] = None) -> AstGroup:
        """Two forms:

        Long (spec):    GROUP [<ident>] BY <col-or-list> AGGREGATE <col> AS <agg>
        Short (v2.0):   GROUP [<ident>] BY <col-or-list> <agg> <col>

        Both forms supported; AGGREGATE keyword is optional in the short form.
        """
        start = self.expect_kw("GROUP")
        name  = subject
        if name is None and self.next_is(TokenType.IDENT):
            name = self.consume().value
        self.expect_kw("BY")
        by = self.parse_col_or_list()

        # Detect which form follows
        if self.next_is_kw("AGGREGATE"):
            # Long form: AGGREGATE <col> AS <agg>
            self.consume()
            column = self.expect_col_name()
            self.expect_kw("AS")
            tok = self.peek()
            if tok.type in (TokenType.IDENT, TokenType.KW):
                agg = self.consume().value.lower()
            else:
                raise ParseError(
                    f"Expected agg function name after AS, "
                    f"got {tok.type.name} {tok.value!r}", tok.line)
        elif self._next_is_agg_func():
            # Short form: <agg> <col>
            agg    = self.consume().value.lower()
            column = self.expect_col_name()
        else:
            raise ParseError(
                f"Expected AGGREGATE keyword or aggregation function name "
                f"(sum/mean/...) after GROUP BY <col>, "
                f"got {self.peek().type.name} {self.peek().value!r}",
                self.peek().line,
            )
        return AstGroup(name=name, by=by, column=column, agg=agg, line=start.line)

    def parse_count(self, subject: Optional[str] = None) -> AstCount:
        """COUNT [<ident>] GROUP BY <col-or-list>"""
        start = self.expect_kw("COUNT")
        name  = subject
        if name is None and self.next_is(TokenType.IDENT):
            name = self.consume().value
        self.expect_kw("GROUP")
        self.expect_kw("BY")
        by = self.parse_col_or_list()
        return AstCount(name=name, by=by, line=start.line)

    def parse_join(self) -> AstJoin:
        """JOIN <ident> WITH <ident> ON <col> [TYPE <jtype>] [AS <ident>]
        JOIN is never implicit — both operands must be named explicitly.
        """
        start  = self.expect_kw("JOIN")
        left   = self.expect_ident().value
        self.expect_kw("WITH")
        right  = self.expect_ident().value
        self.expect_kw("ON")
        on     = self.expect_col_name()
        how    = "inner"
        result = None
        if self.next_is_kw("TYPE"):
            self.consume()
            tok = self.peek()
            if tok.type in (TokenType.IDENT, TokenType.KW):
                how = self.consume().value.lower()
            else:
                raise ParseError(
                    f"Expected join type after TYPE, "
                    f"got {tok.type.name} {tok.value!r}", tok.line)
        if self.next_is_kw("AS"):
            self.consume()
            result = self.expect_ident().value
        return AstJoin(left=left, right=right, on=on, how=how,
                       result=result, line=start.line)

    # -----------------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------------

    def parse_plot(self, subject: Optional[str] = None) -> AstPlot:
        """PLOT [<ident>] TYPE <ptype> <plot-opt>*"""
        start = self.expect_kw("PLOT")
        name  = subject
        if name is None and self.next_is(TokenType.IDENT):
            name = self.consume().value
        self.expect_kw("TYPE")
        tok = self.peek()
        if tok.type in (TokenType.IDENT, TokenType.KW):
            kind = self.consume().value.lower()
        else:
            raise ParseError(
                f"Expected plot type, got {tok.type.name} {tok.value!r}", tok.line)

        x = y = hue = color = title = xlabel = ylabel = save = column = None
        bins: Optional[int] = None
        OPTS = {"X","Y","HUE","COLOR","TITLE","XLABEL","YLABEL",
                "BINS","FIGSIZE","SAVE","COLUMN"}
        while self.next_is_kw(*OPTS):
            kw = self.consume().value
            if kw == "X":      x      = self.expect_col_name()
            elif kw == "Y":    y      = self.expect_col_name()
            elif kw == "HUE":  hue    = self.expect_col_name()
            elif kw == "COLOR":color  = self._strip_quotes(self.expect_string().value)
            elif kw == "TITLE":title  = self._strip_quotes(self.expect_string().value)
            elif kw == "XLABEL":xlabel= self._strip_quotes(self.expect_string().value)
            elif kw == "YLABEL":ylabel= self._strip_quotes(self.expect_string().value)
            elif kw == "BINS": bins   = int(self.expect_int().value)
            elif kw == "FIGSIZE":
                self._consume_float(); self._consume_float()
            elif kw == "SAVE": save   = self._strip_quotes(self.expect_string().value)
            elif kw == "COLUMN": column = self.expect_col_name()
        return AstPlot(name=name, kind=kind, x=x, y=y, hue=hue, color=color,
                       title=title, xlabel=xlabel, ylabel=ylabel, bins=bins,
                       save=save, column=column, line=start.line)

    def _consume_float(self) -> float:
        tok = self.peek()
        if tok.type is TokenType.FLOAT:
            return float(self.consume().value)
        if tok.type is TokenType.INTEGER:
            return float(self.consume().value)
        raise ParseError(
            f"Expected float value, got {tok.type.name} {tok.value!r}", tok.line)

    # -----------------------------------------------------------------------
    # Expressions
    # -----------------------------------------------------------------------

    def parse_expr(self) -> Expr:
        return self._parse_additive()

    def _parse_additive(self) -> Expr:
        left = self._parse_multiplicative()
        while self.next_is(TokenType.ARITH_OP) and self.peek().value in ("+", "-"):
            op    = self.consume()
            right = self._parse_multiplicative()
            left  = ExprBinop(op=op.value, left=left, right=right, line=op.line)
        return left

    def _parse_multiplicative(self) -> Expr:
        left = self._parse_atom()
        while True:
            tok = self.peek()
            if tok.type is TokenType.ARITH_OP and tok.value in ("/", "%"):
                op    = self.consume()
                left  = ExprBinop(op=op.value, left=left,
                                  right=self._parse_atom(), line=op.line)
            elif tok.type is TokenType.STAR:
                op    = self.consume()
                left  = ExprBinop(op="*", left=left,
                                  right=self._parse_atom(), line=op.line)
            else:
                break
        return left

    def _parse_atom(self) -> Expr:
        tok = self.peek()
        if tok.type is TokenType.INTEGER:
            self.consume(); return ExprLiteral(tok.value, "integer", tok.line)
        if tok.type is TokenType.FLOAT:
            self.consume(); return ExprLiteral(tok.value, "float", tok.line)
        if tok.type is TokenType.STRING:
            self.consume(); return ExprLiteral(tok.value, "string", tok.line)
        if tok.type is TokenType.BOOL:
            self.consume(); return ExprLiteral(tok.value, "bool", tok.line)
        if tok.type is TokenType.IDENT:
            self.consume()
            return ExprColRef(name=tok.value, is_loopvar=False, line=tok.line)
        if tok.type is TokenType.LOOPVAR:
            lv = self.consume()
            name = lv.value[1:] if lv.value.startswith("$") else lv.value
            return ExprColRef(name=name, is_loopvar=True, line=lv.line)
        raise ParseError(
            f"Expected expression atom, got {tok.type.name} {tok.value!r}", tok.line)

    # -----------------------------------------------------------------------
    # Conditions
    # -----------------------------------------------------------------------

    def parse_condition(self) -> Condition:
        return self._parse_cond_or()

    def _parse_cond_or(self) -> Condition:
        left = self._parse_cond_and()
        while self.next_is_kw("OR"):
            tok   = self.consume()
            right = self._parse_cond_and()
            left  = CondOr(left=left, right=right, line=tok.line)
        return left

    def _parse_cond_and(self) -> Condition:
        left = self._parse_cond_atom()
        while self.next_is_kw("AND"):
            tok   = self.consume()
            right = self._parse_cond_atom()
            left  = CondAnd(left=left, right=right, line=tok.line)
        return left

    def _parse_cond_atom(self) -> Condition:
        line  = self.peek().line
        left  = self._parse_atom()
        tok   = self.peek()

        if tok.type is TokenType.OP:
            op    = self.consume().value
            right = self._parse_atom()
            return CondCompare(left=left, op=op, right=right, line=line)

        if self.next_is_kw("NOT"):
            not_tok = self.consume()
            self.expect_kw("IN")
            if not isinstance(left, ExprColRef):
                raise ParseError("NOT IN requires a column reference", line)
            return CondIn(col=left, values=self.parse_value_list(),
                          negate=True, line=not_tok.line)

        if self.next_is_kw("IN"):
            self.consume()
            if not isinstance(left, ExprColRef):
                raise ParseError("IN requires a column reference", line)
            return CondIn(col=left, values=self.parse_value_list(),
                          negate=False, line=line)

        if self.next_is_kw("IS"):
            self.consume()
            is_not = False
            if self.next_is_kw("NOT"):
                self.consume(); is_not = True
            self.expect_kw("NULL")
            if not isinstance(left, ExprColRef):
                raise ParseError("IS NULL requires a column reference", line)
            return CondNull(col=left, is_not=is_not, line=line)

        if self.next_is_kw("BETWEEN"):
            self.consume()
            low  = self.parse_value()
            self.expect_kw("AND")
            high = self.parse_value()
            if not isinstance(left, ExprColRef):
                raise ParseError("BETWEEN requires a column reference", line)
            return CondBetween(col=left, low=low, high=high, line=line)

        raise ParseError(
            f"Expected comparison operator, IN, IS, or BETWEEN, "
            f"got {tok.type.name} {tok.value!r}", tok.line)

    # -----------------------------------------------------------------------
    # Meta-conditions (IF guards)
    # -----------------------------------------------------------------------

    def parse_meta_cond(self) -> Condition:
        return self._parse_meta_or()

    def _parse_meta_or(self) -> Condition:
        left = self._parse_meta_and()
        while self.next_is_kw("OR"):
            tok = self.consume()
            left = CondOr(left=left, right=self._parse_meta_and(), line=tok.line)
        return left

    def _parse_meta_and(self) -> Condition:
        left = self._parse_meta_atom()
        while self.next_is_kw("AND"):
            tok = self.consume()
            left = CondAnd(left=left, right=self._parse_meta_atom(), line=tok.line)
        return left

    def _parse_meta_atom(self) -> Condition:
        line = self.peek().line

        if self.next_is_kw("EXISTS"):
            self.consume()
            if self.next_is_kw("COLUMN"):
                self.consume()
                col = self.expect_col_name()
                self.expect_kw("IN")
                df  = self.expect_ident().value
                return CondCompare(
                    left  = ExprColRef(name=col, is_loopvar=False, line=line),
                    op    = "exists_col",
                    right = ExprLiteral(value="true", kind="bool", line=line),
                    line  = line,
                )
            name = self.expect_ident().value
            return CondCompare(
                left  = ExprColRef(name=name, is_loopvar=False, line=line),
                op    = "exists",
                right = ExprLiteral(value="true", kind="bool", line=line),
                line  = line,
            )

        if self.next_is_kw("ROWCOUNT"):
            self.consume()
            name = self.expect_ident().value
            tok  = self.peek()
            if tok.type is not TokenType.OP:
                raise ParseError(
                    f"Expected comparison operator after ROWCOUNT, "
                    f"got {tok.type.name} {tok.value!r}", tok.line)
            op = self.consume().value
            n  = self.expect_int().value
            return CondCompare(
                left  = ExprColRef(name=name, is_loopvar=False, line=line),
                op    = f"rowcount_{op}",
                right = ExprLiteral(value=n, kind="integer", line=line),
                line  = line,
            )

        if self.next_is_kw("NULLCOUNT"):
            self.consume()
            name = self.expect_ident().value
            self.expect_kw("COLUMN")
            col  = self.expect_col_name()
            tok  = self.peek()
            if tok.type is not TokenType.OP:
                raise ParseError(
                    f"Expected comparison operator after NULLCOUNT ... COLUMN <col>, "
                    f"got {tok.type.name} {tok.value!r}", tok.line)
            op = self.consume().value
            n  = self.expect_int().value
            return CondCompare(
                left  = ExprColRef(name=col, is_loopvar=False, line=line),
                op    = f"nullcount_{op}",
                right = ExprLiteral(value=n, kind="integer", line=line),
                line  = line,
            )

        raise ParseError(
            f"Expected EXISTS, ROWCOUNT, or NULLCOUNT in IF guard, "
            f"got {self.peek().type.name} {self.peek().value!r}",
            self.peek().line,
        )

    # -----------------------------------------------------------------------
    # Control Flow
    # -----------------------------------------------------------------------

    def parse_if(self) -> AstIf:
        """IF <meta-cond> THEN <body> [ELSE <body>] END"""
        start = self.expect_kw("IF")
        cond  = self.parse_meta_cond()
        self.expect_kw("THEN")
        then_body: list[ASTNode] = []
        while not self.next_is_kw("ELSE", "END"):
            if self.at_end():
                raise ParseError("Expected ELSE or END after THEN body", self.peek().line)
            then_body.append(self.parse_statement())
        else_body: list[ASTNode] = []
        if self.next_is_kw("ELSE"):
            self.consume()
            while not self.next_is_kw("END"):
                if self.at_end():
                    raise ParseError("Expected END after ELSE body", self.peek().line)
                else_body.append(self.parse_statement())
        self.expect_kw("END")
        return AstIf(condition=cond, then_body=then_body,
                     else_body=else_body, line=start.line)

    def parse_for(self) -> AstFor:
        """FOR EACH $<var> OVER <col-list> DO <body> END"""
        start = self.expect_kw("FOR")
        self.expect_kw("EACH")
        lv_tok = self.peek()
        if lv_tok.type is not TokenType.LOOPVAR:
            raise ParseError(
                f"Expected $identifier after EACH, "
                f"got {lv_tok.type.name} {lv_tok.value!r}", lv_tok.line)
        self.consume()
        var = lv_tok.value[1:] if lv_tok.value.startswith("$") else lv_tok.value
        self.expect_kw("OVER")
        columns = self.parse_col_list()
        self.expect_kw("DO")
        body: list[ASTNode] = []
        while not self.next_is_kw("END"):
            if self.at_end():
                raise ParseError("Expected END after FOR body", self.peek().line)
            body.append(self.parse_statement())
        self.expect_kw("END")
        return AstFor(var=var, columns=columns, body=body, line=start.line)


# ---------------------------------------------------------------------------
# 6. PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def parse(tokens: list[Token]) -> list[ASTNode]:
    """Parse a PolarPandas token stream into an ordered AST node list.

    Parameters
    ----------
    tokens : list[Token]   From lexer.tokenize(); must include EOF.

    Returns
    -------
    list[ASTNode]

    Raises
    ------
    ParseError

    Examples
    --------
    >>> parse(tokenize('LOAD "d.csv" AS df'))
    [AstLoad(file='d.csv', name='df', engine=None, line=1)]

    >>> parse(tokenize('GROUP df BY region SUM amount'))
    [AstGroup(name='df', by=['region'], column='amount', agg='sum', line=1)]

    >>> # Pipeline syntax:
    >>> nodes = parse(tokenize('df -> FILTER WHERE amount > 0 -> SORT BY age DESC'))
    >>> isinstance(nodes[0], AstPipeline)
    True
    >>> len(nodes[0].steps)
    2
    """
    return _Parser(tokens)._parse_all()