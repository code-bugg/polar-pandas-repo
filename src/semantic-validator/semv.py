"""PolarPandas v5 — Semantic Validator.

Walks the AST produced by the parser and enforces constraints that are
correct structurally (syntax is fine) but wrong *semantically*:

  • Type compatibility  — e.g. CAST target dtype must be a known type
  • Arity / format     — e.g. GROUP BY aggregation must be a known agg
  • Logical coherence  — e.g. PLOT kind must be a supported chart type;
                          X/Y column requirements depend on the chart type
  • Variable scoping   — loop-variables ($col) may only be used inside a
                          FOR body; regular column names may not shadow them
  • Pipeline integrity — SAVE inside a pipeline is allowed; READ is not
  • Plot constraints   — certain chart types mandate X and/or Y

The validator does *not* check whether the referenced columns actually
exist in a dataset — that is a runtime concern because PolarPandas
scripts may build columns dynamically (via SET, GROUP BY, etc.) and can
operate on files whose schema is unknown at parse-time.  What it *does*
guarantee is that every instruction is self-consistent and complete.

Usage
-----
    from lexer import tokenize
    from parser import parse
    from semantic_validator import validate, ValidationError

    tokens  = tokenize(source)
    ast     = parse(tokens)
    errors  = validate(ast)          # returns list[ValidationError]
    # — or —
    validate(ast, raise_on_error=True)   # raises on first error
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import PurePath
from typing import Optional

# ---------------------------------------------------------------------------
# Import AST nodes from the parser.  Adjust the import path to match your
# project layout (e.g. "from polarpandas.parser import …").
# ---------------------------------------------------------------------------
from parser import (
    ASTNode,
    AstCast,
    AstDropColumns,
    AstDropDuplicates,
    AstDropEmpty,
    AstFillEmpty,
    AstFilter,
    AstFor,
    AstGroupBy,
    AstIf,
    AstInfo,
    AstKeep,
    AstMerge,
    AstPipeline,
    AstPlot,
    AstPreview,
    AstRead,
    AstRename,
    AstSave,
    AstSet,
    AstSort,
    CondBetween,
    CondCompare,
    CondContains,
    CondIn,
    CondNull,
    CondAnd,
    CondOr,
    Condition,
    ExprBinop,
    ExprColRef,
    ExprLiteral,
    Expr,
    MetaExists,
    MetaRowCount,
    MetaAnd,
    MetaOr,
    MetaCondition,
    OperationNode,
)


# ---------------------------------------------------------------------------
# Public error type
# ---------------------------------------------------------------------------

@dataclass
class ValidationError(Exception):
    """A semantic validation error.

    Inherits from Exception so it can be raised with ``raise_on_error=True``
    and caught with a standard ``except ValidationError`` clause.
    ``__post_init__`` populates ``Exception.args`` so tracebacks and
    ``str(exc)`` behave correctly.
    """
    message: str
    line: int

    def __post_init__(self) -> None:
        # Populate Exception.args so the exception repr is informative.
        super().__init__(self.message, self.line)

    def __str__(self) -> str:
        return f"[Semantic] line {self.line}: {self.message}"


# ---------------------------------------------------------------------------
# Allowed value sets  (single source of truth)
# ---------------------------------------------------------------------------

# CAST target types — the subset PolarPandas exposes
_VALID_DTYPES: set[str] = {"NUMBER", "TEXT", "DATE", "DECIMAL", "BOOLEAN"}

# GROUP BY aggregation functions
_VALID_AGGS: set[str] = {"SUM", "AVERAGE", "COUNT", "MAX", "MIN"}

# FILL EMPTY methods
_VALID_FILL_METHODS: set[str] = {"mean", "median", "mode", "ffill", "bfill"}

# Merge join strategies
_VALID_HOW: set[str] = {"INNER", "LEFT", "RIGHT", "OUTER"}

# Supported plot kinds and their X/Y requirements.
# True  = required
# False = optional
# None  = not applicable (must be absent — e.g. histogram only needs X)
_PLOT_KINDS: dict[str, dict[str, Optional[bool]]] = {
    "line":       {"x": True,  "y": True},
    "bar":        {"x": True,  "y": True},
    "scatter":    {"x": True,  "y": True},
    "hist":       {"x": True,  "y": None},   # Y is meaningless for hist
    "box":        {"x": False, "y": True},   # X is optional grouping column
    "pie":        {"x": True,  "y": False},  # X = category, Y = optional values
}

# ON ERROR modes in CAST
_VALID_ON_ERROR: set[str] = {"SKIP", "STOP"}


# ---------------------------------------------------------------------------
# Internal scoping context
# ---------------------------------------------------------------------------

@dataclass
class _Scope:
    """Holds dataframe variable visibility and current active dataframe."""
    known_datasets: set[str]
    active_dataset: Optional[str] = None


@dataclass
class _Context:
    """Carries validation state through recursive calls."""
    errors: list[ValidationError]
    # Names of loop variables currently in scope (without the leading "$")
    loop_vars: set[str]
    scope: _Scope
    # True when we are inside an OperationNode list (a pipeline body or a
    # standalone operation used as a top-level statement).
    in_pipeline: bool = False

    def error(self, msg: str, line: int) -> None:
        self.errors.append(ValidationError(message=msg, line=line))

    def enter_loop(self, var: str) -> "_Context":
        """Return a child context that includes the new loop variable."""
        return _Context(
            errors=self.errors,
            loop_vars=self.loop_vars | {var},
            scope=self.scope,
            in_pipeline=self.in_pipeline,
        )

    def enter_pipeline(self) -> "_Context":
        return _Context(
            errors=self.errors,
            loop_vars=self.loop_vars,
            scope=self.scope,
            in_pipeline=True,
        )

    def ensure_active_df(self, line: int, statement_name: str) -> bool:
        if self.scope.active_dataset is None:
            self.error(
                f"{statement_name} requires an active dataframe. "
                "Add READ '<file>' before this statement.",
                line,
            )
            return False
        return True


def _dataset_name_from_path(path: str) -> str:
    """Derive dataframe variable name from a READ path (spec section 3.2)."""
    stem = PurePath(path).stem
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", stem)
    if not sanitized:
        return "_"
    if sanitized[0].isdigit():
        return f"_{sanitized}"
    return sanitized


# ---------------------------------------------------------------------------
# Expression validators
# ---------------------------------------------------------------------------

def _validate_expr(expr: Expr, ctx: _Context) -> None:
    """Recursively validate an expression tree."""
    if isinstance(expr, ExprLiteral):
        return  # literals are always valid
    if isinstance(expr, ExprColRef):
        if expr.is_loopvar and expr.name not in ctx.loop_vars:
            ctx.error(
                f"Loop variable '${expr.name}' is used outside a FOR loop",
                expr.line,
            )
        return
    if isinstance(expr, ExprBinop):
        _validate_expr(expr.left, ctx)
        _validate_expr(expr.right, ctx)
        return
    ctx.error(f"Unknown expression type {type(expr).__name__}", 0)


def _validate_condition(cond: Condition, ctx: _Context) -> None:
    """Recursively validate a filter condition tree."""
    if isinstance(cond, CondCompare):
        _validate_expr(cond.left, ctx)
        _validate_expr(cond.right, ctx)
        return
    if isinstance(cond, CondIn):
        _validate_expr(cond.col, ctx)
        for v in cond.values:
            _validate_expr(v, ctx)
        if not cond.values:
            ctx.error("IN clause must have at least one value", cond.line)
        return
    if isinstance(cond, CondNull):
        _validate_expr(cond.col, ctx)
        return
    if isinstance(cond, CondBetween):
        _validate_expr(cond.col, ctx)
        _validate_expr(cond.low, ctx)
        _validate_expr(cond.high, ctx)
        # Warn if both bounds are numeric literals and low > high
        if (
            isinstance(cond.low, ExprLiteral)
            and isinstance(cond.high, ExprLiteral)
            and cond.low.kind in ("integer", "float")
            and cond.high.kind in ("integer", "float")
        ):
            try:
                lo, hi = float(cond.low.value), float(cond.high.value)
                if lo > hi:
                    ctx.error(
                        f"BETWEEN lower bound {lo} is greater than upper bound {hi}",
                        cond.line,
                    )
            except ValueError:
                pass
        return
    if isinstance(cond, CondContains):
        _validate_expr(cond.col, ctx)
        _validate_expr(cond.value, ctx)
        # CONTAINS only makes sense with a string value
        if isinstance(cond.value, ExprLiteral) and cond.value.kind != "string":
            ctx.error(
                "CONTAINS value should be a string literal",
                cond.line,
            )
        return
    if isinstance(cond, (CondAnd, CondOr)):
        _validate_condition(cond.left, ctx)
        _validate_condition(cond.right, ctx)
        return
    ctx.error(f"Unknown condition type {type(cond).__name__}", 0)


def _validate_meta_condition(mcond: MetaCondition, ctx: _Context) -> None:
    if isinstance(mcond, MetaExists):
        if not mcond.name:
            ctx.error("EXISTS requires a dataset name", mcond.line)
        return
    if isinstance(mcond, MetaRowCount):
        if mcond.value < 0:
            ctx.error(
                f"ROWCOUNT comparison value must be non-negative, got {mcond.value}",
                mcond.line,
            )
        if mcond.name not in ctx.scope.known_datasets:
            ctx.error(
                f"ROWCOUNT references unknown dataframe '{mcond.name}'. "
                "READ it before using ROWCOUNT.",
                mcond.line,
            )
        return
    if isinstance(mcond, (MetaAnd, MetaOr)):
        _validate_meta_condition(mcond.left, ctx)
        _validate_meta_condition(mcond.right, ctx)
        return
    ctx.error(f"Unknown meta-condition type {type(mcond).__name__}", 0)


# ---------------------------------------------------------------------------
# Operation-node validators
# ---------------------------------------------------------------------------

def _validate_operation(node: OperationNode, ctx: _Context) -> None:
    """Validate a single pipeline step / standalone operation."""

    # ---- READ ---------------------------------------------------------------
    # READ is not an OperationNode in the parser (it is a top-level ASTNode),
    # but guard anyway in case the caller dispatches incorrectly.
    if isinstance(node, AstRead):
        if ctx.in_pipeline:
            ctx.error("READ cannot appear inside a pipeline", node.line)
        if not node.path:
            ctx.error("READ requires a file path", node.line)
        return

    # ---- SAVE ---------------------------------------------------------------
    if isinstance(node, AstSave):
        if not node.path:
            ctx.error("SAVE requires a file path", node.line)
        return

    # ---- PREVIEW ------------------------------------------------------------
    if isinstance(node, AstPreview):
        if node.rows <= 0:
            ctx.error(
                f"PREVIEW row count must be a positive integer, got {node.rows}",
                node.line,
            )
        return

    # ---- INFO ---------------------------------------------------------------
    if isinstance(node, AstInfo):
        return  # nothing to validate

    # ---- DROP EMPTY ---------------------------------------------------------
    if isinstance(node, AstDropEmpty):
        # columns list may be empty (means "all columns") — that is fine
        return

    # ---- DROP DUPLICATES ----------------------------------------------------
    if isinstance(node, AstDropDuplicates):
        return

    # ---- DROP COLUMNS -------------------------------------------------------
    if isinstance(node, AstDropColumns):
        if not node.columns:
            ctx.error("DROP requires at least one column name", node.line)
        return

    # ---- FILL EMPTY ---------------------------------------------------------
    if isinstance(node, AstFillEmpty):
        if not node.column:
            ctx.error("FILL EMPTY requires a column name", node.line)
        _validate_expr(node.value, ctx)
        if isinstance(node.value, ExprColRef):
            if (not node.value.is_loopvar) and node.value.name.lower() not in _VALID_FILL_METHODS:
                ctx.error(
                    f"FILL EMPTY ... WITH expects a literal or one of "
                    f"{', '.join(sorted(_VALID_FILL_METHODS))}; got '{node.value.name}'",
                    node.line,
                )
        elif not isinstance(node.value, ExprLiteral):
            ctx.error(
                "FILL EMPTY ... WITH supports only a literal or a fill method",
                node.line,
            )
        return

    # ---- KEEP ---------------------------------------------------------------
    if isinstance(node, AstKeep):
        if not node.columns:
            ctx.error("KEEP requires at least one column name", node.line)
        return

    # ---- CAST ---------------------------------------------------------------
    if isinstance(node, AstCast):
        if not node.column:
            ctx.error("CAST requires a column name", node.line)
        if node.dtype not in _VALID_DTYPES:
            ctx.error(
                f"Unknown CAST target type '{node.dtype}'. "
                f"Valid types: {', '.join(sorted(_VALID_DTYPES))}",
                node.line,
            )
        if node.on_error is not None and node.on_error not in _VALID_ON_ERROR:
            ctx.error(
                f"Unknown ON ERROR mode '{node.on_error}'. "
                f"Valid modes: {', '.join(sorted(_VALID_ON_ERROR))}",
                node.line,
            )
        return

    # ---- SET ----------------------------------------------------------------
    if isinstance(node, AstSet):
        if not node.column:
            ctx.error("SET requires a column name", node.line)
        _validate_expr(node.expr, ctx)
        return

    # ---- RENAME -------------------------------------------------------------
    if isinstance(node, AstRename):
        if not node.old:
            ctx.error("RENAME requires a source column name", node.line)
        if not node.new:
            ctx.error("RENAME requires a target column name", node.line)
        if node.old == node.new:
            ctx.error(
                f"RENAME source and target are identical: '{node.old}'",
                node.line,
            )
        return

    # ---- SORT ---------------------------------------------------------------
    if isinstance(node, AstSort):
        if not node.column:
            ctx.error("SORT BY requires a column name", node.line)
        if node.direction not in ("ASC", "DESC"):
            ctx.error(
                f"SORT direction must be ASC or DESC, got '{node.direction}'",
                node.line,
            )
        return

    # ---- GROUP BY -----------------------------------------------------------
    if isinstance(node, AstGroupBy):
        if not node.by:
            ctx.error("GROUP BY requires at least one grouping column", node.line)
        if node.agg not in _VALID_AGGS:
            ctx.error(
                f"Unknown aggregation '{node.agg}'. "
                f"Valid aggregations: {', '.join(sorted(_VALID_AGGS))}",
                node.line,
            )
        if not node.column:
            ctx.error("GROUP BY requires a target column for aggregation", node.line)
        return

    # ---- MERGE --------------------------------------------------------------
    if isinstance(node, AstMerge):
        if not node.source:
            ctx.error("MERGE requires a source dataset or file path", node.line)
        if not node.source_is_file and node.source not in ctx.scope.known_datasets:
            ctx.error(
                f"MERGE source '{node.source}' is not a known dataframe. "
                "READ it first or use a file path.",
                node.line,
            )
        if not node.on:
            ctx.error("MERGE requires an ON column", node.line)
        if node.how not in _VALID_HOW:
            ctx.error(
                f"Unknown MERGE join type '{node.how}'. "
                f"Valid types: {', '.join(sorted(_VALID_HOW))}",
                node.line,
            )
        return

    # ---- FILTER WHERE -------------------------------------------------------
    if isinstance(node, AstFilter):
        _validate_condition(node.condition, ctx)
        return

    # ---- PLOT ---------------------------------------------------------------
    if isinstance(node, AstPlot):
        _validate_plot(node, ctx)
        return

    ctx.error(f"Unknown operation node type {type(node).__name__}", 0)


def _validate_plot(node: AstPlot, ctx: _Context) -> None:
    """Validate plot-specific constraints."""
    if not node.kind:
        ctx.error("PLOT requires a TYPE", node.line)
        return

    if node.kind not in _PLOT_KINDS:
        ctx.error(
            f"Unknown plot type '{node.kind}'. "
            f"Supported types: {', '.join(sorted(_PLOT_KINDS))}",
            node.line,
        )
        return

    reqs = _PLOT_KINDS[node.kind]

    # X axis
    x_req = reqs.get("x")
    if x_req is True and node.x is None:
        ctx.error(f"PLOT TYPE {node.kind} requires an X column", node.line)
    if x_req is None and node.x is not None:
        ctx.error(
            f"PLOT TYPE {node.kind} does not use an X column (remove X {node.x!r})",
            node.line,
        )

    # Y axis
    y_req = reqs.get("y")
    if y_req is True and node.y is None:
        ctx.error(f"PLOT TYPE {node.kind} requires a Y column", node.line)
    if y_req is None and node.y is not None:
        ctx.error(
            f"PLOT TYPE {node.kind} does not use a Y column (remove Y {node.y!r})",
            node.line,
        )


# ---------------------------------------------------------------------------
# Top-level statement validators
# ---------------------------------------------------------------------------

def _validate_statement(node: ASTNode, ctx: _Context) -> None:
    """Dispatch to the appropriate validator for each top-level AST node."""

    if isinstance(node, AstRead):
        if ctx.in_pipeline:
            ctx.error("READ cannot appear inside a pipeline", node.line)
        if not node.path:
            ctx.error("READ requires a file path", node.line)
            return
        dataset = _dataset_name_from_path(node.path)
        ctx.scope.known_datasets.add(dataset)
        ctx.scope.active_dataset = dataset
        return

    if isinstance(node, AstPipeline):
        _validate_pipeline(node, ctx)
        return

    if isinstance(node, AstIf):
        _validate_if(node, ctx)
        return

    if isinstance(node, AstFor):
        _validate_for(node, ctx)
        return

    if isinstance(node, AstPlot) and node.target is not None:
        if node.target not in ctx.scope.known_datasets:
            ctx.error(
                f"PLOT target '{node.target}' is not a known dataframe. "
                "READ it first.",
                node.line,
            )
    else:
        ctx.ensure_active_df(node.line, type(node).__name__.replace("Ast", "").upper())

    # Any remaining ASTNode is also an OperationNode used standalone.
    _validate_operation(node, ctx)  # type: ignore[arg-type]


def _validate_pipeline(node: AstPipeline, ctx: _Context) -> None:
    if node.target is not None:
        if node.target not in ctx.scope.known_datasets:
            ctx.error(
                f"Pipeline target '{node.target}' is not a known dataframe. "
                "READ it first.",
                node.line,
            )
        else:
            ctx.scope.active_dataset = node.target
    else:
        ctx.ensure_active_df(node.line, "PIPELINE")

    pipe_ctx = ctx.enter_pipeline()
    if not node.steps:
        ctx.error("Pipeline has no steps", node.line)
        return
    for step in node.steps:
        if isinstance(step, AstRead):
            ctx.error("READ cannot appear inside a pipeline", step.line)
            continue
        _validate_operation(step, pipe_ctx)


def _validate_if(node: AstIf, ctx: _Context) -> None:
    _validate_meta_condition(node.condition, ctx)
    for child in node.then_body:
        _validate_statement(child, ctx)
    for child in node.else_body:
        _validate_statement(child, ctx)


def _validate_for(node: AstFor, ctx: _Context) -> None:
    if not node.var:
        ctx.error("FOR loop requires a loop variable", node.line)
    if not node.columns:
        ctx.error("FOR loop requires at least one column in its IN list", node.line)

    # Check that the loop variable name doesn't shadow a column already
    # mentioned in the columns list (which would be confusing).
    if node.var in node.columns:
        ctx.error(
            f"FOR loop variable '${node.var}' shadows a column of the same name "
            f"in the IN list — rename either the variable or the column",
            node.line,
        )

    child_ctx = ctx.enter_loop(node.var)
    if not node.body:
        ctx.error("FOR loop body is empty", node.line)
    for child in node.body:
        _validate_statement(child, child_ctx)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate(
    ast: list[ASTNode],
    *,
    raise_on_error: bool = False,
) -> list[ValidationError]:
    """Validate a parsed PolarPandas v5 AST.

    Parameters
    ----------
    ast:
        The list of ASTNodes returned by ``parse()``.
    raise_on_error:
        If *True*, raise the first ``ValidationError`` as an exception
        instead of collecting and returning all errors.

    Returns
    -------
    list[ValidationError]
        All semantic errors found.  Empty list means the AST is valid.
    """
    ctx = _Context(
        errors=[],
        loop_vars=set(),
        scope=_Scope(known_datasets=set(), active_dataset=None),
    )
    for node in ast:
        _validate_statement(node, ctx)

    if raise_on_error and ctx.errors:
        first = ctx.errors[0]
        raise ValidationError(first.message, first.line)

    return ctx.errors
