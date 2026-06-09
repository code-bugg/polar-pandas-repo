"""PolarPandas command-line runner.

Usage:
	python src/main.py samples/tips_demo.pp
	python src/main.py samples/tips_demo.pp --output build/tips_demo.py
"""

from __future__ import annotations

import argparse
import sys
from importlib import util
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent


def _load_module(name: str, rel_path: str):
	spec = util.spec_from_file_location(name, SRC_DIR / rel_path)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Unable to load module from {rel_path}")
	module = util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def _compile_source(source: str) -> str:
	from lexer.lexer import tokenize
	from parser.parser import parse

	semv = _load_module("polar_pandas_semv", "semantic-validator/semv.py")
	cd = _load_module("polar_pandas_cd", "code-generator/cd.py")

	ast = parse(tokenize(source))
	errors = semv.validate(ast)
	if errors:
		message = "\n".join(str(err) for err in errors)
		raise ValueError(message)
	return cd.generate(ast)


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run a PolarPandas .pp script")
	parser.add_argument("script", help="Path to the .pp script")
	parser.add_argument(
		"-o",
		"--output",
		help="Optional path to write the generated Python source",
	)
	return parser


def main(argv: list[str] | None = None) -> int:
	args = _build_parser().parse_args(argv)
	script_path = Path(args.script)

	try:
		source = script_path.read_text(encoding="utf-8")
		generated = _compile_source(source)

		if args.output:
			output_path = Path(args.output)
			output_path.parent.mkdir(parents=True, exist_ok=True)
			output_path.write_text(generated, encoding="utf-8")

		exec(compile(generated, str(script_path), "exec"), {"__name__": "__main__"})
		return 0
	except FileNotFoundError as exc:
		print(f"[PolarPandas] file not found: {exc.filename}", file=sys.stderr)
		return 1
	except (SyntaxError, ValueError, Exception) as exc:
		print(str(exc), file=sys.stderr)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
