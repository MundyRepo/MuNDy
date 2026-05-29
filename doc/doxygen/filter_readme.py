#!/usr/bin/env python3

import sys
import re
from pathlib import Path

README_REF_REPLACEMENTS = {
    "mundy::aggregate": r'\ref mundy::aggregate "mundy::aggregate"',
    "mundy::minimize(...)": r'\ref MundyMathMinimize "mundy::minimize(...)"',
    "mundy::Hilbert": r'\ref MundyMathHilbert "mundy::Hilbert"',
    "mundy::zmort": r'\ref MundyMathZmort "mundy::zmort"',
    "mundy::distance": r'\ref MundyGeomDistance "mundy::distance"',
    "mundy::transform": r'\ref MundyGeomTransform "mundy::transform"',
    "mundy::randomize": r'\ref MundyGeomRandomize "mundy::randomize"',
    "mundy::periodicity": r'\ref MundyGeomPeriodicity "mundy::periodicity"',
    "Primitives": r'\ref MundyGeomPrimitives "Primitives"',
    "mundy::mesh::FieldViews": r'\ref MundyMeshFieldViews "mundy::mesh::FieldViews"',
    "mundy::mesh::Classes": r'\ref MundyMeshClasses "mundy::mesh::Classes"',
    "mundy::mesh::Aggregate": r'\ref mundy::mesh::Aggregate "mundy::mesh::Aggregate"',
    "mundy::mesh::NgpFieldBLAS": r'\ref MundyMeshNgpFieldBLAS "mundy::mesh::NgpFieldBLAS"',
    "mundy::mesh::NgpAccessorExpr": r'\ref MundyMeshNgpAccessorExpr "mundy::mesh::NgpAccessorExpr"',
}

SORTED_README_REF_REPLACEMENTS = sorted(
    README_REF_REPLACEMENTS.items(),
    key=lambda item: len(item[0]),
    reverse=True,
)

GENERATED_REF_BOLD_RE = re.compile(r"\*\*((?:\\ref [^*\n]+?)+)\*\*")


def replace_outside_inline_code(line: str) -> str:
    parts = line.split("`")
    for i in range(0, len(parts), 2):
        for source, replacement in SORTED_README_REF_REPLACEMENTS:
            parts[i] = parts[i].replace(source, replacement)
        parts[i] = GENERATED_REF_BOLD_RE.sub(r"<b>\1</b>", parts[i])
    return "`".join(parts)


def transform_readme(text: str) -> str:
    out = []
    in_fence = False

    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            continue

        if in_fence:
            out.append(line)
        else:
            out.append(replace_outside_inline_code(line))

    return "".join(out)


def main() -> int:
    if len(sys.argv) != 2:
        sys.stdout.write(sys.stdin.read())
        return 0

    path = Path(sys.argv[1])
    text = path.read_text()

    if path.name == "README.md":
        text = transform_readme(text)

    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
