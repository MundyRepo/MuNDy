import re
from pathlib import Path

project = "MuNDy"
author = "Bryce Palmer"

docs_dir = Path(__file__).parent.resolve()
repo_root = docs_dir.parent
root_doc = "index"

extensions = [
    "myst_parser",
    "breathe",
    "exhale",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "sphinx_rtd_theme"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

suppress_warnings = [
    "cpp",
    "duplicate_declaration.cpp",
    "ref.identifier",
    "toc",
    "toc.not_included",
]

primary_domain = "cpp"
highlight_language = "cpp"

breathe_projects = {
    "MuNDy": str(docs_dir / "xml"),
}
breathe_default_project = "MuNDy"

exhale_args = {
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "C++ API Reference",
    "doxygenStripFromPath": str(repo_root),
    "createTreeView": False,
    "exhaleExecutesDoxygen": False,
}

_DOXYGEN_DIRECTIVE_RE = re.compile(r"(?m)^\.\. doxygen[a-z]+::[^\n]*(?:\n   .*)*\n?")
_DETAILED_DESCRIPTION_RE = re.compile(
    r"(?ms)\nDetailed Description\n-+\n\n.*?(?=\n[A-Z][A-Za-z ]+\n-+\n|\Z)"
)


def _dedupe_toctree_entries(text):
    seen = set()
    lines = []
    for line in text.splitlines(keepends=True):
        entry = line.strip()
        if entry.endswith(".rst"):
            if entry in seen:
                continue
            seen.add(entry)
        lines.append(line)
    return "".join(lines)


def _sanitize_generated_exhale_api(app, env, docnames):
    """Keep Exhale labels/pages, but avoid brittle generated declaration markup."""
    api_dir = docs_dir / "api"
    if not api_dir.exists():
        return

    for path in api_dir.rglob("*.rst"):
        text = path.read_text()
        stripped = _DOXYGEN_DIRECTIVE_RE.sub("", text)
        stripped = _DETAILED_DESCRIPTION_RE.sub("", stripped)
        if stripped != text:
            path.write_text(stripped)

    for path in api_dir.rglob("*.rst.include"):
        text = path.read_text()
        stripped = _dedupe_toctree_entries(text)
        if stripped != text:
            path.write_text(stripped)


def setup(app):
    app.connect("env-before-read-docs", _sanitize_generated_exhale_api)
