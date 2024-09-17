import pathlib

root = pathlib.Path(__file__).parent.parent
docs = root / "docs"
examples = docs / "examples"
notebooks = docs / "notebooks"

base = ["  - Examples:", "    - examples/index.md"]


def add_tree(base, folder):
    for file in sorted(folder.rglob("*.py")):
        example_path = file.relative_to(docs)

        if file.parent.name.startswith("."):
            continue
        if file.name.startswith("_wip"):
            continue

        base.append(f"    - {file.stem}: {str(example_path)}")


add_tree(base, examples)
base.append("  - Notebooks:")
base.append("    - notebooks/index.md")
add_tree(base, notebooks)

print("\n".join(base))
