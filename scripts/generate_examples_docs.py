import pathlib
import mkdocs_gen_files

root = pathlib.Path(__file__).parent.parent
docs = root / "docs"
examples = docs / "examples"

nav = mkdocs_gen_files.Nav()

for file in sorted(examples.rglob("*.py")):
    example_file = file.relative_to(examples).with_suffix("")
    example_path = file.relative_to(docs)

    if file.parent.name.startswith("."):
        continue

    parts = tuple(example_file.parts)
    nav[parts] = example_file.as_posix()

with mkdocs_gen_files.open("examples/nav.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
