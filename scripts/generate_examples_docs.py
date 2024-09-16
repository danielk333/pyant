import pathlib
import mkdocs_gen_files

md_doc = """

# API

"""

root = pathlib.Path(__file__).parent.parent
docs = root / "docs"
examples = docs / "examples"

nav = mkdocs_gen_files.Nav()

for file in sorted(examples.rglob("*.py")):
    example_file = file.relative_to(examples).with_suffix("")
    doc_path = file.relative_to(docs).with_suffix(".md")

    parts = tuple(doc_path.parts)
    nav[parts] = example_file.as_posix()

with mkdocs_gen_files.open("examples/nav.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
