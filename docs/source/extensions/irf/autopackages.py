import importlib
import re
from collections import OrderedDict
from pathlib import Path
from typing import List

import docutils.nodes as nodes
import docutils.nodes
import docutils.parsers.rst
import docutils.utils
import docutils.frontend
from docutils.parsers.rst import directives
from sphinx.directives.other import TocTree
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import OptionSpec
from sphinx.parsers import RSTParser
from sphinx.util import logging

logger = logging.getLogger(__name__)

python_extensions = ["py", "pyx"]
title_chars = {
    "section": "=",
    "subsection": "-",
    "subsubsection": "^",
}

pattern = re.compile(r"^\.\. irf_autopackages::.*")

GLOBAL_ENV = {"generated_rst_files": []}


def generate_autosummary_directive(
    subpackage,
    module_list,
    autosummary_opts,
    title_char,
    title=None,
):
    """Generate an autosummary rst directive that includes all subpackages
    and leads with an automodule of the top package.
    """
    autosummary_directive = ""

    if title is None:
        title = subpackage.split(".")[-1]
    if len(title) > 0:
        autosummary_directive += title + "\n" + title_char * len(title)

    autosummary_directive += f"""

.. automodule::
    {subpackage}
"""

    if len(module_list) > 0:
        autosummary_directive += ".. autosummary::"

        template = autosummary_opts["template"]
        toctree = autosummary_opts["toctree"]

        if template is not None:
            autosummary_directive += f"\n   :template: {template}"
        if toctree is not None:
            toc = subpackage.replace(".", "/")
            autosummary_directive += f"\n   :toctree: {toctree}/{toc}"

        autosummary_directive += "\n" * 2 + "    "
        autosummary_directive += "\n    ".join(module_list)
    return autosummary_directive


def generate_toc_directive(autopackages_dir, fname, maxdepth):
    toc_directive = f"""
.. toctree::
    :maxdepth: {maxdepth}

    """
    content = [f"{autopackages_dir}/{fname}"]
    toc_directive += "    " + "\n    ".join(content)
    return toc_directive, content


def get_module_tree(name):
    """Uses an importable name to find the underlying structure of python files
    of that package and groups the modules according to subpackage.
    """
    package = importlib.import_module(name)
    if not hasattr(package, "__path__"):
        raise AttributeError(
            f"Given package {name} is not a package or a \
sub-package ({name} has no __path__, most likley a module)"
        )

    path = package.__path__
    path = Path(path) if isinstance(path, str) else Path(path[0])

    files = []
    for ext in python_extensions:
        files += list(path.rglob(f"*.{ext}"))

    name_ind = len(name.split("."))
    top_ind = len(path.parts) - name_ind

    module_tree = {}
    for file in files:
        sub_package = ".".join(file.parts[top_ind:-1])
        if sub_package not in module_tree:
            module_tree[sub_package] = []
        module = sub_package
        module += "." + file.stem
        if file.stem == "__init__":
            continue

        module_tree[sub_package].append(module)

    for sub_package in module_tree:
        module_tree[sub_package].sort()

    module_tree = OrderedDict(sorted(module_tree.items()))
    return module_tree


def generate_package_rst(names, excludes, title, autosummary_opts):
    """Generate a rst content with autsummary derictives based on the
    input package names.
    """
    src_content = ""
    for name in names:
        module_tree = get_module_tree(name)

        # Exclude subpackages and modules according to option
        for exclude in excludes:
            if exclude in module_tree:
                module_tree.pop(exclude)
        for subpackage, modules in module_tree.items():
            for exclude in excludes:
                if exclude in modules:
                    modules.remove(exclude)

        # Subpackage was removed, skip
        if name not in module_tree:
            continue

        # Autosummary the top level
        modules = module_tree.pop(name)
        src_content += generate_autosummary_directive(
            name,
            modules,
            autosummary_opts,
            title_char=title_chars["section"],
            title=title,
        )
        src_content += "\n" * 2

        for subpackage, modules in module_tree.items():
            src_content += generate_autosummary_directive(
                subpackage,
                modules,
                autosummary_opts,
                title_char=title_chars["subsection"],
            )
            src_content += "\n" * 2

    return src_content


class AutoPackages(SphinxDirective):
    """Automatically genereate rst-files similar to autosummary but for the
    subpackage structure of the input content. The rst file is saved to the
    filename in the first rst argument.
    """

    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec: OptionSpec = {
        "toctree": directives.unchanged,
        "template": directives.unchanged,
        "exclude": directives.unchanged,
        "title": directives.unchanged,
        "maxdepth": directives.unchanged,
    }

    def run(self) -> List[nodes.Node]:
        assert len(self.arguments) > 0, "output file name needed as argument"
        fname = self.arguments[0]
        names = [
            x.strip().split()[0]
            for x in self.content
            if x.strip() and re.search(r"^[~a-zA-Z_]", x.strip()[0])
        ]

        # We dont have an env yet, still in build stage
        try:
            env = self.env
            config = self.config
            docname = env.docname
        except AttributeError:
            env = GLOBAL_ENV["env"]
            config = GLOBAL_ENV["config"]
            docname = GLOBAL_ENV["docname"]

        autopackages_dir = config.irf_autopackages_toctree
        src_dir = Path(env.srcdir)

        # This might not work when source files are not in the
        # dir as the file with the dirictive?
        source_dir = (src_dir / docname).resolve().parent
        out_dir = (src_dir / autopackages_dir).resolve()
        out_dir.mkdir(exist_ok=True)
        out_rst = out_dir / (fname + ".rst")

        level_diff = len(out_dir.parts) - len(source_dir.parts)

        title = self.options.get("title", None)
        maxdepth = self.options.get("maxdepth", 3)
        template = self.options.get("template", None)
        toctree = self.options.get("toctree", "autosummary")
        excludes = self.options.get("exclude", "").split()
        toctree = "../" * level_diff + toctree

        autosummary_opts = {
            "template": template,
            "toctree": toctree,
        }

        if fname not in GLOBAL_ENV["generated_rst_files"]:
            src_content = generate_package_rst(
                names,
                excludes,
                title,
                autosummary_opts,
            )
            logger.info(f"[autopackages] Writing {out_rst}")
            with open(out_rst, "w") as fh:
                fh.write(src_content)

            # Add the new file to the enviornment
            fpath = str(Path(autopackages_dir) / fname)
            env.found_docs.add(fpath)

            # Remember we were here when we parse the source files
            GLOBAL_ENV["generated_rst_files"].append(fname)
            out_nodes = []
        else:
            block_text, content = generate_toc_directive(autopackages_dir, fname, maxdepth)
            toc = TocTree(
                name=self.name,
                arguments=[],
                options={
                    "maxdepth": maxdepth,
                },
                content=content,
                lineno=self.lineno,
                content_offset=self.content_offset,
                block_text=block_text,
                state=self.state,
                state_machine=self.state_machine,
            )
            out_nodes = toc.run()

        return out_nodes


def parse_rst(text, app):
    parser = RSTParser()
    parser.set_application(app)
    components = (RSTParser,)
    settings = docutils.frontend.OptionParser(components=components).get_default_values()
    document = docutils.utils.new_document("", settings=settings)
    parser.parse(text, document)
    return document


def generate_rst_files(app):
    """Generate the rst files that include all the autosummar derictives from
    the current set of source files for this build.
    """

    logger.info("[autopackages] Generating AutoPackages RST files")

    env = app.builder.env
    GLOBAL_ENV["env"] = env
    GLOBAL_ENV["config"] = app.config

    src_dir = Path(env.srcdir)
    source_docs = [
        env.doc2path(x, base=None) for x in env.found_docs if Path(env.doc2path(x)).is_file()
    ]

    found_derictives = []
    # Find all autopackages derictives in the files
    for file in source_docs:
        # logger.info(f'[autopackages] Searching {file}')
        fh = open(src_dir / file, "r")
        derictive = None
        for line in fh:
            if derictive is not None:
                if line.startswith(" " * 3) or len(line.strip()) == 0:
                    derictive += line
                else:
                    found_derictives.append((derictive, file))
                    derictive = None

            if derictive is None:
                results = pattern.search(line)
                if results is None:
                    continue
                derictive = line
        fh.close()

    for autopackages_text, file in found_derictives:
        logger.info(f"[autopackages] Parsing {file}")
        GLOBAL_ENV["docname"] = file
        parse_rst(autopackages_text, app)


def setup(app):
    app.add_directive("irf_autopackages", AutoPackages)
    app.add_config_value("irf_autopackages_toctree", "autopackages", True)
    app.connect("builder-inited", generate_rst_files, priority=90)
    return {
        "version": "0.1",
        "parallel_read_safe": False,
        "parallel_write_safe": False,
    }
