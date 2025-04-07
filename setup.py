from setuptools import Extension, setup

clib_beam = Extension(
    name="pyant.clibbeam",
    sources=[
        "src/clibbeam/array.c",
        "src/clibbeam/beam.c",
    ],
    include_dirs=["src/clibbeam/"],
)


setup(
    ext_modules=[clib_beam],
)
