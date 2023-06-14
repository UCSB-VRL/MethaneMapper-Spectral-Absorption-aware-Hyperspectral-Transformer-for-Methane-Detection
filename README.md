# MethaneMapper: Spectral Absorption aware Hyperspectral Transformer for Methane Detection


The source code and data upload is under progress. Thank you so much for the patience.


# For developers:
Pre-Commit
In order to provide some degree of uniformity in style, we can use the pre-commit tool to clean up source files prior to being committed. Pre-Commit runs a number of plugins defined in .pre-commit-config.yaml. These plugins enforce coding style guidelines.

Install pre-commit by following the instructions here: https://pre-commit.com/#install

Linux:
```
pip install pre-commit
```

Once pre-commit is installed, install the git hooks by typing:
```
# In git repo root dir
pre-commit install
```
Now, whenver you commit code, pre-commit will clean it up before it is committed. You can then add the cleaned-up code and commit it. This enforces coding standards and consistency across developers.
