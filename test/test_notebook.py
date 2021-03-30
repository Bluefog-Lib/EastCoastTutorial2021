import atexit
import functools
import glob
import os
import subprocess
import time

import papermill as pm
import pytest

SKIP_NOTEBOOKS = []

def _list_all_notebooks():
    output = subprocess.check_output(["git", "ls-files", "*.ipynb"])
    return set(output.decode("utf-8").splitlines())


def _tested_notebooks():
    """We list all notebooks here, even those that are not """

    all_notebooks = _list_all_notebooks()
    skipped_notebooks = functools.reduce(
        lambda a, b: a.union(b),
        list(set(glob.glob(g, recursive=True)) for g in SKIP_NOTEBOOKS),
    )

    return sorted(
        os.path.abspath(n) for n in all_notebooks.difference(skipped_notebooks)
    )


@pytest.mark.parametrize("notebook_path", _list_all_notebooks())
def test_notebooks_against_bluefog(notebook_path):
    try:
        notebook_file = os.path.basename(notebook_path)
        notebook_rel_dir = os.path.dirname(os.path.relpath(notebook_path, "."))
        os.chdir(notebook_rel_dir)
        out_path = f".output/{notebook_rel_dir}/{notebook_file[:-6]}.out.ipynb"
        if not os.path.exists(f".output/{notebook_rel_dir}"):
            os.makedirs(f".output/{notebook_rel_dir}")
        print("Start papermill on ", notebook_path)
        pm.execute_notebook(
            notebook_file,
            out_path,
            log_output=True,
            start_timeout=60,
            execution_timeout=120,
        )
        print("End papermill")
    except:
        raise RuntimeError(f"Failed to run {notebook_path}")
