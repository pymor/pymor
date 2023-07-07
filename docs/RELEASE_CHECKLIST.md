# Release checklist

## After hard freeze

- [ ] Bump all CI requirements to current versions:

    ```bash
    rm requirements-ci-current.txt requirements-ci-fenics.txt conda-lock.yml
    make ci_requirements
    ```

    Commit all changes.
    Before proceeding, wait until changes are merged.
- [ ] Check if all tutorials are rendered correctly.
- [ ] Check if binder links are functional.
- [ ] Use `hatch build` to create wheel (`pip install hatch` or install pyMOR with `dev` extras).
      `pip install` wheel (including extras) into new venv.
      Check if installation is working as expected.

## Release day

Replace `RELEASE_TAG` below with the actual release tag.

- [ ] Check that release notes are finished and merged.
      (Omit in case of a bugfix release.)
- [ ] Update `README.md`.
- [ ] Create a release branch in `pymor/pymor`.
      Should have a `.x` as the last part of the branch name in contrast
      to the `RELEASE_TAG`.
      (Omit in case of a bugfix release.)
- [ ] Use `hatch version RELEASE_TAG` to update `__version__` in `src/pymor/__init__.py`.
      Merge into release branch.
- [ ] Run

    ```bash
    rm -r ./dist  # ensure that we do not accidentally publish old wheels
    hatch build
    ```

    to generate release sdist and wheel. Check that versions are correct.
- [ ] Tag commit in release branch as `RELEASE_TAG`.
      Use an annotated tag (`git tag -a RELEASE_TAG -m RELEASE_TAG`) with the
      annotation being `RELEASE_TAG`.
      Push `RELEASE_TAG` to GitHub.
- [ ] Wait for CI build for tagged commit to finish (see the list of pipelines at
      [zivgitlab](https://zivgitlab.uni-muenster.de/pymor/pymor/-/pipelines)).
- [ ] Merge the automatic PR at [`pymor/docs`](https://github.com/pymor/docs) and
      wait for the CI build to finish.
- [ ] Check again that documentation for tagged commit (not release branch) is built correctly.
      Check that binder links work and `.binder/Dockerfile` in `pymor/docs@RELEASE_TAG` uses the
      correctly tagged base image.
      (Should, for some reason, CI fail to produce correct setup, manually push release tags to
      registry using

    ```bash
    make ci_images_pull
    make TARGET_TAG=RELEASE_TAG ci_images_push
    ```

    and update `Dockerfile` in TARGET_TAG branch of `pymor/docs` manually.)
- [ ] Publish wheel to PyPI using

    ```bash
    hatch publish
    ```

- [ ] Update homepage
      (`gh-pages` branch in `pymor/pymor`, similar to changes in `README.md`).
- [ ] Check if the [docs](https://docs.pymor.org) got updated to point to new release.
      (Should happen hourly via `Release update` action. Can also be triggered manually.)
- [ ] Check if [`conda-forge/pymor-feedstock`](https://github.com/conda-forge/pymor-feedstock)
      got updated.
- [ ] Make a GitHub release
      ([Zenodo](https://zenodo.org/record/7494334) is hooked into that) and
      announce the release in
      [GitHub Discussions](https://github.com/pymor/pymor/discussions)
      (can be done as part of making the release).
- [ ] Close the [GitHub milestone](https://github.com/pymor/pymor/milestones)
      for the release.
      (Omit in case of a bugfix release.)
- [ ] Update MOR Wiki:
      [pyMOR page](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/PyMOR),
      [software comparison table](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Comparison_of_Software).
- [ ] Submit release to [NA-digest](http://icl.utk.edu/na-digest/websubmit.html).
      (Omit in case of a bugfix release.)

## After release

- [ ] Bump version in main branch to NEXT_TARGET_TAG.dev0.
- [ ] All developers check if (stale) branches can be pruned.
- [ ] All developers check for `.mailmap` correctness.
- [ ] Remove deprecated features in main in `pymor/pymor`.
      (Omit in case of a bugfix release.)
