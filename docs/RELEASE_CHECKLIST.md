# Release checklist

Replace `RELEASE_TAG` below with the actual release tag.

- [ ] Check that release notes are finished and merged.
      (Omit in case of a bugfix release.)
- [ ] Update `README.md`.
- [ ] Create a release branch in `pymor/pymor`.
      Should have a `.x` as the last part of the branch name in contrast
      to the `RELEASE_TAG`.
      (Omit in case of a bugfix release.)
- [ ] Tag commit in `pymor/pymor` as `RELEASE_TAG`.
      Use an annotated tag (`git tag -a RELEASE_TAG -m RELEASE_TAG`) with the
      annotation being `RELEASE_TAG`.
- [ ] Run `python setup.py sdist` (on checked out tag `RELEASE_TAG`) and check
      that it produces `dist/pymor-RELEASE_TAG.tar.gz`.
- [ ] Tag commit in `pymor/docker` as `RELEASE_TAG`, make sure to use the commit
      mentioned in the `.env` in the tagged commit in `pymor/pymor`.
      Use an annotated tag with the annotation being the version number.
      For instance:

      ```bash
      source pymor/pymor/.env
      cd pymor/docker
      git checkout "${CI_IMAGE_TAG}"
      git tag -a RELEASE_TAG -m RELEASE_TAG
      ```

- [ ] Push tag `RELEASE_TAG` to `pymor/docker`.
- [ ] Wait for CI build to finish in `pymor/docker`.
- [ ] Push tag `RELEASE_TAG` to `pymor/pymor`.
- [ ] Wait for CI build to finish in `pymor/pymor`.
      The tagged commit will be deployed to PyPI automatically, see
      [the developer docs](https://docs.pymor.org/main/developer_docs.html#stage-deploy).
- [ ] Bump/create demo docker, i.e., in `pymor/docker` go to the `demo` folder
      and copy the subfolder of the last version, change the version in the
      `Dockerfile` (lines 1 and 6) and extend the `DEMO_TAGS` in `common.mk`
      (last line).
      (Omit in case of a bugfix release.)
- [ ] Update homepage
      (`gh-pages` branch in `pymor/pymor`, similar to changes in `README.md`).
- [ ] Check if the [docs](https://docs.pymor.org) got updated.
- [ ] Check if [`conda-forge/pymor-feedstock`](https://github.com/conda-forge/pymor-feedstock)
      got updated.
- [ ] Make a GitHub release
      ([Zenodo](https://zenodo.org/record/7494334) is hooked into that) and
      announce the release in
      [GitHub Discussions](https://github.com/pymor/pymor/discussions)
      (can be done as part of making the release).
- [ ] Update MOR Wiki:
      [pyMOR page](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/PyMOR),
      [software comparison table](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Comparison_of_Software).
- [ ] Submit release to [NA-digest](http://icl.utk.edu/na-digest/websubmit.html).
      (Omit in case of a bugfix release.)
- [ ] All developers check if (stale) branches can be pruned.
- [ ] All developers check for `.mailmap` correctness.
- [ ] Remove deprecated features in main in `pymor/pymor`.
      (Omit in case of a bugfix release.)
- [ ] Close the [GitHub milestone](https://github.com/pymor/pymor/milestones)
      for the release.
      (Omit in case of a bugfix release.)
