# Release checklist

Replace `RELEASE_TAG` below with the actual release tag.

 1. [ ] Write release notes. All new deprecations need to be mentioned.
 1. [ ] Update `README.md`.
 1. [ ] Create a release branch in `pymor/pymor`. Should have a `.x` as the last part of the branch name in contrast
        to the `RELEASE_TAG`.
 1. [ ] Tag commit in `pymor/pymor` as `RELEASE_TAG`.
        Use an annotated tag (`git tag -a RELEASE_TAG -m RELEASE_TAG`) with the annotation being `RELEASE_TAG`.
 1. [ ] Run `python setup.py sdist` (on checked out tag `RELEASE_TAG`) and check that it
        produces `dist/pymor-RELEASE_TAG.tar.gz`.
 1. [ ] Tag commit in `pymor/docker` as `RELEASE_TAG`, make sure to use the commit mentioned in the `.env` in the
        tagged commit in `pymor/pymor`. Use an annotated tag with the annotation being the version number.
        For instance:

       ```bash
       source pymor/pymor/.env
       cd pymor/docker
       git checkout "${CI_IMAGE_TAG}"
       git tag -a RELEASE_TAG -m RELEASE_TAG
       ```
 1. [ ] Push tag `RELEASE_TAG` to `pymor/docker`.
 1. [ ] Wait for CI build to finish in `pymor/docker`.
 1. [ ] Push tag `RELEASE_TAG` to `pymor/pymor`.
 1. [ ] Wait for CI build to finish in `pymor/pymor`. The tagged commit will be deployed to PyPI automatically,
        see [the developer docs](https://docs.pymor.org/main/developer_docs.html#stage-deploy).
 1. [ ] Bump/create demo docker, i.e. in `pymor/docker` go to the `demo`-folder and copy the subfolder of the last
        version, change the version in the `Dockerfile` (lines 1 and 6) and extend the `DEMO_TAGS` in `common.mk`
        (last line).
 1. [ ] Update homepage (`gh-pages` branch in `pymor/pymor`, similar to changes in `README.md`).
 1. [ ] Make a GitHub release. [Zenodo](https://zenodo.org/record/7494334) is hooked into that.
 1. [ ] Update MOR Wiki:
        [pyMOR page](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/PyMOR),
        [software comparison table](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Comparison_of_Software).
 1. [ ] Update [ResearchGate](https://www.researchgate.net/project/pyMOR-Model-Order-Reduction-with-Python)
        (check formatting after submit!).
 1. [ ] Submit release to [NA-digest](http://icl.utk.edu/na-digest/websubmit.html).
 1. [ ] Announce release in
        [GitHub discussions](https://github.com/pymor/pymor/discussions).
 1. [ ] All developers check if (stale) branches can be pruned.
 1. [ ] All developers check for `.mailmap` correctness.
 1. [ ] Remove deprecated features in main in `pymor/pymor` (omit this step in case of a bugfix release).
 1. [ ] Close the [GitHub milestone](https://github.com/pymor/pymor/milestones) for the release.
