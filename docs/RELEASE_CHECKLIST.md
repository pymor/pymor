# Release checklist

 1. [ ] Write release notes. All new deprecations need to be mentioned.
 1. [ ] Update `README.md`.
 1. [ ] (Create release branch.)
 1. [ ] Tag commit.
 1. [ ] Produce sdist with checked out tag, make sure sdist version is correct.
 1. [ ] Produce tagged docker with `make VER=CURRENT_TAG && make VER=CURRENT_TAG push`.
 1. [ ] Push tag.
 1. [ ] Wait for CI build to finish.
 1. [ ] Bump/create demo docker.
 1. [ ] Update homepage.
 1. [ ] Make a GitHub release. Zenodo is hooked into that.
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
 1. [ ] Remove deprecated features in main.
 1. [ ] Close the [GitHub milestone](https://github.com/pymor/pymor/milestones) for the release.
