
 1. [ ] With each 20XX.1 release, update Copyright notices accordingly
 1. [ ] Check wheel install on debian stretch produces an OK error message due to too old python
 1. [ ] Write Release Notes. All new deprecations need to be mentioned.
 1. [ ] Update Readme
 1. [ ] (Create release branch)
 1. [ ] Tag commit
 1. [ ] produce sdist with checked out tag, make sure sdist version is correct
 1. [ ] produce tagged docker-all with `make VER=CURRENT_TAG && make VER=CURRENT_TAG push`
 1. [ ] Push tag
 1. [ ] Wait for CI build to finish
 1. [ ] upload sdist+wheels to pypi (conda builds are automatic via forge)
 1. [ ] bump/create demo docker
 1. [ ] Update https://github.com/pymor/docs/edit/master/index.html to point to new release
 1. [ ] Update Homepage
 1. [ ] Make a github release. Zenodo is hooked into that.
 1. [ ] update MOR Wiki:
        [pyMOR page](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/PyMOR),
        [software comparison table](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Comparison_of_Software)
 1. [ ] update researchgate https://www.researchgate.net/project/pyMOR-Model-Order-Reduction-with-Python
        (check formatting after submit!)
 1. [ ] Submit release to NA-digest: http://icl.utk.edu/na-digest/websubmit.html
 1. [ ] Send release announcement to pymor-dev
 1. [ ] add a new section in https://github.com/pymor/docker-all/blob/master/docs/releases/Dockerfile
 1. [ ] all developers check if (stale) branches can be pruned
 1. [ ] Remove deprecated features in master
 1. [ ] close the GH milestone for the release https://github.com/pymor/pymor/milestones
