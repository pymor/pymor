# Contributing to pyMOR

pyMOR is an open project and we welcome any contributions
which improve or extend pyMOR. The purpose of this document
is to make the contribution process easier for you and to answer
some questions which might arise.


## Contribution process

When you have written some code you want to contribute to 
pyMOR, the first thing you should do is to ensure that
your code is contained in one or several [git](https://git-scm.com/)
commits which have commit messages appropriately describing
their content.

The recommended way to send us your code is to create a
[pull request](https://help.github.com/articles/creating-a-pull-request/)
on [github](https://github.com/pymor/pymor) containing the code.
For this to work you will have to create a new github account
(if you do not have one already) and push your commits into a
[fork](https://guides.github.com/activities/forking/) of pyMOR 
owned by you. Alternatively, you may send patches to our
[mailinglist](http://listserv.uni-muenster.de/mailman/listinfo/pymor-dev)
that have been created using `git format-patch`.

Once we have received your code, it will be reviewed and
discussed with you by pyMOR's 
[main developers](#becoming-a-main-developer). If it is found suitable
for inclusion into pyMOR, your code will be merged into pyMOR's
main repository. We may also make suggestions how to modify
or improve your code to make it better fit into the project.

Of course, were are happy to help you prepare contributions to pyMOR.
Feel free to [ask](http://listserv.uni-muenster.de/mailman/listinfo/pymor-dev)
for help at any time.


## License

By submitting contributions to pyMOR, you give us the right to
publish your code under pyMOR's 
[license](https://github.com/pymor/pymor/blob/master/LICENSE.txt).
In order to do so, you need to have the copyright for your code.
If you do not hold the copyright, you have to ask the copyright 
holder for permission to contribute the code to pyMOR.

pyMOR's license ([2-clause BSD](https://opensource.org/licenses/BSD-2-Clause))
is a permissive open source license without
[copyleft](https://en.wikipedia.org/wiki/Copyleft). In particular,
be aware that this license allows commercial use of your code when the 
license including the preceding copyright notice is reproduced.
On the other hand, this will also enable you to create a commercial
project based on pyMOR (including the parts of pyMOR written by others).

Please note that the copyright over your code is fully retained by
you and not transferred in any way to the pyMOR project. However,
you should be aware that there is no turning back: once your code
is published under the BSD license, there is no way of revoking
this license. Of course, you are free to not publish future versions
of your code under the same license.


## Attribution

When you have contributed code to pyMOR you and the content of your
contribution will be mentioned in the project's AUTHORS.md file.
Contributions are grouped by release, so if you have contributed
code to multiple releases, you will be mentioned for each of
these releases. Moreover, you may add attribution notices (e.g.
author name, corresponding publications, funding institutions)
to the code you contribute.

If, for some reason, you do not wish to be mentioned in the AUTHORS.md
file, please give us a short note. Also note that we cannot give
attributions to trivial changes, such as fixing a typo in the
documentation or correcting a very simple bug. However, your changes
including your authorship will always be included in the git
history of the project.


## Coding style

pyMOR follows the coding style of 
[PEP8](https://www.python.org/dev/peps/pep-0008/) apart from a
few exceptions. Configurations for the 
[PEP8](https://pypi.python.org/pypi/pep8) and 
[flake8](https://pypi.python.org/pypi/flake8) code
checkers are contained in 
[setup.cfg](https://github.com/pymor/pymor/blob/master/setup.cfg).

As an additional rule when calling functions, positional
arguments should genereally be passed as positional arguments
whereas keyword arguments should be passed as keyword arguments.
This will make your code less likely to break, when the called
function is extended.

All functions and classes called or instantiated by users should
be sufficiently well documented.


## Becoming a main developer

pyMOR's main developers form a small 
[group](https://github.com/orgs/pymor/people?query=role:owner+)
of developers which, apart from making contributions to pyMOR,
have the job of guiding and maintaining the future development of
pyMOR. They are the only persons with direct push acces to pyMOR's
main repository and have administrative priviliges over the
[pyMOR organization](https://github.com/pymor) on github.

As pyMOR is an open project, everyone is invited to step up to
become a main developer. The current main developers will decide
by simple majority vote if a candidate should be included into the
group. In order to be accepted, an applicant should have made
major contributions to pyMOR, both in form of code and by taking part
in discussions on the mailing list and through github. The applicant
should be able to commit to the project for a time period of at
least one year.

Main developers will be automatically retired (losing all priviliges)
if they have not shown any relevant activity over a period of one
year. Under special circumstances, the main developers 
(excluding the developer in question) may decide by simple majority
vote to keep the developer in the group of main developers.
