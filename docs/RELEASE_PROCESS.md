# Release process and schedule

The release manager is responsible for ensuring that the following steps are performed:
1. [ ] Fix the following dates (at least 10 weeks before the release, i.e. 5 weeks before the soft freeze):
       soft freeze &rarr; (4 weeks) &rarr; hard freeze &rarr; (1 week) &rarr; release day
1. [ ] As soon as they are fixed, communicate dates for soft freeze, hard freeze and release (via Github discussions).
1. [ ] Go through pull requests and issues without a milestone and assign them to the upcoming release if necessary.
1. [ ] At the day of the soft freeze:
    - [ ] Communicate soft freeze (via Github discussions).
    - [ ] Make all developers aware that they should finish their pull requests (ping in the PRs).
    - [ ] Assign main developers to pull requests that still require review. Make sure that reviews
       are finished early, such that enough time remains to incorporate requested changes.
    - [ ] Start working on the release notes. Make sure that all features merged later are mentioned
       in the release notes as well. All new deprecations need to be mentioned in the release notes.
1. [ ] One week before the hard freeze:
    - [ ] Remind everyone of the upcoming hard freeze and missing reviews or unfinished pull requests (ping in the PRs).
    - [ ] Evaluate if there are pull requests that will not be merged before the hard freeze and decide
       if the hard freeze has to be postponed (which implies moving the release day as well).
    - [ ] If the hard freeze has to be moved, it is moved by exactly one week. This has to be communicated.
1. [ ] At the day of the hard freeze:
    - [ ] Communicate hard freeze (via Github discussions). At this point, no new pull requests
       (except for potential bug fixes) can enter the release. Merging into `main` branch is not allowed until the
       release has happened.
    - [ ] Determine number of commits and so on for the release notes and add these information. Merge the
       release notes branch into `main` when they are finished.
