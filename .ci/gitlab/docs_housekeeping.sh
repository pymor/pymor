#!/bin/sh
# This script is periodically executed on docs.pymor.org.

BASEDIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# remove documentation from merge queue builds
find /var/www/docs.pymor.org -maxdepth 1 -type d -name 'gh-readonly-queue-*' -exec rm -r '{}' \;

# remove old documentation
find /var/www/docs.pymor.org -maxdepth 1 -type d -mtime +90 -not -name '20??-*-*' -not -name '20??.*.*' -not -name '0.?.?' -not -name 'main' -not -name 'asv' -exec rm -r '{}' \;

# rebuild https://docs.pymor.org/list.html
$BASEDIR/docs_makeindex.py /var/www/docs.pymor.org

# update version information
$BASEDIR/docs_update_versions.py
