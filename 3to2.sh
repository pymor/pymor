#! /bin/bash

set -u

FIXER="\
 -f annotations\
 -f bitlength\
 -f bool\
 -f bytes\
 -f classdecorator\
 -f collections\
 -f division\
 -f features\
 -f fullargspec\
 -f funcattrs\
 -f getcwd\
 -f imports\
 -f imports2\
 -f input\
 -f intern\
 -f itertools\
 -f kwargs\
 -f memoryview\
 -f metaclass\
 -f methodattrs\
 -f newstyle\
 -f next\
 -f numliterals\
 -f printfunction\
 -f raise\
 -f reduce\
 -f super\
 -f unpacking"

CORES=$(cat /proc/cpuinfo | \grep processor | tail -n 1 | awk '{printf("%d",$3 + 1 );} ')
3to2 -j ${CORES}  -wn --no-diffs ${FIXER} ${1} 
