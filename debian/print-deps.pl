#!/usr/bin/perl
use strict;
use warnings;

use Dpkg::Control::Info;
use Dpkg::Deps;

my $control = Dpkg::Control::Info->new("control");
my $fields = $control->get_source();
my $build_depends = deps_parse($fields->{'Build-Depends'});
print deps_concat($build_depends) . "\n";
foreach my $dep_and ($build_depends) {
    foreach my $dep ($dep_and->get_deps()) {
        print "    - ".$dep->{'package'}."\n";
    }
}
