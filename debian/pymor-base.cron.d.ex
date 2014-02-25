#
# Regular cron jobs for the pymor-base package
#
0 4	* * *	root	[ -x /usr/bin/pymor-base_maintenance ] && /usr/bin/pymor-base_maintenance
