#!/usr/bin/env python3
# This script is periodically executed on docs.pymor.org.

# Regenerate versions.json, badge.json and index.html

import json
import re
from collections import OrderedDict
from pathlib import Path

base_path = Path('/var/www/docs.pymor.org/')

def versions():
    pattern = re.compile(r'^20\d\d-\d-\d$')

    for d in base_path.iterdir():
        if not d.is_dir():
            continue
        if not pattern.match(d.name):
            continue
        dir_name = d.name
        version_name = '.'.join(dir_name.split('-')[:2])
        yield version_name, dir_name

versions_dict = OrderedDict()
for v, d in sorted(versions(), reverse=True):
    if v not in versions_dict:
        versions_dict[v] = d
versions_dict['development'] = 'main'
current_version, current_version_dir = next(iter(versions_dict.items()))

versions_file = base_path / 'versions.json'
versions_file.write_text(json.dumps(versions_dict))

badge_data = {'schemaVersion': 1,
              'label': 'docs',
              'message': current_version_dir.replace('-', '.'),
              'color': 'green'}
badge_file = base_path / 'badge.json'
badge_file.write_text(json.dumps(badge_data))

index_html = f"""
<head>
  <!-- This should always redirect to the last release -->
  <!-- this file is deployed to the pages root from CI -->
  <meta http-equiv="refresh" content="0; URL=https://docs.pymor.org/{current_version_dir}/index.html" />
</head>
"""[1:]
with open('/var/www/docs.pymor.org/index.html', 'w') as f:
    f.write(index_html)
