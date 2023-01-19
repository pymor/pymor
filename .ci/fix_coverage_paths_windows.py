import sqlite3
import sys
from pathlib import Path

"""Replace all windows path delimiters with *nix in coverage sqlite dbs
"""


def _fix(path: Path):
    db = sqlite3.connect(path)
    db.cursor().execute('UPDATE file SET path = REPLACE(path,"\\", "/")')
    db.commit()


if __name__ == '__main__':
    for fn in sys.argv[1:]:
        _fix(Path(fn))
