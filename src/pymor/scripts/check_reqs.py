#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import typer
import pkg_resources


def _output_rich(filename, status):
    from rich.console import Console
    from rich.table import Table
    tbl = Table("Requirement", "Status", "Reason", title=f"Requirements checked for {filename}")
    for req, st in status.items():
        if st is None:
            tbl.add_row(req, "[green]✓[/green]", "")
        else:
            tbl.add_row(req, "[red]x[/red]", str(st))
    console = Console()
    console.print(tbl)


def _output_plain(filename, status):
    print(f"Requirements checked for {filename}")
    for req, st in status.items():
        if st is None:
            print(f"OK {req}")
        else:
            print(f"Failed {req}: {st}")


def _process_file(filename):
    req_file = open(filename).read().splitlines()
    status = {}
    for line in req_file:
        try:
            requirement = str(pkg_resources.Requirement.parse(line))
            pkg_resources.require(requirement)
            status[requirement] = None
        except pkg_resources.DistributionNotFound as e:
            status[requirement] = e
        except pkg_resources.VersionConflict as d:
            status[requirement] = d
        # Requirement.parse cannot raises on include and comment directives
        except (ValueError,) as f:
            if line.startswith('-r'):
                status.update(_process_file(line[3:]))
            elif line.startswith('http'):
                status[line] = 'http link skipped'
            elif line.startswith('#'):
                continue
            else:
                raise RuntimeError(f'Error parsing {filename}:\n{f}')

    return status


def main(filename: str):
    status = _process_file(filename)
    try:
        _output_rich(filename, status)
    except (ImportError, ModuleNotFoundError):
        _output_plain(filename, status)
    fails = sum(int(st is not None) for st in status.values())
    raise typer.Exit(code=fails)


def run():
    typer.run(main)


if __name__ == '__main__':
    run()
