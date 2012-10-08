#!/usr/bin/env python
# Copyright (C) 2010 Tobi Vollebregt

def main():
    # Modify path so that development copy is picked up first, if available.
    import imp, os, sys
    sys.path = [os.path.normpath(os.path.dirname(os.path.realpath(__file__)))] + sys.path

    if len(sys.argv) < 2:
        print """Usage: %s foo.py [arguments...]'
Runs foo.py as if it was run directly, except that sys.path is modified so that
development copies of modules (relative to this script) are picked up first.""" % sys.argv[0]
        sys.exit(1)

    # Cloak =)
    # (i.e. remove the `main' function from the global scope)
    global main
    del main

    # Load module by filename after first correcting sys.argv
    sys.argv = sys.argv[1:]
    imp.load_source('__main__', sys.argv[0])


if __name__ == '__main__':
    main()
