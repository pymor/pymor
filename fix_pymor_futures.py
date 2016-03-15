'''
Add 'from __future__ import absolute_import' to any file
that uses imports.
'''
from lib2to3 import fixer_base
from lib2to3.pygram import python_symbols as syms
from lib3to2.fixer_util import future_import

class FixPymorFutures(fixer_base.BaseFix):
    order = 'post'
    run_order = 10 
    
    def __init__(self, options, log):
        super(FixPymorFutures, self).__init__(options, log)
        self.__pymor_added = None
        
    def start_tree(self, tree, filename):
        super(FixPymorFutures, self).start_tree(tree, filename)
        self.__pymor_added = False
        
    def match(self, node):
        return (node.type in (syms.import_name, syms.import_from) 
                and not self.__pymor_added)
    
    def transform(self, node, results):
        try:
            future_import('absolute_import', node)
            future_import('division', node)
            future_import('print_function', node)
        except ValueError:
            pass
        else:
            self.__pymor_added = True
        
    def finish_tree(self, tree, filename):
        fixer_base.BaseFix.finish_tree(self, tree, filename)
