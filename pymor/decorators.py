# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:12:55 2012

@author: r_milk01
"""

"""Collection of function/class based decorators."""

import functools
import types
import inspect
import logging
import contracts
import copy

def _is_decorated(func):
	return 'decorated' in dir(func)


class DecoratorBase(object):
	"""A base for all decorators that does the common automagic"""
	def __init__(self, func):
		functools.wraps(func)(self)
		func.decorated = self
		self.func = func
		assert _is_decorated(func)

	def __get__(self, obj, ownerClass=None):
		# Return a wrapper that binds self as a method of obj (!)
		self.obj = obj
		return types.MethodType(self, obj)


class DecoratorWithArgsBase(object):
	"""A base for all decorators with args that sadly can do little common automagic"""
	def mark(self, func):
		functools.wraps(func)
		func.decorated = self

	def __get__(self, obj, ownerClass=None):
		# Return a wrapper that binds self as a method of obj (!)
		self.obj = obj
		return types.MethodType(self, obj)


class Deprecated(DecoratorBase):
	"""This is a decorator which can be used to mark functions
	as deprecated. It will result in a warning being emitted
	when the function is used.
	"""

	def __init__(self,alt='no alternative given'):
		self._alt = alt

	def __call__(self,func):
		func.decorated = self
		@functools.wraps(func)
		def new_func(*args, **kwargs):
			frame = inspect.currentframe().f_back
			msg = "DeprecationWarning. Call to deprecated function %s in %s:%s\nUse %s instead" % (
							func.__name__,frame.f_code.co_filename,
							frame.f_code.co_firstlineno,self._alt)
			logging.warning(msg)
			return func(*args, **kwargs)
		return new_func

