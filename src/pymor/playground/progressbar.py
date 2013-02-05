#!/usr/bin/env python

import sys

############################################################
#
# A progress bar that actually shows progress!
#
# Source:
# http://code.activestate.com/recipes/168639-progress-bar-class/
#
############################################################

class ProgressBar:
	""" Creates a text-based progress bar. Call the object with the `print'
		command to see the progress bar, which looks something like this:

		[=======>		22%				  ]

		You may specify the progress bar's width, min and max values on init.
	"""

	def __init__(self, minValue = 0, maxValue = 100, totalWidth=79):
		self.progBar = "[]"   # This holds the progress bar string
		self.min = minValue
		self.max = maxValue
		self.width = totalWidth
		self.amount = minValue           # When amount == max, we are 100% done
		self.update_amount(self.amount)  # Build progress bar string

	def update_amount(self, newAmount = 0):
		""" Update the progress bar with the new amount (with min and max
			values set at initialization; if it is over or under, it takes the
			min or max value as a default. """
		if newAmount < self.min: newAmount = self.min
		if newAmount > self.max: newAmount = self.max
		self.amount = newAmount

		# Figure out the new percent done, round to an integer
		diffFromMin = float(self.amount - self.min)
		percentDone = (diffFromMin / float(self.max - self.min)) * 100.0
		percentDone = int(round(percentDone))

		# Figure out how many hash bars the percentage should be
		allFull = self.width - 2
		numHashes = (percentDone / 100.0) * allFull
		numHashes = int(round(numHashes))

		# Build a progress bar with an arrow of equal signs; special cases for
		# empty and full
		if numHashes == 0:
			self.progBar = "[>%s]" % (' '*(allFull-1))
		elif numHashes == allFull:
			self.progBar = "[%s]" % ('='*allFull)
		else:
			self.progBar = "[%s>%s]" % ('='*(numHashes-1),
										' '*(allFull-numHashes))

		# figure out where to put the percentage, roughly centered
		percentPlace = (len(self.progBar) / 2) - len(str(percentDone))
		percentString = str(percentDone) + "%"

		# slice the percentage into the bar
		self.progBar = ''.join([self.progBar[0:percentPlace], percentString,
								self.progBar[percentPlace+len(percentString):]
								])

	def __str__(self):
		return str(self.progBar)

	def __call__(self, value):
		""" Increases the amount by value, and writes to stdout. Prints a
		    carriage return first, so it will overwrite the current line in
		    stdout."""
		if self.amount < self.max:
			print '\r',
			self.update_amount(self.amount + value)
			sys.stdout.write(str(self))
			sys.stdout.write(self.amount < self.max and "\r" or "\n")
			sys.stdout.flush()

	def setMaximum(self, value):
		self.max = value

	def maximum(self):
		return self.max

if __name__ == '__main__':
	from time import sleep
	p = ProgressBar()
	for i in range(0, 201):
		p(1)
		if i == 90:
			p.max = 200
		sleep(0.02)
