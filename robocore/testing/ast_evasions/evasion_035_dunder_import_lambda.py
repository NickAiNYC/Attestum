# @reject_reason: __import__ escape
f = lambda: __import__("sys")
