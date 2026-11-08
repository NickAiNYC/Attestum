# @reject_reason: frame introspection
import sys
def f():
    return sys._getframe(1).f_globals
