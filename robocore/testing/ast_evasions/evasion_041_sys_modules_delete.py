# @reject_reason: module mutation
import sys
del sys.modules["os"]
