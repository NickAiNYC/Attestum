# @reject_reason: module mutation
import sys
sys.modules["fake"] = None
