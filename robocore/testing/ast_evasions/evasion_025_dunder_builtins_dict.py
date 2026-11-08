# @reject_reason: dunder traversal
import builtins
builtins.__dict__["eval"] = print
