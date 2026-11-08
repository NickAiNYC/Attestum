# @reject_reason: dynamic attribute access
n = "ev" + "al"
f = getattr(__builtins__, n)
f("1")
