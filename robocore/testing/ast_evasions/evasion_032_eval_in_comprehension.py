# @reject_reason: dynamic code execution
q = [eval("1") for _ in range(1)]
