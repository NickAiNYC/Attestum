# @reject_reason: dynamic code execution
w = exec(compile('1', '', 'eval'))
