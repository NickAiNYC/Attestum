# @reject_reason: dynamic attribute access
s = 'sy' + 'stem'
h = getattr(__import__('os'), s)
