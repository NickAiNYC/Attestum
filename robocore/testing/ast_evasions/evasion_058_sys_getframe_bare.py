# @reject_reason: frame introspection
def g():
    fr = sys._getframe(0)
    return fr
