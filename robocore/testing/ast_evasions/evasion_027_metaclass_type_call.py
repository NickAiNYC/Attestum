# @reject_reason: metaclass injection
C = type('C', (object,), {'K': 1})
