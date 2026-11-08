# @reject_reason: metaclass injection
class Meta(type):
    pass
class C(metaclass=Meta):
    pass
