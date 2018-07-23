#!/usr/bin/env python
# coding=utf-8

class Foo(object):

    instance = None

    def __init__(self, name):
        self.name = name


    @classmethod
    def get_instance(cls):
        if cls.instance:
            return cls.instance
        else:
            obj = cls('hexm')
            cls.instance = obj
            return obj

obj = Foo.get_instance()
obj1 = Foo.get_instance()
print(obj.name)
print(obj1.name)
print(Foo.instance)
print(obj)


class FF(object):
    _instance=None
    def __init__(self):
        self.s=123

    def __new__(cls, *args, **kw):
        if not cls._instance:
            cls._instance = super(FF, cls).__new__(cls, *args, **kw)
        return cls._instance



f1=FF()
f2=FF()
print(f1==f2)
f1.s=345
print(f2.s)