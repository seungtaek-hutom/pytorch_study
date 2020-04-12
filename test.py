class Foo:
    def __init__(self, num):
        self.num = num
        print(num)

    def __call__(self, num):
        print(num)


f = Foo(3)
f(5)