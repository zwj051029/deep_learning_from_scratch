class Man:
    def __init__(self, name):
        self.name_ = name
        print("Initilized!")

    def hello(self):
        print("hello " + self.name_ + "!")

    def goodbye(self):
        print("goodbye " + self.name_ + "!")

man = Man("Tom")
man.hello()
man.goodbye()