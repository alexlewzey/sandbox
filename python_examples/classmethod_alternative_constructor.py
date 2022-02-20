class Dog:
    legs = 4
    nose = 1

    def __init__(self, name, color, breed):
        self.name = name
        self.color = color
        self.breed = breed

    def run(self):
        return f'{self.name} is playing with {self.legs} legs'

    @classmethod
    def from_string(cls, s):
        return cls(*s.split('-'))


mole = Dog('mole', 'kidney', 'spaner')
mole.run()

pooh = Dog.from_string('pooh-black-whipet')
pooh.run()


