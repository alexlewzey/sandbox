class Dog:
    species = 'mamel'

    def __init__(self, name, n_legs, color, pack):
        self.name = name
        self.n_legs = n_legs
        self.color = color
        self.pack = pack

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, n_legs={self.n_legs}, color={self.color}, pack={self.pack})'

    def running(self):
        print(self.name + f'is running! with {self.n_legs} legs')

    def iterpack(self):
        for m in self.pack:
            yield m


class Moje(Dog):
    def __init__(self, mood):
        super().__init__('moje', 4, 'liver white', ['dad', 'wanda'])
        self.mood = mood
        self.dinner = 'samlmon skin'

    def binks(self):
        print(f'{self.name} is a binks')

    def __getattr__(self, item): return getattr(self.dinner, item)



# you can deligate getattr to a property of the class
moje = Moje('playful')
getattr(moje, 'split')()

moje.split()
