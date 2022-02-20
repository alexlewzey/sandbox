"""how to use getters and setters in python"""


class Rectangle:
    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width: float):
        if width <= 0:
            raise ValueError('Must be a positive value')
        else:
            self._width = width

    def __repr__(self):
        return f'Rectangle(width={self.width}, height={self.height})'

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height: float):
        if height <= 0:
            raise ValueError('height must be positive')
        else:
            self._height = height

    def area(self):
        return self.width * self.height

    def __eq__(self, other):
        if isinstance(other, Rectangle):
            return True if (self.width == other.width) and (self.height == other.height) else False
        else:
            return False


a = Rectangle(10, 15)
b = Rectangle(10, 15)
print(a)
a.width = -100
