import random;


class RandomIterator(object):
    def __init__(self, ends, len=-1):
        self._ends = ends;
        self.current = 0;
        self._indexs = range(0, ends);
        random.shuffle(self._indexs);

        if len > 0 :
            self._indexs = self._indexs[:len]

    def __iter__(self):
        return self

    def next(self):  # Python 3: def __next__(self)
        if self.current >= len(self._indexs):
            raise StopIteration
        else:
            self.current += 1
            return self._indexs[self.current - 1];


# si no se pasa limit se devolveran tantos elementos como la menor de las dimensiones
class Random3DRange(object):
    def __init__(self, shape, len_x, len_y, len_z, limit=-1):
        if shape[0] - len_x < 0:
            raise RuntimeError('dimension en x ('+ str(shape[0]) + ') es menor que la longitud en x ('+len_x + ')')

        self._x = RandomIterator(shape[0] - len_x, limit)

        if shape[1] - len_y < 0:
            raise RuntimeError('dimension en y (' + str(shape[1]) + ') es menor que la longitud en x (' + len_y + ')')

        self._y = RandomIterator(shape[1] - len_y, limit)

        if shape[2] - len_z < 0:
            raise RuntimeError('dimension en z (' + str(shape[2]) + ') es menor que la longitud en z (' + len_z + ')')

        self._z = RandomIterator(shape[2] - len_z, limit)

        self.len_x = len_x;
        self.len_y = len_y;
        self.len_z = len_z;

        self.current = 0;

    def __iter__(self):
        return self

    def next(self):  # Python 3: def __next__(self)
        x = self._x.next()
        y = self._y.next()
        z = self._z.next()
        return (x, x+self.len_x, y, y+self.len_y, z, z+self.len_z);



