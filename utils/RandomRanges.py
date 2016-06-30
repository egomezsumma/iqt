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

class Random1DRange(object):
    def __init__(self, ends, len = 1, limit=-1):

        if ends - len < 0:
            raise RuntimeError('dimension de ' + str(ends) + ' es menor que la longitud del rango de ' + len +'')

        self._x = RandomIterator(ends - len, limit)
        self.len = len;
        self.current = 0;

    def __iter__(self):
        return self

    def next(self):  # Python 3: def __next__(self)
        x = self._x.next()
        return (x, x + self.len);

class Fixed1DRange(object):
        def __init__(self, start, ends, limit=-1):
            if start > ends :
                raise RuntimeError('los valores pasados no forman un rango valido '+ str(start) + ":"+str(ends))
            self.start = start
            self.ends = ends
            self.limit=limit
            self.current = 0;

        def __iter__(self):
            return self

        def next(self):  # Python 3: def __next__(self)
            # There's no limit
            if self.limit < 0 :
                return (self.start, self.ends);


            if self.current <= self.limit:
                return (self.start, self.ends);
            else:
                raise StopIteration;



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



