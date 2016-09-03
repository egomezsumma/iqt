import numpy as np

class parray(object):

    @staticmethod
    def from_file(filename):
        arr = np.loadtxt(filename)
        return parray(filename, list(arr))

    def __init__(self, filename, array=[]):
        self.array = list(array)
        self.filename = filename

    def append(self, elem):
        self.array.append(elem)
        self.persist()

    def persist(self):
        np.savetxt(self.filename, self.array)
        #print 'persistiendo en %s'% self.filename

    def __len__(self):
        return len(self.array)

    def __getitem__(self, item):
        result = self.array[item]
        return result

    def __getattr__(self, item):
        result = getattr(self.array, item)
        if callable(result):
            result = result()
        return result

    def __str__(self):
        return str(self.array)

    def __add__(self, other):
        self.array = self.array + other
        self.persist()
        return self

    def asnumpy(self):
        return np.array(self.array)


"""
a = parray('./archivo.txt')
a.append(1)
a+[4.0e-3, 3]
print a, 'saasas'
b = parray.from_file('./archivo.txt')
b.append(890000.8)
print 'b=', b[1], len(b)
"""