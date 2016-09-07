

class DmriPatchIterator(object):

    def __init__(self, rangeX, rangeY, rangeZ):
        self.rangeX = rangeX;
        self.rangeY = rangeY;
        self.rangeZ = rangeZ;
        self._currentX = 0
        self._currentY = 0
        self._currentZ = 0
        self._stops = False

    def __iter__(self):
        return self

    def next(self):  # Python 3: def __next__(self)
        if self._stops :
            raise StopIteration

        nX, nY, nZ = self.rangeX[self._currentX], self.rangeY[self._currentY], self.rangeZ[self._currentZ]

        self._currentX = (self._currentX + 1) % len(self.rangeX)
        if self._currentX == 0 :
            self._currentY = (self._currentY + 1) % len(self.rangeY)
            if self._currentY == 0:
                self._currentZ = (self._currentZ + 1) % len(self.rangeZ)
                if self._currentZ == 0:
                    self._stops = True
        return nX, nY, nZ