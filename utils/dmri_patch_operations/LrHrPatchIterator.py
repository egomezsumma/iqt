

class LrHrPatchIterator(object):

    def __init__(self, shapeLrImg, n, m):
        self.n = n;
        self.m = m;
        self.shapeLrImg = shapeLrImg
        self.patchN1 = 2 * n + 1
        self.patchN2 = m
        self._current = 0;
        self.indexes_ranges = [];
        self._calculate_indexs()

    def __iter__(self):
        return self

    def next(self):  # Python 3: def __next__(self)
        if self._current >= len(self.indexes_ranges):
            raise StopIteration
        else:
            self._current += 1
            return self.indexes_ranges[self._current - 1];


    def _calculate_indexs(self):
        for i in range(0, self.shapeLrImg[0] - self.patchN1 + 1):
            for j in range(0, self.shapeLrImg[1] - self.patchN1 + 1):
                for k in range(0, self.shapeLrImg[2] - self.patchN1 + 1):

                    #sum = sum + 1
                    # Coordinate voxels in HR
                    x = (i + self.n) * self.patchN2;
                    y = (j + self.n) * self.patchN2;
                    z = (k + self.n) * self.patchN2;

                    #unos = np.ones((self.patchN2, self.patchN2, self.patchN2))
                    #sum_acum = sum_acum + unos.sum()
                    #hrimg[x:x + patchN2, y:y + patchN2, z:z + patchN2] = unos

                    #print i, ':', i + self.patchN1, j, ':', j + self.patchN1, k, ':', k + self.patchN1, ' -->', \
                    #    x, ':', x + self.patchN2, y, ':', y + self.patchN2, z, ':', z + self.patchN2;

                    res = {}
                    res['lr'] = (i, i + self.patchN1, j, j + self.patchN1, k,  k + self.patchN1);
                    res['hr'] = (x, x + self.patchN2, y, y + self.patchN2, z, z + self.patchN2);

                    self.indexes_ranges.append(res);
