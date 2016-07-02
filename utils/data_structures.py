class circular_list(object):
    def __init__(self, size, init=None):
        self._data = []
        self.MAX_SIZE = size
        if init is not None:
            for i in range(size):
                self._data.append(init)

    def push(self, item):
        """
        push item to the tail
        :param item: item to insert
        :return:
        """
        if len(self._data) == self.MAX_SIZE:
            # full we have to pop the oldest item (head)
            self._data.pop(0)
        self._data.append(item)

    def pop(self):
        """
        pops the first item in the queue
        :return: head of queue
        """
        if len(self._data) == 0:
            return None
        else:
            return self._data.pop(0)

    def __iter__(self):
        self.index = 0
        return self

    def next(self):
        if self.index == len(self._data):
            raise StopIteration
        else:
            self.index += 1
            return self._data[self.index - 1]

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    def __len__(self):
        return len(self._data)


def test_circular_list():
    clist = circular_list(5)
    clist.push(1)
    clist.push(2)
    clist.push(3)
    clist.push(4)
    clist.push(5)
    clist.push(6)
    clist.push(7)

    clist[1] = 8

    assert clist[0] == 3
    assert clist[1] == 8
    assert clist[2] == 5
    assert clist[3] == 6
    assert clist[4] == 7
    assert len(clist) == 5

    clist2 = circular_list(7, 'hello')
    for item in clist2:
        assert item == 'hello'


if __name__ == '__main__':
    test_circular_list()
