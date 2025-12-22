import weakref
from typing import Callable, Optional

type MutationFunc[T] = Callable[[T], T]

class AlgoParam[T]:
    _registry: weakref.WeakSet["AlgoParam"] = weakref.WeakSet()

    def __init__(self, mutate: MutationFunc[T], value: T, name: Optional[str] = None, freeze: bool = False):
        self.mutate = mutate
        self.value = value
        self.name = name
        self.freeze = freeze
        AlgoParam._registry.add(self)

    @classmethod
    def parameters(cls):
        yield from cls._registry

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, value={self.value}, mutate_func={getattr(self.mutate, '__name__', None)})"

if __name__ == "__main__":
    def ff(x): return x + 1
    a = AlgoParam[int](ff, name="a", value=1)
    c = AlgoParam[int](lambda x: x * 2, value=2, name="c")
    def f():
        return AlgoParam[str](lambda x: x * 2, value="hello", name="b")
    f()

    for item in AlgoParam.parameters():
        print(item)