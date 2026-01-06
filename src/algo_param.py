import weakref
from typing import Callable, Optional


class AlgoParam[T]:
    _registry: weakref.WeakValueDictionary[str, "AlgoParam"] = (
        weakref.WeakValueDictionary()
    )

    def __init__(
        self,
        name: str,
        value: Optional[T] = None,
        freeze: bool = False,
    ):
        self._value = value
        self.name = name
        self.freeze = freeze
        AlgoParam._registry[name] = self

    @classmethod
    def parameters(cls):
        yield from cls._registry.items()

    @classmethod
    def parameter(cls, name: str):
        return cls._registry[name]

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: T):
        if not self.freeze:
            self._value = value

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, value={self.value})"


if __name__ == "__main__":

    def ff(x):
        return x + 1

    a = AlgoParam[int](name="a", value=1)
    c = AlgoParam[int](value=2, name="c")

    def f():
        AlgoParam[str](value="hello", name="b")

    f()

    for item in AlgoParam.parameters():
        print(item)
