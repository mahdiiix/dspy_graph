import weakref
from typing import Callable, Optional, Self

import dspy


class Node:
    _registry: weakref.WeakValueDictionary[str, "Node"] = weakref.WeakValueDictionary()

    @classmethod
    def get_node(cls, name: str) -> "Node":
        try:
            return cls._registry[name]
        except KeyError:
            raise ValueError(f"Node {name} not found")

    @classmethod
    def _register_node(cls, name: str, node: Self):
        if name in cls._registry:
            raise ValueError(f"Node {name} already exists")
        cls._registry[name] = node

    @classmethod
    def get_nodes(cls):
        return cls._registry

    def __init__(
        self,
        name: str,
        fn: Callable[..., str],
        program: Optional[dspy.Module] = None,
        program_freeze: bool = False,
    ):
        self.name = name
        self.llm_program = program
        self.run = fn
        self.program_freeze = program_freeze
        self._register_node(name, self)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


def create_node(
    name: str,
    program: Optional[dspy.Module] = None,
    freeze: bool = False,
    llm_freeze: bool = False,
):
    def decorator(fn: Callable[..., str]) -> Node:
        return Node(name, fn, program, llm_freeze)

    return decorator


END = Node("END", lambda _: "END")

if __name__ == "__main__":
    llm = dspy.Module()

    @create_node("test", llm)
    def f(state, ctx, llm_program) -> str:
        print(type(llm_program), type(state), type(ctx))
        return "something"

    print(f.name)

    @create_node("asynctest", llm)
    async def af(something: int) -> int:
        return something

    import asyncio

    print(asyncio.run(af(5)))
