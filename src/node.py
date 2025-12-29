from typing import Optional, Union, Callable, Self
import dspy

class Node:
    _registry: dict[str, Self] = {}

    @classmethod
    def get_node(cls, name: str) -> Self:
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
        yield from cls._registry.items()

    def __init__(self, name: str, fn: Callable[..., str], program: Optional[dspy.Module] = None, freeze: bool = False):
        self.name = name
        self.llm_program = program
        self.run = fn
        self.freeze = freeze
        self._register_node(name, self)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

def create_node(name: str, program: Optional[dspy.Module] = None):
    def decorator(fn: Callable[..., str]):
        return Node(name, fn, program)
    return decorator

END = Node("END", lambda _: "END", freeze=True)

if __name__ == "__main__":
    llm = dspy.Module()
    @create_node("test", llm)
    def f(state, ctx, llm_program) -> int:
       print(type(program), type(state), type(ctx))
       return 'something'
    print(f.name)

    @create_node("asynctest", llm)
    async def af(something: int) -> int:
        return something
    import asyncio
    print(asyncio.run(af(5)))