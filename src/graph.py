import asyncio
import inspect
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Protocol, Union

import dspy
import pydantic

from .node import END, Node


class IsDataclass(Protocol):
    # Checking for this attribute is currently the most reliable way
    # to ascertain that something is a dataclass
    __dataclass_fields__: ClassVar[Dict[str, Any]]


class Graph:
    def __init__(
        self,
        nodes: List[Node],
        max_iterations: int = -1,
        freeze: bool = False,
        state: Dict[Any, Any] | pydantic.BaseModel | IsDataclass | None = None,
        context: Dict[Any, Any] | pydantic.BaseModel | IsDataclass | None = None,
    ):
        self.nodes = nodes
        self.start_node: Node = nodes[0]
        self.max_iterations = max_iterations
        self.freeze = freeze
        self.context = context  # Read-only object
        self.state = state  # Use async or thread lock to update
        self.alock = asyncio.Lock()
        self.tlock = threading.Lock()

    def set_start_node(self, node: Node):
        self.start_node = node

    def __call__(self):  # TODO: Node can return a list of noded too
        current_node = self.start_node
        iterations = 0
        while current_node is not END and iterations != self.max_iterations:
            bound = self._inject_params(current_node)
            current_node = Node.get_node(current_node(**bound.arguments))
            iterations += 1

        if 0 <= self.max_iterations <= iterations:
            raise RuntimeError(
                f"Graph execution exceeded max iterations ({self.max_iterations})"
            )

        return self.state

    def _inject_params(self, current_node: Node) -> inspect.BoundArguments:
        sig = inspect.signature(current_node.run)
        bound = sig.bind_partial()
        if "state" in sig.parameters:
            bound = sig.bind_partial(**bound.arguments, state=self.state)
        if "ctx" in sig.parameters:
            bound = sig.bind_partial(**bound.arguments, ctx=self.context)
        if "llm_program" in sig.parameters:
            bound = sig.bind_partial(
                **bound.arguments, llm_program=current_node.llm_program
            )
        if "alock" in sig.parameters:
            bound = sig.bind_partial(**bound.arguments, alock=self.alock)
        if "tlock" in sig.parameters:
            bound = sig.bind_partial(**bound.arguments, tlock=self.tlock)
        return sig.bind(**bound.arguments)

    def save(self, path: Path):
        pass

    def load(self, path: Path):
        pass

    def visualize(self):
        pass


class CompiledDspy(dspy.Module):
    def __init__(self, graph: Graph, result_name: str = "result"):
        super().__init__()
        self.graph = graph
        self._register_modules()
        self.result_name = result_name

    def _register_modules(self):
        if not self.graph.freeze:
            for node in self.graph.nodes:
                if (
                    (not node.freeze)
                    and (not node.llm_freeze)
                    and isinstance(node.llm_program, dspy.Module)
                ):
                    setattr(self, node.name, node.llm_program)

    def forward(self) -> Optional[dspy.Prediction]:
        self.graph()
        return (
            getattr(self.graph.state, self.result_name, None)
            if not isinstance(self.graph.state, dict)
            else self.graph.state.get(self.result_name, None)
        )
