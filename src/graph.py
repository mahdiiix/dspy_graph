import dspy
from typing import Optional, List, Optional, Dict, Any
import inspect

from .node import Node, END


class Graph:
    def __init__(self, nodes: List[Node], max_iterations: int = -1, freeze: bool = False,
                 state: Optional[Dict[Any, Any]]=None, context: Optional[Dict[Any, Any]]=None):
        self.nodes = nodes
        self.start_node: Node = nodes[0]
        self.max_iterations = max_iterations
        self.freeze = freeze
        self.context = context # TODO make it thread and async safe
        self.state = state # TODO make it thread and async safe

    def set_start_node(self, node: Node):
        self.start_node = node

    def __call__(self): # TODO: Node can return a list of noded too
        current_node = self.start_node
        iterations = 0
        while current_node is not END and iterations != self.max_iterations:
            bound = self._inject_params(current_node)
            current_node = Node.get_node(current_node(**bound.arguments))
            iterations += 1

        if iterations >= self.max_iterations:
            raise RuntimeError(f"Graph execution exceeded max iterations ({self.max_iterations})")

    def _inject_params(self, current_node: Node) -> inspect.BoundArguments:
        sig = inspect.signature(current_node.run)
        bound = sig.bind_partial()
        if "state" in sig.parameters:
            bound = sig.bind_partial(**bound.arguments, state=self.state)
        if "ctx" in sig.parameters:
            bound = sig.bind_partial(**bound.arguments, ctx=self.context)
        if 'llm_program' in sig.parameters:
            bound = sig.bind_partial(**bound.arguments, llm_program = current_node.llm_program)
        return sig.bind(**bound.arguments)


class CompiledGraph(dspy.Module):
    def __init__(self, graph: Graph):
        super().__init__()
        self.graph = graph
        self._register_modules()

    def _register_modules(self):
        if not self.graph.freeze:
            for node in self.graph.nodes:
                if (not node.freeze) and (not node.llm_freeze) and isinstance(node.llm_program, dspy.Module):
                    setattr(self, node.name, node.llm_program)

    def forward(self):
        self.graph()