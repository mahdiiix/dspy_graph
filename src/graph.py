import asyncio
import inspect
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Protocol, Union

import dspy
import graphviz
import pydantic

from node import END, Node, create_node


class IsDataclass(Protocol):
    # Checking for this attribute is currently the most reliable way
    # to ascertain that something is a dataclass
    __dataclass_fields__: ClassVar[Dict[str, Any]]


class Graph:
    def __init__(
        self,
        nodes: Optional[List[Node]] = None,
        max_iterations: int = -1,
        freeze: bool = False,
        state: Dict[Any, Any] | pydantic.BaseModel | IsDataclass | None = None,
        context: Dict[Any, Any] | pydantic.BaseModel | IsDataclass | None = None,
    ):
        self.nodes = nodes if nodes else list(Node.get_nodes().values())
        self.start_node: Node = self.nodes[0]
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

    @staticmethod
    def _extract_next_nodes(node: Node):
        next_nodes = []
        for l in inspect.getsourcelines(node.run)[0]:
            matches = re.match(r"^\s*return\s+(.+?)(?:\s+#.*)?$", l)
            if matches:
                if matches[1] == "END":
                    next_nodes.append("END")
                else:
                    next_nodes.append(eval(matches[1]))
        return next_nodes

    def _create_mapping_nodes(self) -> defaultdict[Any, list]:
        # Build visual dictionary mapping nodes to their next nodes
        mapping = defaultdict(list)
        for node in self.nodes:
            if node is END:
                continue
            mapping[node.name].extend(self._extract_next_nodes(node))
        return mapping

    def visualize(
        self,
        filepath: Optional[str | Path] = None,
        format: str = "png",
        view: bool = False,
    ):
        """
        Visualize the graph structure using graphviz.

        Args:
            filepath: Output file path (without extension).
            format: Output format (png, pdf, svg, etc.)
            view: Whether to open the rendered graph automatically
        """

        mapping = self._create_mapping_nodes()

        # Create graphviz Digraph
        dot = graphviz.Digraph(comment="Graph Visualization")
        dot.attr(rankdir="TB")

        # Add all nodes with styling
        for node in self.nodes:
            if node is END:
                continue

            node_attrs = {
                "shape": "box",
                "style": "rounded,filled",
                "fillcolor": "lightblue",
            }

            # Highlight start node
            if node == self.start_node:
                node_attrs["color"] = "lightgreen"
                node_attrs["penwidth"] = "3"

            # Mark frozen nodes
            if node.llm_freeze:
                node_attrs["fillcolor"] = "lightyellow"
            if node.freeze or self.freeze:
                node_attrs["fillcolor"] = "lightgray"

            dot.node(node.name, node.name, **node_attrs)

        # Add END node
        dot.node("END", "END", shape="box", style="rounded,filled", color="salmon")

        # Add edges from visual dictionary
        for source_name, targets in mapping.items():
            for target in targets:
                dot.edge(source_name, target)

        # Save and/or view
        if filepath:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            dot.render(
                filepath.stem,
                directory=filepath.parent or ".",
                format=format,
                view=view,
                cleanup=True,
            )

        print(dot.source)


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


if __name__ == "__main__":
    # Create simple test nodes using decorator
    CHECK = "check"

    @create_node("start")
    def start_node():
        print("Starting...")
        if True:
            return "process"
        else:
            return "check"

    @create_node("process", llm_freeze=True)
    def process_node():
        print("Processing...")
        return f"{CHECK}"
        return "start"

    @create_node("check", freeze=True)
    def check_node():
        print("Checking...")
        return "END"

    # Create graph
    test_graph = Graph(
        #nodes=[start_node, process_node, check_node],
        max_iterations=10,
        state={"counter": 0},
    )
    test_graph.set_start_node(start_node)

    # Visualize the graph
    print("Creating graph visualization...")
    test_graph.visualize("/Users/mahdimajd/test_graph", format="png", view=True)
    print("Graph visualization saved as test_graph.png")
