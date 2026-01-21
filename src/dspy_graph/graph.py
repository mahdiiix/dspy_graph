import asyncio
import orjson
import inspect
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, Optional, Protocol, Union

import dspy
import graphviz
import pydantic

from dspy_graph.node import END, Node, create_node
from dspy_graph.algo_param import AlgoParam


class IsDataclass(Protocol):
    # Checking for this attribute is currently the most reliable way
    # to ascertain that something is a dataclass
    __dataclass_fields__: ClassVar[dict[str, Any]]


class Graph:
    def __init__(
        self,
        nodes: Optional[list[Node]] = None,
        max_iterations: int = -1,
        freeze: bool = False,
        state: dict[Any, Any] | pydantic.BaseModel | IsDataclass | None = None,
        context: dict[Any, Any] | pydantic.BaseModel | IsDataclass | None = None,
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

    def __call__(self,
                 state: dict[Any, Any] | pydantic.BaseModel | IsDataclass | None = None,
                 context: dict[Any, Any] | pydantic.BaseModel | IsDataclass | None = None,
                 ):  # TODO: Node can return a list of noded too
        self.state = state or self.state
        self.context = context or self.context
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
            if node.program_freeze:
                node_attrs["fillcolor"] = "lightyellow"
            if self.freeze:
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

    def available_llm_programs(self) -> list[Node]:
       return [node for node in self.nodes if (node.llm_program and (not node.program_freeze))]

    def available_algo_params(self) -> list[AlgoParam]:
        return {k: v for k, v in AlgoParam.parameters() if not v.freeze}

    def freeze_algo_params(self, names: list[str]):
        pass
    def unfreeze_algo_params(self, names: list[str]):
        pass
    def freeze_llm_programs(self, names: list[str]):
        pass
    def unfreeze_llm_programs(self, names: list[str]):
        pass


class CompiledDspy(dspy.Module):
    def __init__(self, graph: Graph, result_name: str = "result"):
        super().__init__()
        self.graph = graph
        self._register_modules()
        self.result_name = result_name

    def _register_modules(self):
        if not self.graph.freeze:
            for node in self.graph.available_llm_programs():
                setattr(self, node.name, node.llm_program)

    def forward(self) -> Optional[dspy.Prediction]:
        self.graph()
        return (
            getattr(self.graph.state, self.result_name, None)
            if not isinstance(self.graph.state, dict)
            else self.graph.state.get(self.result_name, None)
        )

    def save(self, path: Path | str):
        path = Path(path)

        dspy_state = self.dump_state()
        dspy_state['metadata'] = dspy.utils.saving.get_dependency_versions()

        algo_state = dict((k, v.value) for k, v in AlgoParam.parameters())

        state = {'dspy_state': dspy_state, 'algo_state': algo_state}

        with open(path, "wb") as f:
            f.write(orjson.dumps(state, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))

    def load(self, path: Path | str):
        path = Path(path)
        with open(path, "rb") as f:
            state = orjson.loads(f.read())
        # dependency_versions = dspy.utils.saving.get_dependency_versions()
        # saved_dependency_versions = state['dspy_state']["metadata"]["dependency_versions"]
        # for key, saved_version in saved_dependency_versions.items():
        #     if dependency_versions[key] != saved_version:
        #         logger.warning(
        #             f"There is a mismatch of {key} version between saved model and current environment. "
        #             f"You saved with `{key}=={saved_version}`, but now you have "
        #             f"`{key}=={dependency_versions[key]}`. This might cause errors or performance downgrade "
        #             "on the loaded model, please consider loading the model in the same environment as the "
        #             "saving environment."
        #         )
        self.load_state(state['dspy_state'])
        for k, v in state['algo_state'].items():
            AlgoParam.parameter(k).value = v

    def reload(self, new_graph: Graph):
        self.__init__(new_graph, self.result_name)


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
