"""Helper for tree construction"""
from __future__ import annotations
from typing import Tuple

from anytree import NodeMixin


class SNode(NodeMixin):
    """Structure tree node.

    Used to create a structural tree based of qubit gates.
    This is given to the initialization to rebuild as TTN.

    Attributes
    ----------
    name: str
        Node name.
    parent: Tuple[SNode]
        Parent node.
    children: Tuple[SNode]
        Children nodes.
    """

    def __init__(
        self,
        name: str,
        parent: Tuple[SNode] = None,
        children: Tuple[SNode] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.parent = parent
        if children:
            self.children = children
