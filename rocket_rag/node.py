"""
A node structure for hybrid vector search
The node can include sparse and dense vector

Sparse vector is represented as a dictionary in the format of {indices: values},
Dense vector is represented as a list of float

The implementation is referred as the source code from LlamaIndex: 
https://github.com/run-llama/llama_index/blob/main/llama_index/schema.py
"""

import json
import uuid
import loguru

from abc import abstractmethod
from enum import Enum, auto
from typing import List, Dict, Any, Optional
from typing_extensions import Self
import numpy as np
from pydantic import BaseModel, Field

from typing import List, Optional


DEFAULT_TEXT_NODE_TMPL = "{metadata_str}\n\n{content}"
DEFAULT_METADATA_TMPL = "{key}: {value}"


class MetadataMode(str, Enum):
    ALL = auto()
    EMBED = auto()
    LLM = auto()
    NONE = auto()


class BaseComponent(BaseModel):
    """Base component object to capture class names."""

    class Config:
        @staticmethod
        def json_schema_extra(schema: Dict[str, Any], model: "BaseComponent") -> None:
            """Add class name to schema."""
            schema["properties"]["class_name"] = {
                "title": "Class Name",
                "type": "string",
                "default": model.class_name(),
            }

    @classmethod
    def class_name(cls) -> str:
        """
        Get the class name, used as a unique ID in serialization.

        This provides a key that makes serialization robust against actual class
        name changes.
        """
        return "base_component"

    def json(self, **kwargs: Any) -> str:
        return self.to_json(**kwargs)

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        data = super().dict(**kwargs)
        data["class_name"] = self.class_name()
        return data

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()

        # tiktoken is not pickleable
        # state["__dict__"] = self.dict()
        state["__dict__"].pop("tokenizer", None)

        # remove local functions
        keys_to_remove = []
        for key, val in state["__dict__"].items():
            if key.endswith("_fn"):
                keys_to_remove.append(key)
            if "<lambda>" in str(val):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            state["__dict__"].pop(key, None)

        # remove private attributes -- kind of dangerous
        state["__private_attribute_values__"] = {}

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # Use the __dict__ and __init__ method to set state
        # so that all variable initialize
        try:
            self.__init__(**state["__dict__"])  # type: ignore
        except Exception:
            # Fall back to the default __setstate__ method
            super().__setstate__(state)

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        data = self.dict(**kwargs)
        data["class_name"] = self.class_name()
        return data

    def to_json(self, **kwargs: Any) -> str:
        data = self.to_dict(**kwargs)
        return json.dumps(data)

    # TODO: return type here not supported by current mypy version
    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> Self:  # type: ignore
        if isinstance(kwargs, dict):
            data.update(kwargs)

        data.pop("class_name", None)
        return cls(**data)

    @classmethod
    def from_json(cls, data_str: str, **kwargs: Any) -> Self:  # type: ignore
        data = json.loads(data_str)
        return cls.from_dict(data, **kwargs)
    

# Node classes for indexes
class BaseNode(BaseComponent):
    """Base node Object.

    Generic abstract interface for retrievable nodes

    """

    class Config:
        populate_by_name = True

    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique ID of the node."
    )
    rocket_feature: Optional[List[Any]] = Field(
        default=None, description="Rocket feature of the node."
    )
    dense_embedding: Optional[List[Any]] = Field(
        default=None, description="Dense embedding of the node."
    )
    sparse_dict: Optional[Dict[str, List[Any]]] = Field(
        default=None, description="Dictionary of sparse embedding of the node."
    )


    """"
    metadata fields
    - injected as part of the text shown to LLMs as context
    - injected as part of the text for generating embeddings
    - used by vector DBs for metadata filtering

    """
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="A flat dictionary of metadata fields",
        alias="extra_info",
    )
    excluded_embed_metadata_keys: List[str] = Field(
        default_factory=list,
        description="Metadata keys that are excluded from text for the embed model.",
    )
    excluded_llm_metadata_keys: List[str] = Field(
        default_factory=list,
        description="Metadata keys that are excluded from text for the LLM.",
    )
    hash: str = Field(default="", description="Hash of the node content.")

    # @classmethod
    # @abstractmethod
    # def get_type(cls) -> str:
    #     """Get Object type."""

    @abstractmethod
    def get_content(self, metadata_mode: MetadataMode = MetadataMode.ALL) -> str:
        """Get object content."""

    @abstractmethod
    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        """Metadata string."""

    @abstractmethod
    def set_content(self, value: Any) -> None:
        """Set the content of the node."""

    @property
    def node_id(self) -> str:
        return self.id_

    @node_id.setter
    def node_id(self, value: str) -> None:
        self.id_ = value

    def get_rocket_feature(self) -> List[float]:
        """Get rocket features

        Error if rocket features is None    
        """
        if self.rocket_feature is None:
            raise ValueError("rocket feature not set.")
        return self.rocket_feature

    def get_embedding(self) -> List[float]:
        """Get dense embedding.

        Errors if embedding is None.
        """
        if self.dense_embedding is None:
            raise ValueError("embedding not set.")
        return self.dense_embedding

    def get_sparse_dict(self) -> Dict[str, float]:
        """Get sparse dict.

        Errors if sparse dictionary is None.
        """
        if self.sparse_dict is None:
            raise ValueError("sparse_dict not set.")
        return self.sparse_dict


class Node(BaseNode):
    text: str = Field(default="", description="Text content of the node.")
    start_char_idx: Optional[int] = Field(
        default=None, description="Start char index of the node."
    )
    end_char_idx: Optional[int] = Field(
        default=None, description="End char index of the node."
    )
    text_template: str = Field(
        default=DEFAULT_TEXT_NODE_TMPL,
        description=(
            "Template for how text is formatted, with {content} and "
            "{metadata_str} placeholders."
        ),
    )
    metadata_template: str = Field(
        default=DEFAULT_METADATA_TMPL,
        description=(
            "Template for how metadata is formatted, with {key} and "
            "{value} placeholders."
        ),
    )
    metadata_seperator: str = Field(
        default="\n",
        description="Separator between metadata fields when converting to string.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "TextNode"

    # @root_validator
    # def _check_hash(cls, values: dict) -> dict:
    #     """Generate a hash to represent the node."""
    #     text = values.get("text", "")
    #     metadata = values.get("metadata", {})
    #     doc_identity = str(text) + str(metadata)
    #     values["hash"] = str(
    #         sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest()
    #     )
    #     return values

    # @classmethod
    # def get_type(cls) -> str:
    #     """Get Object type."""
    #     return ObjectType.TEXT

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        """Get object content."""
        metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
        if not metadata_str:
            return self.text

        return self.text_template.format(
            content=self.text, metadata_str=metadata_str
        ).strip()

    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        """Metadata info string."""
        if mode == MetadataMode.NONE:
            return ""

        usable_metadata_keys = set(self.metadata.keys())
        if mode == MetadataMode.LLM:
            for key in self.excluded_llm_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)
        elif mode == MetadataMode.EMBED:
            for key in self.excluded_embed_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)

        return self.metadata_seperator.join(
            [
                self.metadata_template.format(key=key, value=str(value))
                for key, value in self.metadata.items()
                if key in usable_metadata_keys
            ]
        )

    def set_content(self, value: str) -> None:
        """Set the content of the node."""
        self.text = value

    def get_node_info(self) -> Dict[str, Any]:
        """Get node info."""
        return {"start": self.start_char_idx, "end": self.end_char_idx}

    def get_text(self) -> str:
        return self.get_content(metadata_mode=MetadataMode.NONE)


if __name__ == "__main__":
    loguru.logger.debug(f'Testing on node construction...')
    node = Node()
    print(node)
    loguru.logger.info(f'Node construction successful!')
