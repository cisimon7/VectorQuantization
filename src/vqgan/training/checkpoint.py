"""Serialize trainable arrays, optimizer moments, and step together."""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import equinox as eqx

from vqgan.config import DiscriminatorConfig, ModelConfig

from .state import TrainState


def _read_metadata(stream):
    metadata = json.loads(stream.readline())
    if metadata["format_version"] not in (1, 2):
        raise ValueError("unsupported checkpoint format")
    mode = "vqvae" if metadata["discriminator"] is None else "vqgan"
    if metadata.get("mode", mode) != mode:
        raise ValueError("checkpoint mode does not match its discriminator configuration")
    # Version 1 contains the same array payload; infer its implicit GAN mode.
    return {**metadata, "format_version": 2, "mode": mode}


def read_checkpoint_configs(path: str | Path) -> tuple[ModelConfig, DiscriminatorConfig | None]:
    """Read architecture metadata to construct a matching restore template."""
    with Path(path).open("rb") as stream:
        metadata = _read_metadata(stream)
    generator = metadata["generator"]
    generator["channel_multipliers"] = tuple(generator["channel_multipliers"])
    discriminator = metadata["discriminator"]
    return ModelConfig(**generator), (
        DiscriminatorConfig(**discriminator) if discriminator is not None else None
    )


def _metadata(state):
    return {
        "format_version": 2,
        "mode": state.mode,
        "generator": asdict(state.generator.config),
        "discriminator": (
            asdict(state.discriminator.config) if state.discriminator is not None else None
        ),
    }


def save_checkpoint(path: str | Path, state: TrainState):
    """Atomic checkpoint; use the same optimizer configuration when resuming."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((json.dumps(_metadata(state)) + "\n").encode())
            eqx.tree_serialise_leaves(stream, state)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: str | Path, template: TrainState) -> TrainState:
    """The template supplies static structure; mismatched model configs fail early."""
    with Path(path).open("rb") as stream:
        metadata = _read_metadata(stream)
        expected = json.loads(json.dumps(_metadata(template)))
        if metadata != expected:
            raise ValueError("checkpoint model configuration does not match the template")
        return eqx.tree_deserialise_leaves(stream, template)
