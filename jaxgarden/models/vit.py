from dataclasses import dataclass

import flax.nnx as nnx
import jax.numpy as jnp
from flax.nnx.nn.attention import dot_product_attention_weights

from jaxgarden.models.base import BaseConfig


@dataclass
class ViTConfig(BaseConfig):
    hidden_size: int = 768
    num_hidden_heads: int = 12
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    qkv_bias: bool = True
    encoder_stride: int = 16
    pooler_output_size: int | None = None
    pooler_act: str = "tanh"
    dtype: jnp.dtype = jnp.float32


class ViTPatchEmbeddings(nnx.Module):
    def __init__(
        self, config: ViTConfig, *, dtype: jnp.dtype = jnp.float32, rngs: nnx.Rngs
    ) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype

        image_size = self.config.image_size
        patch_size = self.config.patch_size
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.num_channels = self.config.num_channels
        kernel_initializer = nnx.initializers.variance_scaling(
            scale=self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
        )
        self.projection = nnx.Conv(
            in_features=self.num_channels,
            out_features=self.config.hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            dtype=self.dtype,
            kernel_init=kernel_initializer,
            rngs=rngs,
        )

    def __call__(self, pixel_values: jnp.ndarray) -> jnp.ndarray:
        num_channels = pixel_values.shape[-1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."  # noqa: E501
            )
        embeddings = self.projection(pixel_values)
        return jnp.reshape(embeddings, (embeddings.shape[0], -1, embeddings.shape[-1]))
