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


class ViTEmbeddings(nnx.Module):
    def __init__(
        self, config: ViTConfig, *, dtype: jnp.dtype = jnp.float32, rngs: nnx.Rngs
    ) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype

        cls_token_initializer = nnx.initializers.variance_scaling(
            scale=self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
        )
        cls_token_value = cls_token_initializer(
            key=rngs.params(), shape=(1, 1, self.config.hidden_size), dtype=self.dtype
        )
        self.cls_token = nnx.Param(value=cls_token_value)

        self.patch_embeddings = ViTPatchEmbeddings(config, dtype=dtype, rngs=rngs)

        position_embeddings_initializer = nnx.initializers.variance_scaling(
            scale=self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
        )
        position_embeddings_value = position_embeddings_initializer(
            key=rngs.params(),
            shape=(1, self.patch_embeddings.num_patches + 1, self.config.hidden_size),
            dtype=self.dtype,
        )
        self.position_embeddings = nnx.Param(position_embeddings_value)

        self.dropout = nnx.Dropout(rate=self.config.hidden_dropout_prob, rngs=rngs)

    def __call__(self, pixel_values: jnp.ndarray, rngs: nnx.Rngs | None = None) -> jnp.ndarray:
        embeddings = self.patch_embeddings(pixel_values=pixel_values)
        cls_tokens = jnp.broadcast_to(
            self.cls_token.value, shape=(pixel_values.shape[0], 1, self.config.hidden_size)
        )
        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)
        embeddings = embeddings + self.position_embeddings.value
        embeddings = self.dropout(embeddings, rngs=rngs)
        return embeddings


class ViTSelfAttention(nnx.Module):
    def __init__(
        self, config: ViTConfig, *, dtype: jnp.dtype = jnp.float32, rngs: nnx.Rngs
    ) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype

        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`:"  # noqa: E501
                " {self.config.num_attention_heads}"
            )

        self.query = nnx.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            use_bias=config.qkv_bias,
            kernel_init=nnx.initializers.variance_scaling(
                scale=self.config.initializer_range**2,
                mode="fan_in",
                distribution="truncated_normal",
            ),
            dtype=dtype,
            rngs=rngs,
        )
        self.key = nnx.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            use_bias=config.qkv_bias,
            dtype=dtype,
            kernel_init=nnx.initializers.variance_scaling(
                scale=self.config.initializer_range**2,
                mode="fan_in",
                distribution="truncated_normal",
            ),
            rngs=rngs,
        )
        self.value = nnx.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            use_bias=config.qkv_bias,
            dtype=dtype,
            kernel_init=nnx.initializers.variance_scaling(
                scale=self.config.initializer_range**2,
                mode="fan_in",
                distribution="truncated_normal",
            ),
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        deterministic: bool = True,
        rngs: nnx.Rngs | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        query_states = self.query(hidden_states)
        query_states = query_states.reshape(
            (*hidden_states.shape[:2], self.config.num_attention_heads, head_dim)
        )

        value_states = self.value(hidden_states)
        value_states = value_states.reshape(
            (*hidden_states.shape[:2], self.config.num_attention_heads, head_dim)
        )

        key_states = self.key(hidden_states)
        key_states = key_states.reshape(
            (*hidden_states.shape[:2], self.config.num_attention_heads, head_dim)
        )

        attention_weights = dot_product_attention_weights(
            query=query_states,
            key=key_states,
            dropout_rate=self.config.attention_probs_dropout_prob if not deterministic else 0.0,
            dropout_rng=rngs.dropout() if rngs is not None and not deterministic else None,
            deterministic=deterministic,
            dtype=self.dtype,
        )

        attention_output = jnp.einsum("...hqk,...khd->...qhd", attention_weights, value_states)
        attention_output = attention_output.reshape((*attention_output.shape[:2], -1))

        return attention_output, attention_weights
