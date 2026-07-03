# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    overload,
    runtime_checkable,
)

import torch
import torch.nn as nn
from typing_extensions import TypeIs, TypeVar

from aphrodite.logger import init_logger
from aphrodite.tasks import ScoreType
from aphrodite.utils.func_utils import supports_kw

if TYPE_CHECKING:
    from aphrodite.config import AphroditeConfig
    from aphrodite.config.model import AttnTypeStr
    from aphrodite.config.pooler import SequencePoolingType, TokenPoolingType
    from aphrodite.model_executor.layers.pooler import Pooler
else:
    AphroditeConfig = Any
    Pooler = Any
    SequencePoolingType = Any
    TokenPoolingType = Any
    AttnTypeStr = Any

logger = init_logger(__name__)

# The type of hidden states
# Currently, T = torch.Tensor for all models except for Medusa
# which has T = list[torch.Tensor]
T = TypeVar("T", default=torch.Tensor)
T_co = TypeVar("T_co", default=torch.Tensor, covariant=True)

# NOTE: Unlike those in `interfaces.py`, we don't define `ClassVar` tags
# for the base interfaces to avoid breaking OOT registration for existing models
# that don't inherit from the base interface classes


@runtime_checkable
class AphroditeModel(Protocol[T_co]):
    """The interface required for all models in Aphrodite."""

    def __init__(self, aphrodite_config: AphroditeConfig, prefix: str = "") -> None: ...

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings to `input_ids`."""
        ...

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> T_co: ...


def _check_aphrodite_model_init(model: type[object] | object) -> bool:
    model_init = model.__init__
    return supports_kw(model_init, "aphrodite_config")


def _check_aphrodite_model_embed_input_ids(model: type[object] | object) -> bool:
    model_embed_input_ids = getattr(model, "embed_input_ids", None)
    if not callable(model_embed_input_ids):
        logger.warning(
            "The model (%s) is missing the `embed_input_ids` method.",
            model,
        )
        return False

    return True


def _check_aphrodite_model_forward(model: type[object] | object) -> bool:
    model_forward = getattr(model, "forward", None)
    if not callable(model_forward):
        return False

    aphrodite_kws = ("input_ids", "positions")
    missing_kws = tuple(kw for kw in aphrodite_kws if not supports_kw(model_forward, kw))

    if missing_kws and (isinstance(model, type) and issubclass(model, nn.Module)):
        logger.warning(
            "The model (%s) is missing "
            "Aphrodite-specific keywords from its `forward` method: %s",
            model,
            missing_kws,
        )

    return len(missing_kws) == 0


@overload
def is_aphrodite_model(model: type[object]) -> TypeIs[type[AphroditeModel]]: ...


@overload
def is_aphrodite_model(model: object) -> TypeIs[AphroditeModel]: ...


def is_aphrodite_model(
    model: type[object] | object,
) -> TypeIs[type[AphroditeModel]] | TypeIs[AphroditeModel]:
    return (
        _check_aphrodite_model_init(model)
        and _check_aphrodite_model_embed_input_ids(model)
        and _check_aphrodite_model_forward(model)
    )


@runtime_checkable
class AphroditeModelForTextGeneration(AphroditeModel[T], Protocol[T]):
    """The interface required for all generative models in Aphrodite."""

    def compute_logits(
        self,
        hidden_states: T,
    ) -> T | None:
        """Return `None` if TP rank > 0."""
        ...


@overload
def is_text_generation_model(
    model: type[object],
) -> TypeIs[type[AphroditeModelForTextGeneration]]: ...


@overload
def is_text_generation_model(model: object) -> TypeIs[AphroditeModelForTextGeneration]: ...


def is_text_generation_model(
    model: type[object] | object,
) -> TypeIs[type[AphroditeModelForTextGeneration]] | TypeIs[AphroditeModelForTextGeneration]:
    if not is_aphrodite_model(model):
        return False

    if isinstance(model, type):
        return isinstance(model, AphroditeModelForTextGeneration)

    return isinstance(model, AphroditeModelForTextGeneration)


@runtime_checkable
class AphroditeModelForPooling(AphroditeModel[T_co], Protocol[T_co]):
    """The interface required for all pooling models in Aphrodite."""

    is_pooling_model: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports pooling.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    default_seq_pooling_type: ClassVar[SequencePoolingType] = "LAST"
    """
    Indicates the [aphrodite.config.pooler.PoolerConfig.seq_pooling_type][]
    to use by default.

    You can use the
    [aphrodite.model_executor.models.interfaces_base.default_pooling_type][]
    decorator to conveniently set this field.
    """

    default_tok_pooling_type: ClassVar[TokenPoolingType] = "ALL"
    """
    Indicates the [aphrodite.config.pooler.PoolerConfig.tok_pooling_type][]
    to use by default.

    You can use the
    [aphrodite.model_executor.models.interfaces_base.default_pooling_type][]
    decorator to conveniently set this field.
    """

    attn_type: ClassVar[AttnTypeStr] = "decoder"
    """
    Indicates the
    [aphrodite.config.model.ModelConfig.attn_type][]
    to use by default.

    You can use the
    [aphrodite.model_executor.models.interfaces_base.attn_type][]
    decorator to conveniently set this field.
    """

    score_type: ClassVar[ScoreType] = "bi-encoder"
    """
    Indicates the
    [aphrodite.config.model.ModelConfig.score_type][]
    to use by default.
    
    Scoring API handles score/rerank for:\n
    - "classify" task (score_type: cross-encoder models)\n
    - "embed" task (score_type: bi-encoder models)\n
    - "token_embed" task (score_type: late interaction models)\n
    
    score_type defaults to bi-encoder, then the Score API uses the "embed" task.\n
    If you set score_type to cross-encoder via 
    [aphrodite.model_executor.models.interfaces.SupportsCrossEncoding][], 
    then the Score API uses the "score" task.\n
    If you set score_type to late-interaction via 
    [aphrodite.model_executor.models.interfaces.SupportsLateInteraction][], 
    then the Score API uses the "token_embed" task.\n
    """

    pooler: Pooler
    """The pooler is only called on TP rank 0."""


@overload
def is_pooling_model(model: type[object]) -> TypeIs[type[AphroditeModelForPooling]]: ...


@overload
def is_pooling_model(model: object) -> TypeIs[AphroditeModelForPooling]: ...


def is_pooling_model(
    model: type[object] | object,
) -> TypeIs[type[AphroditeModelForPooling]] | TypeIs[AphroditeModelForPooling]:
    if not is_aphrodite_model(model):
        return False

    return getattr(model, "is_pooling_model", False)


_T = TypeVar("_T", bound=type[nn.Module])


def default_pooling_type(
    *,
    seq_pooling_type: SequencePoolingType = "LAST",
    tok_pooling_type: TokenPoolingType = "ALL",
):
    """Decorator to set `AphroditeModelForPooling.default_*_pooling_type`."""

    def func(model: _T) -> _T:
        model.default_seq_pooling_type = seq_pooling_type  # type: ignore
        model.default_tok_pooling_type = tok_pooling_type  # type: ignore
        return model

    return func


def get_default_seq_pooling_type(
    model: type[object] | object,
) -> SequencePoolingType:
    return getattr(model, "default_seq_pooling_type", "LAST")


def get_default_tok_pooling_type(
    model: type[object] | object,
) -> TokenPoolingType:
    return getattr(model, "default_tok_pooling_type", "ALL")


def attn_type(attn_type: AttnTypeStr):
    """Decorator to set `AphroditeModelForPooling.attn_type`."""

    def func(model: _T) -> _T:
        model.attn_type = attn_type  # type: ignore
        return model

    return func


def get_attn_type(model: type[object] | object) -> AttnTypeStr:
    return getattr(model, "attn_type", "decoder")


def get_score_type(model: type[object] | object) -> ScoreType:
    score_types = set()
    for m in model.__mro__:
        score_type = getattr(m, "score_type", "bi-encoder")
        if score_type != "bi-encoder":
            score_types.add(score_type)
    assert len(score_types) < 2
    return "bi-encoder" if not score_types else list(score_types)[0]
