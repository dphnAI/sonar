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

from sonar.logger import init_logger
from sonar.tasks import ScoreType
from sonar.utils.func_utils import supports_kw

if TYPE_CHECKING:
    from sonar.config import SonarConfig
    from sonar.config.model import AttnTypeStr
    from sonar.config.pooler import SequencePoolingType, TokenPoolingType
    from sonar.model_executor.layers.pooler import Pooler
else:
    SonarConfig = Any
    Pooler = Any
    SequencePoolingType = Any
    TokenPoolingType = Any
    AttnTypeStr = Any

logger = init_logger(__name__)

# NOTE: Unlike those in `interfaces.py`, we don't define `ClassVar` tags
# for the base interfaces to avoid breaking OOT registration for existing models
# that don't inherit from the base interface classes


@runtime_checkable
class SonarModel(Protocol):
    """The interface required for all models in Sonar."""

    def __init__(self, sonar_config: SonarConfig, prefix: str = "") -> None: ...

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings to `input_ids`."""
        ...

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor: ...

    def is_unused_checkpoint_weight(self, name: str) -> bool:
        """Return True for checkpoint weights this model does not load.

        Defining this method is optional for satisfying this interface.
        Override with the same rule used in `load_weights`. Models without
        a selection rule keep every checkpoint weight by default.
        """
        return False


def _check_sonar_model_init(model: type[object] | object) -> bool:
    model_init = getattr(model, "__init__")  # noqa: B009
    return supports_kw(model_init, "sonar_config")


def _check_sonar_model_embed_input_ids(model: type[object] | object) -> bool:
    model_embed_input_ids = getattr(model, "embed_input_ids", None)
    if not callable(model_embed_input_ids):
        logger.warning(
            "The model (%s) is missing the `embed_input_ids` method.",
            model,
        )
        return False

    return True


def _check_sonar_model_forward(model: type[object] | object) -> bool:
    model_forward = getattr(model, "forward", None)
    if not callable(model_forward):
        return False

    sonar_kws = ("input_ids", "positions")
    missing_kws = tuple(kw for kw in sonar_kws if not supports_kw(model_forward, kw))

    if missing_kws and (isinstance(model, type) and issubclass(model, nn.Module)):
        logger.warning(
            "The model (%s) is missing Sonar-specific keywords from its `forward` method: %s",
            model,
            missing_kws,
        )

    return len(missing_kws) == 0


@overload
def is_sonar_model(model: type[object]) -> TypeIs[type[SonarModel]]: ...


@overload
def is_sonar_model(model: object) -> TypeIs[SonarModel]: ...


def is_sonar_model(
    model: type[object] | object,
) -> TypeIs[type[SonarModel]] | TypeIs[SonarModel]:
    return (
        _check_sonar_model_init(model)
        and _check_sonar_model_embed_input_ids(model)
        and _check_sonar_model_forward(model)
    )


@runtime_checkable
class SonarModelForTextGeneration(SonarModel, Protocol):
    """The interface required for all generative models in Sonar."""

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return `None` if TP rank > 0."""
        ...


@overload
def is_text_generation_model(
    model: type[object],
) -> TypeIs[type[SonarModelForTextGeneration]]: ...


@overload
def is_text_generation_model(model: object) -> TypeIs[SonarModelForTextGeneration]: ...


def is_text_generation_model(
    model: type[object] | object,
) -> TypeIs[type[SonarModelForTextGeneration]] | TypeIs[SonarModelForTextGeneration]:
    if not is_sonar_model(model):
        return False

    # Optional protocol defaults are not inherited by structural models.
    return callable(getattr(model, "compute_logits", None))


@runtime_checkable
class SonarModelForPooling(SonarModel, Protocol):
    """The interface required for all pooling models in Sonar."""

    is_pooling_model: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports pooling.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    default_seq_pooling_type: ClassVar[SequencePoolingType] = "LAST"
    """
    Indicates the [sonar.config.pooler.PoolerConfig.seq_pooling_type][]
    to use by default.

    You can use the
    [sonar.model_executor.models.interfaces_base.default_pooling_type][]
    decorator to conveniently set this field.
    """

    default_tok_pooling_type: ClassVar[TokenPoolingType] = "ALL"
    """
    Indicates the [sonar.config.pooler.PoolerConfig.tok_pooling_type][]
    to use by default.

    You can use the
    [sonar.model_executor.models.interfaces_base.default_pooling_type][]
    decorator to conveniently set this field.
    """

    attn_type: ClassVar[AttnTypeStr] = "decoder"
    """
    Indicates the
    [sonar.config.model.ModelConfig.attn_type][]
    to use by default.

    You can use the
    [sonar.model_executor.models.interfaces_base.attn_type][]
    decorator to conveniently set this field.
    """

    score_type: ClassVar[ScoreType] = "bi-encoder"
    """
    Indicates the
    [sonar.config.model.ModelConfig.score_type][]
    to use by default.

    Scoring API handles score/rerank for:

    - "classify" task (score_type: cross-encoder models)
    - "embed" task (score_type: bi-encoder models)
    - "token_embed" task (score_type: late interaction models)

    score_type defaults to bi-encoder, then the Score API uses the "embed" task.
    If you set score_type to cross-encoder via
    [sonar.model_executor.models.interfaces.SupportsCrossEncoding][],
    then the Score API uses the "score" task.
    If you set score_type to late-interaction via
    [sonar.model_executor.models.interfaces.SupportsLateInteraction][],
    then the Score API uses the "token_embed" task.
    """

    pooler: Pooler
    """The pooler is only called on TP rank 0."""


@overload
def is_pooling_model(model: type[object]) -> TypeIs[type[SonarModelForPooling]]: ...


@overload
def is_pooling_model(model: object) -> TypeIs[SonarModelForPooling]: ...


def is_pooling_model(
    model: type[object] | object,
) -> TypeIs[type[SonarModelForPooling]] | TypeIs[SonarModelForPooling]:
    if not is_sonar_model(model):
        return False

    return getattr(model, "is_pooling_model", False)


_T = TypeVar("_T", bound=type[nn.Module])


def default_pooling_type(
    *,
    seq_pooling_type: SequencePoolingType = "LAST",
    tok_pooling_type: TokenPoolingType = "ALL",
):
    """Decorator to set `SonarModelForPooling.default_*_pooling_type`."""

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
    """Decorator to set `SonarModelForPooling.attn_type`."""

    def func(model: _T) -> _T:
        model.attn_type = attn_type  # type: ignore
        return model

    return func


def get_attn_type(model: type[object] | object) -> AttnTypeStr:
    return getattr(model, "attn_type", "decoder")


def get_score_type(model: type[object] | object) -> ScoreType:
    score_types: set[ScoreType] = set()
    model_cls = model if isinstance(model, type) else type(model)
    for m in model_cls.__mro__:
        score_type: ScoreType = getattr(m, "score_type", "bi-encoder")
        if score_type != "bi-encoder":
            score_types.add(score_type)
    assert len(score_types) < 2
    return "bi-encoder" if not score_types else next(iter(score_types))
