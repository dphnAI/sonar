---
title: Adding a new model
---

This document provides a high-level guide on integrating a Hugging Face transformers model into Aphrodite Engine.

The complexity of adding a new model depends heavily on the model's architecture. The process is straightforward if the model shares a similar architecture with an existing model in Aphrodite. However, for models that include new operators (e.g. a new attention mechanism), the process can be a bit more complex.

By default, Aphrodite models do not support multi-modal inputs. We have separate guide for enabling that after implementing the model here.

:::tip
If you're having problems implementing the model, feel free to open an issue on the GitHub repo. We'll be happy to help if we can!
:::

## Step 0: Fork the Aphrodite Repository

Start by forking our [GitHub repository](https://github.com/PygmalionAI/aphrodite-engine) and the build it from source. This gives you the ability to modify the source code and test your model.

## 1. Bring your model code

First, clone the PyTorch model code from the source repository.
For instance, Aphrodite's [OPT model](https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/opt.py) was adapted from
HuggingFace's [modeling_opt.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py) file.

:::warning
Make sure to review and adhere to the original code's copyright and licensing terms!
:::

## 2. Make your code compatible with Aphrodite

To ensure compatibility with Aphrodite, your model must meet the following requirements:

### Initialization Code

All Aphrodite modules within the model must include a `prefix` argument in their constructor. This `prefix` is typically the full name of the module in the model's state dictionary and is crucial for:

- Runtime support: Aphrodite's attention operators are registered in a model's state by their full names. Each attention operator must have a unique prefix as its layer name to avoid conflicts.
- Non-uniform quantization support: A quantized checkpoint can selectively quantize certain layers while keeping others in full precision. By providing the `prefix` during initialization, Aphrodite can match the current layer's `prefix` with the quantization configuration to determine if the layer should be initialized in quantized mode.

The initialization code should look like this:

```python
from torch import nn
from aphrodite.config import AphroditeConfig
from aphrodite.attention import Attention

class MyAttention(nn.Module):
    def __init__(self, aphrodite_config: AphroditeConfig, prefix: str):
        super().__init__()
        self.attn = Attention(prefix=f"{prefix}.attn")

class MyDecoderLayer(nn.Module):
    def __init__(self, aphrodite_config: AphroditeConfig, prefix: str):
        super().__init__()
        self.self_attn = MyAttention(prefix=f"{prefix}.self_attn")

class MyModel(nn.Module):
    def __init__(self, aphrodite_config: AphroditeConfig, prefix: str):
        super().__init__()
        self.layers = nn.ModuleList(
            [MyDecoderLayer(aphrodite_config, prefix=f"{prefix}.layers.{i}") for i in range(aphrodite_config.model_config.hf_config.num_hidden_layers)]
        )

class MyModelForCausalLM(nn.Module):
    def __init__(self, aphrodite_config: AphroditeConfig, prefix: str = ""):
        super().__init__()
        self.model = MyModel(aphrodite_config, prefix=f"{prefix}.model")
```

### Computation Code

- Add a `get_input_embeddings` method inside `MyModel` module that returns the text embeddings given `input_ids`. This is equivalent to directly calling the text embedding layer, but provides a unified interface in case `MyModel` is used within a composite multimodal model.

```python
class MyModel(nn.Module):
        ...

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        ... 
```

- Rewrite the [forward][torch.nn.Module.forward] method of your model to remove any unnecessary code, such as training-specific code. Modify the input parameters to treat `input_ids` and `positions` as flattened tensors with a single batch size dimension, without a max-sequence length dimension.

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: Optional[IntermediateTensors] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ...
```

:::note
Currently, Aphrodite supports the basic multi-head attention mechanism and its variant with rotary positional embeddings.
If your model employs a different attention mechanism, you will need to implement a new attention layer in Aphrodite.
:::

For reference, check out our [Llama implementation](https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/llama.py). Aphrodite already supports a large number of models. It is recommended to find a model similar to yours and adapt it to your model's architecture. Check out <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/> for more examples.

## 3. (Optional) Implement tensor parallelism and quantization support

If your model is too large to fit into a single GPU, you can use tensor parallelism to manage it.
To do this, substitute your model's linear and embedding layers with their tensor-parallel versions.
For the embedding layer, you can simply replace [torch.nn.Embedding][] with `VocabParallelEmbedding`. For the output LM head, you can use `ParallelLMHead`.
When it comes to the linear layers, we provide the following options to parallelize them:

- `ReplicatedLinear`: Replicates the inputs and weights across multiple GPUs. No memory saving.
- `RowParallelLinear`: The input tensor is partitioned along the hidden dimension. The weight matrix is partitioned along the rows (input dimension). An *all-reduce* operation is performed after the matrix multiplication to reduce the results. Typically used for the second FFN layer and the output linear transformation of the attention layer.
- `ColumnParallelLinear`: The input tensor is replicated. The weight matrix is partitioned along the columns (output dimension). The result is partitioned along the column dimension. Typically used for the first FFN layer and the separated QKV transformation of the attention layer in the original Transformer.
- `MergedColumnParallelLinear`: Column-parallel linear that merges multiple `ColumnParallelLinear` operators. Typically used for the first FFN layer with weighted activation functions (e.g., SiLU). This class handles the sharded weight loading logic of multiple weight matrices.
- `QKVParallelLinear`: Parallel linear layer for the query, key, and value projections of the multi-head and grouped-query attention mechanisms. When number of key/value heads are less than the world size, this class replicates the key/value heads properly. This class handles the weight loading and replication of the weight matrices.

Note that all the linear layers above take `linear_method` as an input. Aphrodite will set this parameter according to different quantization schemes to support weight quantization.

## 4. Implement the weight loading logic

You now need to implement the `load_weights` method in your `*ForCausalLM` class.
This method should load the weights from the HuggingFace's checkpoint file and assign them to the corresponding layers in your model. Specifically, for `MergedColumnParallelLinear` and `QKVParallelLinear` layers, if the original model has separated weight matrices, you need to load the different parts separately.

## 5. Register your model

Aphrodite relies on a model registry to determine how to run each model.

If your model is not on this list, you must register it to Aphrodite.
This page provides detailed instructions on how to do so.

### Built-in models

To add a model directly to the Aphrodite library, start by forking our [GitHub repository](https://github.com/aphrodite-engine/aphrodite) and then [build it from source][build-from-source].
This gives you the ability to modify the codebase and test your model.

After you have implemented your model, put it into the <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models> directory.
Then, add your model class to `_APHRODITE_MODELS` in <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/registry.py> so that it is automatically registered upon importing Aphrodite.

:::note
    The list of models in each section should be maintained in alphabetical order.
:::

### Out-of-tree models

You can load an external model using a plugin without modifying the Aphrodite codebase.

To register the model, use the following code:

```python
# The entrypoint of your plugin
def register():
    from aphrodite import ModelRegistry
    from your_code import YourModelForCausalLM

    ModelRegistry.register_model("YourModelForCausalLM", YourModelForCausalLM)
```

If your model imports modules that initialize CUDA, consider lazy-importing it to avoid errors like `RuntimeError: Cannot re-initialize CUDA in forked subprocess`:

```python
# The entrypoint of your plugin
def register():
    from aphrodite import ModelRegistry

    ModelRegistry.register_model(
        "YourModelForCausalLM",
        "your_code:YourModelForCausalLM"
    )
```

## 6. Multimodal support

This section walks you through the steps to extend a basic model so that it accepts multi-modal inputs.

### 1. Update the base Aphrodite model

It is assumed that you have already implemented the model in Aphrodite according to the previously outlined steps.
Further update the model as follows:

- Implement [get_placeholder_str][aphrodite.modeling.models.interfaces.SupportsMultiModal.get_placeholder_str] to define the placeholder string which is used to represent the multi-modal item in the text prompt. This should be consistent with the chat template of the model.

```python
class YourModelForImage2Seq(nn.Module):
    ...

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<image>"

        raise ValueError("Only image modality is supported")
```

- Reserve a keyword parameter in [forward][torch.nn.Module.forward] for each input tensor that corresponds to a multi-modal input, as shown in the following example:

```diff
  def forward(
      self,
      input_ids: torch.Tensor,
      positions: torch.Tensor,
+     pixel_values: torch.Tensor,
  ) -> SamplerOutput:
```
  
  More conveniently, you can simply pass `**kwargs` to the [forward][torch.nn.Module.forward] method and retrieve the keyword parameters for multimodal inputs from it.

- Implement `get_multimodal_embeddings` (`aphrodite.modeling.models.interfaces.SupportsMultiModal.get_multimodal_embeddings`) that returns the embeddings from running the multimodal inputs through the multimodal tokenizer of the model. Below we provide a boilerplate of a typical implementation pattern, but feel free to adjust it to your own needs.

```python
class YourModelForImage2Seq(nn.Module):
    ...

    def _process_image_input(self, image_input: YourModelImageInputs) -> torch.Tensor:

        assert self.vision_encoder is not None
        image_features = self.vision_encoder(image_input)
        return self.multi_modal_projector(image_features)

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:

        # Validate the multimodal input keyword arguments
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        # Run multimodal inputs through encoder and projector
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings
```

:::important
The returned `multimodal_embeddings` must be either a **3D [torch.Tensor][]** of shape `(num_items, feature_size, hidden_size)`, or a **list / tuple of 2D [torch.Tensor][]'s** of shape `(feature_size, hidden_size)`, so that `multimodal_embeddings[i]` retrieves the embeddings generated from the `i`-th multimodal data item (e.g, image) of the request.
:::

- Implement `get_input_embeddings` (`aphrodite.modeling.models.interfaces.SupportsMultiModal.get_input_embeddings`) to merge `multimodal_embeddings` with text embeddings from the `input_ids`. If input processing for the model is implemented correctly (see sections below), then you can leverage the utility function we provide to easily merge the embeddings.

```python
from .utils import merge_multimodal_embeddings

class YourModelForImage2Seq(nn.Module):
    ...

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:

        # `get_input_embeddings` should already be implemented for the language 
        # model as one of the requirements of basic Aphrodite model implementation.
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)

        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids, 
                inputs_embeds=inputs_embeds, 
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=self.config.image_token_index)

        return inputs_embeds
```

- Implement `get_language_model` (`aphrodite.modeling.models.interfaces.SupportsMultiModal.get_language_model`) getter to provide stable access to the underlying language model.

    ```python
    class YourModelForImage2Seq(nn.Module):
        ...

        def get_language_model(self) -> torch.nn.Module:
            # Change `language_model` according to your implementation.
            return self.language_model
    ```

- Once the above steps are done, update the model class with the `SupportsMultiModal` interface.

  ```diff
  + from aphrodite.modeling.models.interfaces import SupportsMultiModal

  - class YourModelForImage2Seq(nn.Module):
  + class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
  ```

:::note
    The model class does not have to be named `*ForCausalLM`.
    Check out [the HuggingFace Transformers documentation](https://huggingface.co/docs/transformers/model_doc/auto#multimodal) for some examples.
:::

### 2. Specify processing information

Next, create a subclass of `BaseProcessingInfo` (`aphrodite.multimodal.processing.BaseProcessingInfo`)
to provide basic information related to HF processing.

### Maximum number of input items

You need to override the abstract method `get_supported_mm_limits` (`aphrodite.multimodal.processing.BaseProcessingInfo.get_supported_mm_limits`)
to return the maximum number of input items for each modality supported by the model.

For example, if the model supports any number of images but only one video per prompt:

```python
def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
    return {"image": None, "video": 1}
```

### 3. Specify dummy inputs

Then, inherit `BaseDummyInputsBuilder` (`aphrodite.multimodal.profiling.BaseDummyInputsBuilder`) to construct dummy inputs for
HF processing as well as memory profiling.

### For memory profiling

Override the abstract methods `get_dummy_text` (`aphrodite.multimodal.profiling.BaseDummyInputsBuilder.get_dummy_text`) and `get_dummy_mm_data` (`aphrodite.multimodal.profiling.BaseDummyInputsBuilder.get_dummy_mm_data`) to construct dummy inputs for memory profiling. These dummy inputs should result in the worst-case memory usage of the model so that Aphrodite can reserve the correct amount of memory for it.

Assuming that the memory usage increases with the number of tokens, the dummy inputs can be constructed to maximize the number of output embeddings, which is the same number as placeholder feature tokens.

=== "Basic example: LLaVA"

    Looking at the code of HF's `LlavaForConditionalGeneration`:

```python
# https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/llava/modeling_llava.py#L530-L544
n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
n_image_features = image_features.shape[0] * image_features.shape[1]

if n_image_tokens != n_image_features:
    raise ValueError(
        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
    )
special_image_mask = (
    (input_ids == self.config.image_token_index)
    .unsqueeze(-1)
    .expand_as(inputs_embeds)
    .to(inputs_embeds.device)
)
image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
```

The number of placeholder feature tokens per image is `image_features.shape[1]`.
`image_features` is calculated inside the `get_image_features` method:

```python
# https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/llava/modeling_llava.py#L290-L300
image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)

selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
if vision_feature_select_strategy == "default":
    selected_image_feature = selected_image_feature[:, 1:]
elif vision_feature_select_strategy == "full":
    selected_image_feature = selected_image_feature
else:
    raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")
image_features = self.multi_modal_projector(selected_image_feature)
return image_features
```

We can infer that `image_features.shape[1]` is based on `image_outputs.hidden_states.shape[1]` from the vision tower
(`CLIPVisionModel` for the [`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf) model).
Moreover, we only need the sequence length (the second dimension of the tensor) to get `image_features.shape[1]`.
The sequence length is determined by the initial hidden states in `CLIPVisionTransformer` since the attention
mechanism doesn't change the sequence length of the output hidden states.

```python
# https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/modeling_clip.py#L1094-L1102
hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
hidden_states = self.pre_layrnorm(hidden_states)

encoder_outputs = self.encoder(
    inputs_embeds=hidden_states,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict,
)
```

To find the sequence length, we turn to the code of `CLIPVisionEmbeddings`:

```python
# https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/modeling_clip.py#L247-L257
target_dtype = self.patch_embedding.weight.dtype
patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

class_embeds = self.class_embedding.expand(batch_size, 1, -1)
embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
if interpolate_pos_encoding:
    embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
else:
    embeddings = embeddings + self.position_embedding(self.position_ids)
return embeddings
```

We can infer that `embeddings.shape[1] == self.num_positions`, where

```python
# https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/modeling_clip.py#L195-L196
self.num_patches = (self.image_size // self.patch_size) ** 2
self.num_positions = self.num_patches + 1
```

Overall, the number of placeholder feature tokens for an image can be calculated as:

??? code

    ```python
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        hf_processor = self.get_hf_processor()

        image_size = hf_config.vision_config.image_size
        patch_size = hf_config.vision_config.patch_size

        num_image_tokens = (image_size // patch_size) ** 2 + 1
        if hf_processor.vision_feature_select_strategy == "default":
            num_image_tokens -= 1

        return num_image_tokens
    ```

Notice that the number of image tokens doesn't depend on the image width and height.
We can simply use a dummy `image_size` to calculate the multimodal profiling data:

```python
# NOTE: In actuality, this is usually implemented as part of the
# model's subclass of `BaseProcessingInfo`, but we show it as is
# here for simplicity.
def get_image_size_with_most_features(self) -> ImageSize:
    hf_config = self.get_hf_config()
    width = height = hf_config.image_size
    return ImageSize(width=width, height=height)

def get_dummy_mm_data(
    self,
    seq_len: int,
    mm_counts: Mapping[str, int],
) -> MultiModalDataDict:
    num_images = mm_counts.get("image", 0)

    target_width, target_height = \
        self.info.get_image_size_with_most_features()

    return {
        "image":
        self._get_dummy_images(width=target_width,
                            height=target_height,
                            num_images=num_images)
    }
```

For the text, we simply expand the multimodal image token from the model config to match the desired number of images.

```python
def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
    num_images = mm_counts.get("image", 0)

    processor = self.info.get_hf_processor()
    image_token = processor.image_token

    return image_token * num_images
```

=== "No input placeholders: PaliGemma"

Unlike LLaVA, PaliGemma's HF processor does not expect image placeholder tokens in the input prompt; the placeholder feature tokens are instead inserted afterwards. So the dummy prompt text is empty regardless of the number of images:

```python
def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
    return ""
```

PaliGemma resizes every image to a square of `vision_config.image_size`, so the number of placeholder feature tokens per image is fixed at `(image_size // patch_size) ** 2`. This is computed by the SigLIP vision encoder that PaliGemma uses:

```python
class SiglipEncoderInfo(VisionEncoderInfo[SiglipVisionConfig]):
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        return self.get_patch_grid_length() ** 2

    def get_patch_grid_length(self) -> int:
        image_size, patch_size = self.get_image_size(), self.get_patch_size()
        return image_size // patch_size
```

Since the number of image tokens does not depend on the input image dimensions, we can use a dummy image of the model's expected input size for the multimodal profiling data:

```python
def get_dummy_mm_data(
    self,
    seq_len: int,
    mm_counts: Mapping[str, int],
) -> MultiModalDataDict:
    hf_config = self.info.get_hf_config()
    vision_config = hf_config.vision_config
    max_image_size = vision_config.image_size
    num_images = mm_counts.get("image", 0)

    return {
        "image": self._get_dummy_images(
            width=max_image_size,
            height=max_image_size,
            num_images=num_images,
        )
    }
```

### 4. Specify processing details

Afterwards, create a subclass of `BaseMultiModalProcessor` (`aphrodite.multimodal.processing.BaseMultiModalProcessor`)
to fill in the missing details about HF processing.

### Multi-modal fields

Override `_get_mm_fields_config` (`aphrodite.multimodal.processing.BaseMultiModalProcessor._get_mm_fields_config`) to
return a schema of the tensors outputted by the HF processor that are related to the input multi-modal items.

### Basic example: LLaVA

The output of `CLIPImageProcessor` is a simple tensor with shape
`(num_images, num_channels, image_height, image_width)`:

```python
# https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/image_processing_clip.py#L339-L345
images = [
    to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
    for image in all_images
]

data = {"pixel_values": images}
return BatchFeature(data=data, tensor_type=return_tensors)
```

So, we override `_get_mm_fields_config` (`aphrodite.multimodal.processing.BaseMultiModalProcessor._get_mm_fields_config`) as follows:

```python
def _get_mm_fields_config(
    self,
    hf_inputs: BatchFeature,
    hf_processor_mm_kwargs: Mapping[str, object],
) -> Mapping[str, MultiModalFieldConfig]:
    return dict(
        pixel_values=MultiModalFieldConfig.batched("image"),
    )
```

:::note
Our [actual code](https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/llava.py) additionally supports
pre-computed image embeddings, which can be passed to be model via the `image_embeds` argument.
:::

### With postprocessing: Mistral3

The `pixel_values` output of Mistral3's HF processor pads every image in the batch to a common size, so that they can be stacked into a single tensor.

To use `MultiModalFieldConfig.batched` (`aphrodite.multimodal.inputs.MultiModalFieldConfig.batched`) like in LLaVA, each image's features must be independent of the others. So, we un-pad each image back to its own size by overriding `BaseMultiModalProcessor._call_hf_processor` (`aphrodite.multimodal.processing.BaseMultiModalProcessor._call_hf_processor`):

```python
def _call_hf_processor(
    self,
    prompt: str,
    mm_data: Mapping[str, object],
    mm_kwargs: Mapping[str, object],
    tok_kwargs: Mapping[str, object],
) -> BatchFeature:
    processed_outputs = super()._call_hf_processor(
        prompt=prompt,
        mm_data=mm_data,
        mm_kwargs=mm_kwargs,
        tok_kwargs=tok_kwargs,
    )

    pixel_values = processed_outputs.get("pixel_values")
    if pixel_values is not None:
        image_sizes = processed_outputs["image_sizes"]
        assert len(pixel_values) == len(image_sizes)

        processed_outputs["pixel_values"] = [
            p[:, :h, :w] for p, (h, w) in zip(pixel_values, image_sizes)
        ]

    return processed_outputs
```

:::note
The `_call_hf_processor` method specifies both `mm_kwargs` and `tok_kwargs` for processing. `mm_kwargs` is used to both initialize and call the huggingface processor, whereas `tok_kwargs` is only used to call the huggingface processor.
:::

Since `pixel_values` is now a list with one tensor per image, we can override [_get_mm_fields_config][aphrodite.multimodal.processing.BaseMultiModalProcessor._get_mm_fields_config] as follows:

```python
def _get_mm_fields_config(
    self,
    hf_inputs: BatchFeature,
    hf_processor_mm_kwargs: Mapping[str, object],
) -> Mapping[str, MultiModalFieldConfig]:
    return dict(
        pixel_values=MultiModalFieldConfig.batched("image"),
        image_embeds=MultiModalFieldConfig.batched("image"),
    )
```

:::note
See our [actual code](https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/mistral3.py) for the full implementation.
:::

### Prompt updates

Override [_get_prompt_updates][aphrodite.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates] to
return a list of [PromptUpdate][aphrodite.multimodal.processing.PromptUpdate] instances.

Each [PromptUpdate][aphrodite.multimodal.processing.PromptUpdate] instance specifies an update operation
(e.g.: insertion, replacement) performed by the HF processor.

#### Basic example: LLaVA

Looking at HF's `LlavaProcessor`:

```python
# https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/llava/processing_llava.py#L167-L170
prompt_strings = []
for sample in text:
    sample = sample.replace(self.image_token, self.image_token * num_image_tokens)
    prompt_strings.append(sample)
```

It simply repeats each input `image_token` a number of times equal to the number of placeholder feature tokens (`num_image_tokens`).
Based on this, we override `_get_prompt_updates` (`aphrodite.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates`) as follows:

```python
def _get_prompt_updates(
    self,
    mm_items: MultiModalDataItems,
    hf_processor_mm_kwargs: Mapping[str, object],
    out_mm_kwargs: MultiModalKwargsItems,
) -> Sequence[PromptUpdate]:
    hf_config = self.info.get_hf_config()
    image_token_id = hf_config.image_token_index

    def get_replacement(item_idx: int):
        images = mm_items.get_items("image", ImageProcessorItems)

        image_size = images.get_image_size(item_idx)
        num_image_tokens = self.info.get_num_image_tokens(
            image_width=image_size.width,
            image_height=image_size.height,
        )

        return [image_token_id] * num_image_tokens

    return [
        PromptReplacement(
            modality="image",
            target=[image_token_id],
            replacement=get_replacement,
        ),
    ]
```

#### Handling additional tokens: PaliGemma

PaliGemma's HF processor inserts, after the prompt's leading `<bos>` token, a run of image tokens followed by a second `<bos>` token that marks the start of the text prompt. We start by building the run of image tokens, one per placeholder feature token:

```python
def get_insertion(item_idx: int):
    images = mm_items.get_items(
        "image", (ImageEmbeddingItems, ImageProcessorItems)
    )

    if isinstance(images, ImageEmbeddingItems):
        num_image_tokens = images.get_feature_size(item_idx)
    else:
        image_size = images.get_image_size(item_idx)
        num_image_tokens = self.info.get_num_image_tokens(
            image_width=image_size.width,
            image_height=image_size.height,
        )

    image_tokens = [image_token_id] * num_image_tokens
    ...
```

The trailing `<bos>` token is an additional token that must not receive a vision embedding. To assign the vision embeddings to only the image tokens, return an instance of `PromptUpdateDetails` (`aphrodite.multimodal.processing.PromptUpdateDetails`) and mark the embedding tokens with `embed_token_id`:

```python
return PromptUpdateDetails.select_token_id(
    image_tokens + [bos_token_id],
    embed_token_id=image_token_id,
)
```

Putting it together, we override [_get_prompt_updates][aphrodite.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates]. Since these tokens are inserted rather than replacing an existing placeholder after the prompt's leading `<bos>`, we use [PromptInsertion][aphrodite.multimodal.processing.PromptInsertion] with a prefix target:

```python
def _get_prompt_updates(
    self,
    mm_items: MultiModalDataItems,
    hf_processor_mm_kwargs: Mapping[str, object],
    out_mm_kwargs: MultiModalKwargsItems,
) -> Sequence[PromptUpdate]:
    hf_config = self.info.get_hf_config()
    image_token_id = hf_config.image_token_index

    tokenizer = self.info.get_tokenizer()
    bos_token_id = tokenizer.bos_token_id
    assert isinstance(bos_token_id, int)

    def get_insertion(item_idx: int):
        images = mm_items.get_items(
            "image", (ImageEmbeddingItems, ImageProcessorItems)
        )

        if isinstance(images, ImageEmbeddingItems):
            num_image_tokens = images.get_feature_size(item_idx)
        else:
            image_size = images.get_image_size(item_idx)
            num_image_tokens = self.info.get_num_image_tokens(
                image_width=image_size.width,
                image_height=image_size.height,
            )

        image_tokens = [image_token_id] * num_image_tokens

        return PromptUpdateDetails.select_token_id(
            image_tokens + [bos_token_id],
            embed_token_id=image_token_id,
        )

    return [
        PromptInsertion(
            modality="image",
            target=PromptIndexTargets.prefix(
                [bos_token_id] if tokenizer.add_bos_token else []
            ),
            insertion=get_insertion,
        )
    ]
```

### 5. Register processor-related classes

After you have defined `BaseProcessingInfo` (`aphrodite.multimodal.processing.BaseProcessingInfo`) (Step 2),
`BaseDummyInputsBuilder` (`aphrodite.multimodal.profiling.BaseDummyInputsBuilder`) (Step 3),
and `BaseMultiModalProcessor` (`aphrodite.multimodal.processing.BaseMultiModalProcessor`) (Step 4),
decorate the model class with `MULTIMODAL_REGISTRY.register_processor` (`aphrodite.multimodal.registry.MultiModalRegistry.register_processor`)
to register them to the multi-modal registry:

```diff
  from aphrodite.modeling.models.interfaces import SupportsMultiModal
+ from aphrodite.multimodal import MULTIMODAL_REGISTRY

+ @MULTIMODAL_REGISTRY.register_processor(YourMultiModalProcessor,
+                                         info=YourProcessingInfo,
+                                         dummy_inputs=YourDummyInputsBuilder)
  class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
```

### Notes

### Inserting feature tokens without replacement

Some HF processors directly insert feature tokens without replacing anything in the original prompt. In that case, you can use [PromptInsertion][aphrodite.multimodal.processing.PromptInsertion] instead of [PromptReplacement][aphrodite.multimodal.processing.PromptReplacement] inside [_get_prompt_updates][aphrodite.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates].

Examples:

- BLIP-2 (insert at start of prompt): <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/blip2.py>
- Florence2 (insert at start of prompt): <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/florence2.py>
- Molmo (insert after `<|endoftext|>` token): <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/molmo.py>

### Handling prompt updates unrelated to multi-modal data

[_get_prompt_updates][aphrodite.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates] assumes that each application of prompt update corresponds to one multi-modal item. If the HF processor performs additional processing regardless of how many multi-modal items there are, you should override [_apply_hf_processor_tokens_only][aphrodite.multimodal.processing.BaseMultiModalProcessor._apply_hf_processor_tokens_only] so that the processed token inputs are consistent with the result of applying the HF processor on text inputs. This is because token inputs bypass the HF processor according to our design.

Examples:

- Chameleon (appends `sep_token`): <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/chameleon.py>
- Molmo2 (prepends `bos_token`): <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/molmo2.py>
- Molmo (applies chat template which is not defined elsewhere): <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/molmo.py>

### Custom HF processor

Some models don't define an HF processor class on HF Hub. In that case, you can define a custom HF processor that has the same call signature as HF processors and pass it to [_call_hf_processor][aphrodite.multimodal.processing.BaseMultiModalProcessor._call_hf_processor].

Examples:

- DeepSeek-VL2: <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/deepseek_vl2.py>
- InternVL: <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/internvl.py>
- Qwen-VL: <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/qwen_vl.py>

## Transcription Models

# Speech-to-Text (Transcription/Translation) Support

This section walks you through the steps to add support for speech-to-text (ASR) models to Aphrodite’s transcription and translation APIs by implementing `SupportsTranscription` (`aphrodite.modeling.models.interfaces.SupportsTranscription`).

## Update the base Aphrodite model

It is assumed you have already implemented your model in Aphrodite according to the basic model guide. Extend your model with the `SupportsTranscription` (`aphrodite.modeling.models.interfaces.SupportsTranscription`) interface and implement the following class attributes and methods.

### `supported_languages` and `supports_transcription_only`

Declare supported languages and capabilities:

- The `supported_languages` mapping is validated at init time.
- Set `supports_transcription_only=True` if the model should not serve text generation (eg Whisper).

```python
from typing import ClassVar, Mapping, Optional, Literal
import numpy as np
import torch
from torch import nn

from aphrodite.config import ModelConfig, SpeechToTextConfig
from aphrodite.inputs.data import PromptType
from aphrodite.modeling.models.interfaces import SupportsTranscription

class YourASRModel(nn.Module, SupportsTranscription):
    # Map of ISO 639-1 language codes to language names
    supported_languages: ClassVar[Mapping[str, str]] = {
        "en": "English",
        "it": "Italian",
        # ... add more as needed
    }
    
    # If your model only supports audio-conditioned generation
    # (no text-only generation), enable this flag.
    supports_transcription_only: ClassVar[bool] = True
```

Provide an ASR configuration via `get_speech_to_text_config` (`aphrodite.modeling.models.interfaces.SupportsTranscription.get_speech_to_text_config`).

This is for controlling general behavior of the API when serving your model:

```python
class YourASRModel(nn.Module, SupportsTranscription):
    ...

    @classmethod
    def get_speech_to_text_config(
        cls,
        model_config: ModelConfig,
        task_type: Literal["transcribe", "translate"],
    ) -> SpeechToTextConfig:
        return SpeechToTextConfig(
            sample_rate=16_000,
            max_audio_clip_s=30,
            # Set to None to disable server-side chunking if your
            # model/processor handles it already
            min_energy_split_window_size=None,
        )
```

See [Audio preprocessing and chunking](#audio-preprocessing-and-chunking) for what each field controls.

Implement the prompt construction via `get_generation_prompt` (`aphrodite.modeling.models.interfaces.SupportsTranscription.get_generation_prompt`). The server passes you the resampled waveform and task parameters; you return a valid `PromptType` (`aphrodite.inputs.data.PromptType`). There are two common patterns:

#### Multimodal LLM with audio embeddings (e.g., Voxtral, Gemma3n)

Return a dict containing `multi_modal_data` with the audio, and either a `prompt` string or `prompt_token_ids`:

```python
class YourASRModel(nn.Module, SupportsTranscription):
    ...

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
        language: Optional[str],
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: Optional[str],
    ) -> PromptType:
        # Example with a free-form instruction prompt
        task_word = "Transcribe" if task_type == "transcribe" else "Translate"
        prompt = (
            "<start_of_turn>user\n"
            f"{task_word} this audio: <audio_soft_token>"
            "<end_of_turn>\n<start_of_turn>model\n"
        )

        return {
            "multi_modal_data": {"audio": (audio, stt_config.sample_rate)},
            "prompt": prompt,
        }
```

For further clarification on multi modal inputs, please refer to Multi-Modal Inputs.

#### Encoder–decoder audio-only (e.g., Whisper)

Return a dict with separate `encoder_prompt` and `decoder_prompt` entries:

```python
class YourASRModel(nn.Module, SupportsTranscription):
    ...

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
        language: Optional[str],
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: Optional[str],
    ) -> PromptType:
        if language is None:
            raise ValueError("Language must be specified")

        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": (audio, stt_config.sample_rate),
                },
            },
            "decoder_prompt": (
                (f"<|prev|>{request_prompt}" if request_prompt else "")
                + f"<|startoftranscript|><|{language}|>"
                + f"<|{task_type}|><|notimestamps|>"
            ),
        }
        return cast(PromptType, prompt)
```

### `validate_language` (optional)

Language validation via `validate_language` (`aphrodite.modeling.models.interfaces.SupportsTranscription.validate_language`)

If your model requires a language and you want a default, override this method (see Whisper):

```python
@classmethod
def validate_language(cls, language: Optional[str]) -> Optional[str]:
    if language is None:
        logger.warning(
            "Defaulting to language='en'. If you wish to transcribe audio in a different language, pass the `language` field.")
        language = "en"
    return super().validate_language(language)
```

### `get_num_audio_tokens` (optional)

Token accounting for streaming via `get_num_audio_tokens` (`aphrodite.modeling.models.interfaces.SupportsTranscription.get_num_audio_tokens`)

Provide a fast duration→token estimate to improve streaming usage statistics:

```python
class YourASRModel(nn.Module, SupportsTranscription):
    ...

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> Optional[int]:
        # Return None if unknown; otherwise return an estimate.
        return int(audio_duration_s * stt_config.sample_rate // 320)  # example
```

## Audio preprocessing and chunking

The API server takes care of basic audio I/O and optional chunking before building prompts:

- Resampling: Input audio is resampled to `SpeechToTextConfig.sample_rate` using `librosa`.
- Chunking: If `SpeechToTextConfig.allow_audio_chunking` is True and the duration exceeds `max_audio_clip_s`, the server splits the audio into overlapping chunks and generates a prompt per chunk. Overlap is controlled by `overlap_chunk_second`.
- Energy-aware splitting: When `min_energy_split_window_size` is set, the server finds low-energy regions to minimize cutting within words.

Relevant server logic:

```python
# aphrodite/entrypoints/openai/speech_to_text.py
async def _preprocess_speech_to_text(...):
    language = self.model_cls.validate_language(request.language)
    ...
    y, sr = librosa.load(bytes_, sr=self.asr_config.sample_rate)
    duration = librosa.get_duration(y=y, sr=sr)
    do_split_audio = (self.asr_config.allow_audio_chunking
                    and duration > self.asr_config.max_audio_clip_s)
    chunks = [y] if not do_split_audio else self._split_audio(y, int(sr))
    prompts = []
    for chunk in chunks:
        prompt = self.model_cls.get_generation_prompt(
            audio=chunk,
            stt_config=self.asr_config,
            model_config=self.model_config,
            language=language,
            task_type=self.task_type,
            request_prompt=request.prompt,
            to_language=to_language,
        )
        prompts.append(prompt)
    return prompts, duration
```

## Exposing tasks automatically

Aphrodite automatically advertises transcription support if your model implements the interface:

```python
if supports_transcription(model):
    if model.supports_transcription_only:
        return ["transcription"]
    supported_tasks.append("transcription")
```

When enabled, the server initializes the transcription and translation handlers:

```python
state.openai_serving_transcription = OpenAIServingTranscription(...) if "transcription" in supported_tasks else None
state.openai_serving_translation = OpenAIServingTranslation(...) if "transcription" in supported_tasks else None
```

No extra registration is required beyond having your model class available via the model registry and implementing `SupportsTranscription`.

## Examples in-tree

- Whisper encoder–decoder (audio-only): <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/whisper.py>
- Voxtral decoder-only (audio embeddings + LLM): <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/voxtral.py>
- Gemma3n decoder-only with fixed instruction prompt: <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/gemma3n_mm.py>

## Test with the API

Once your model implements `SupportsTranscription`, you can test the endpoints (API mimics OpenAI):

- Transcription (ASR):

    ```bash
    curl -s -X POST \
      -H "Authorization: Bearer $APHRODITE_API_KEY" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@/path/to/audio.wav" \
      -F "model=$MODEL_ID" \
      http://localhost:2242/v1/audio/transcriptions
    ```

- Translation (source → English unless otherwise supported):

    ```bash
    curl -s -X POST \
      -H "Authorization: Bearer $APHRODITE_API_KEY" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@/path/to/audio.wav" \
      -F "model=$MODEL_ID" \
      http://localhost:2242/v1/audio/translations
    ```

Or check out more examples in <https://github.com/aphrodite-engine/aphrodite-engine/tree/main/examples/online_serving>.

:::note

- If your model handles chunking internally (e.g., via its processor or encoder), set `min_energy_split_window_size=None` in the returned `SpeechToTextConfig` to disable server-side chunking.
- Implementing `get_num_audio_tokens` improves accuracy of streaming usage metrics (`prompt_tokens`) without an extra forward pass.
- For multilingual behavior, keep `supported_languages` aligned with actual model capabilities.

:::

## Frequently Asked Questions

### How to support models with interleaving sliding windows?

For models with interleaving sliding windows (e.g. `google/gemma-2-2b-it` and `mistralai/Ministral-8B-Instruct-2410`), the scheduler will treat the model as a full-attention model, i.e., kv-cache of all tokens will not be dropped. This is to make sure prefix caching works with these models. Sliding window only appears as a parameter to the attention kernel computation.

To support a model with interleaving sliding windows, we need to take care of the following details:

- Make sure the model's `config.json` contains `layer_types`.
- In the modeling code, parse the correct sliding window value for every layer, and pass it to the attention layer's `per_layer_sliding_window` argument. For reference, check [these lines](https://github.com/aphrodite-engine/aphrodite-engine/blob/c166e76b0e96b0e7c9ae96b76ae32eb274be553b/aphrodite/modeling/models/llama.py#L171-L190).

With these two steps, interleave sliding windows should work with the model.

### How to support models that use Mamba?

We consider 3 different scenarios:

1. Models that use Mamba layers (either Mamba-1 or Mamba-2) but do not use attention layers.
2. Models that combine Mamba layers (either Mamba-1 or Mamba-2) together with attention layers.
3. Models that combine Mamba-like mechanisms (e.g., Linear Attention, ShortConv) together with attention layers.

For case (1), we recommend looking at the implementation of [`MambaForCausalLM`](https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/mamba.py) (for Mamba-1) or [`Mamba2ForCausalLM`](https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/mamba2.py) (for Mamba-2) as a reference.
The model should inherit protocol `IsAttentionFree` and also implement class methods `get_mamba_state_dtype_from_config` and `get_mamba_state_shape_from_config` to calculate the state shapes and data types from the config.
For the mamba layers themselves, please use the [`MambaMixer`](https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/layers/mamba/mamba_mixer.py) (for Mamba-1) or [`MambaMixer2`](https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/layers/mamba/mamba_mixer2.py) (for Mamba-2) classes.
Please *do not* use the `MambaCacheManager` (deprecated in V1) or replicate any of the V0-specific code paths in the existing model implementations.
V0-only classes and code will be removed in the very near future.
The model should also be added to the `MODELS_CONFIG_MAP` dictionary in <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/config.py> to ensure that the runtime defaults are optimized.

For case (2), we recommend using as a reference the implementation of [`JambaForCausalLM`](https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/jamba.py) (for an example of a model that uses Mamba-1 and attention together) or [`BambaForCausalLM`](https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/bamba.py) (for an example of a model that uses Mamba-2 and attention together).
These models should follow the same instructions as case (1), but they should inherit protocol `IsHybrid` (instead of `IsAttentionFree`) and it is *not* necessary to add them to the `MODELS_CONFIG_MAP` (their runtime defaults will be inferred from the protocol).

For case (3), we recommend looking at the implementation of [`MiniMaxText01ForCausalLM`](https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/minimax_text_01.py) or [`Lfm2ForCausalLM`](https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/lfm2.py) as a reference, which use custom "mamba-like" layers `MiniMaxText01LinearAttention` and `ShortConv` respectively.
Please follow the same guidelines as case (2) for implementing these models.
We use "mamba-like" to refer to layers that posses a state that is updated in-place, rather than being appended-to (like KV cache for attention).
For implementing new custom mamba-like layers, one should inherit from `MambaBase` and implement the methods `get_state_dtype`, `get_state_shape` to calculate the data types and state shapes at runtime, as well as `mamba_type` and `get_attn_backend`.
It is also necessary to implement the "attention meta-data" class which handles the meta-data that is common across all layers.
Please see [`LinearAttentionMetadata`](https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/v1/attention/backends/linear_attn.py) or [`ShortConvAttentionMetadata`](https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/v1/attention/backends/short_conv_attn.py) for examples of this.
Finally, if one wants to support torch compile and CUDA graphs, it necessary to wrap the call to the mamba-like layer inside a custom op and register it.
Please see the calls to `direct_register_custom_op` in <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/models/minimax_text_01.py> or <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/modeling/layers/mamba/short_conv.py> for examples of this.
The new custom op should then be added to the list `_attention_ops` in <https://github.com/aphrodite-engine/aphrodite-engine/blob/main/aphrodite/config/compilation.py> to ensure that piecewise CUDA graphs works as intended.
