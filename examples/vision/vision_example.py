# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use Aphrodite for running offline inference
with the correct prompt format on vision language models.
For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""

import os

import cv2
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from aphrodite import LLM, SamplingParams
from aphrodite.assets.video import VideoAsset
from aphrodite.multimodal.utils import sample_frames_from_video
from aphrodite.platforms import current_platform
from aphrodite.utils import FlexibleArgumentParser

# Input image and question
image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "burg.jpg")
image = Image.open(image_path).convert("RGB")
img_question = "What is the content of this image?"

# Input video and question
video_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nadeko.mp4")
vid_question = "What's in this video?"


def load_video_frames(video_path: str, num_frames: int) -> np.ndarray:
    """
    Load video frames from a local file path.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to sample from the video

    Returns:
        np.ndarray: Array of sampled video frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    frames = np.stack(frames)
    return sample_frames_from_video(frames, num_frames)


# LLaVA-1.5
def run_llava(question, modality):
    assert modality == "image"

    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    llm = LLM(model="llava-hf/llava-1.5-7b-hf")
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LLaVA-1.6/LLaVA-NeXT
def run_llava_next(question, modality):
    assert modality == "image"

    prompt = f"[INST] <image>\n{question} [/INST]"
    llm = LLM(model="llava-hf/llava-v1.6-mistral-7b-hf", max_model_len=8192)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LlaVA-NeXT-Video
# Currently only support for video input
def run_llava_next_video(question, modality):
    assert modality == "video"

    prompt = f"USER: <video>\n{question} ASSISTANT:"
    llm = LLM(model="llava-hf/LLaVA-NeXT-Video-7B-hf")
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LLaVA-OneVision
def run_llava_onevision(question, modality):
    if modality == "video":
        prompt = f"<|im_start|>user <video>\n{question}<|im_end|> \
        <|im_start|>assistant\n"

    elif modality == "image":
        prompt = f"<|im_start|>user <image>\n{question}<|im_end|> \
        <|im_start|>assistant\n"

    llm = LLM(model="llava-hf/llava-onevision-qwen2-7b-ov-hf", max_model_len=32768)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Phi-3-Vision
def run_phi3v(question, modality):
    assert modality == "image"

    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"  # noqa: E501
    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (128k) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # In this example, we override max_num_seqs to 5 while
    # keeping the original context length of 128k.
    llm = LLM(
        model="microsoft/Phi-3-vision-128k-instruct",
        trust_remote_code=True,
        max_num_seqs=5,
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# PaliGemma
def run_paligemma(question, modality):
    assert modality == "image"

    # PaliGemma has special prompt format for VQA
    prompt = "caption en"
    llm = LLM(model="google/paligemma-3b-mix-224")
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Chameleon
def run_chameleon(question, modality):
    assert modality == "image"

    prompt = f"{question}<image>"
    llm = LLM(model="facebook/chameleon-7b")
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# MiniCPM-V
def run_minicpmv(question, modality):
    assert modality == "image"

    # 2.0
    # The official repo doesn't work yet, so we need to use a fork for now
    # model_name = "HwwwH/MiniCPM-V-2"

    # 2.5
    # model_name = "openbmb/MiniCPM-Llama3-V-2_5"

    # 2.6
    model_name = "openbmb/MiniCPM-V-2_6"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
    )
    # NOTE The stop_token_ids are different for various versions of MiniCPM-V
    # 2.0
    # stop_token_ids = [tokenizer.eos_id]

    # 2.5
    # stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

    # 2.6
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    messages = [{"role": "user", "content": f"(<image>./</image>)\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return llm, prompt, stop_token_ids


# InternVL
def run_internvl(question, modality):
    assert modality == "image"

    model_name = "OpenGVLab/InternVL2-2B"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_num_seqs=5,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [{"role": "user", "content": f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B#service
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    return llm, prompt, stop_token_ids


def run_mono_internvl(question, modality):
    assert modality == "image"

    model_name = "OpenGVLab/Mono-InternVL-2B-S1-3"

    llm = LLM(model=model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [{"role": "user", "content": f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    return llm, prompt, stop_token_ids


# BLIP-2
def run_blip2(question, modality):
    assert modality == "image"

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa
    prompt = f"Question: {question} Answer:"
    llm = LLM(model="Salesforce/blip2-opt-2.7b")
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Qwen
def run_qwen_vl(question, modality):
    assert modality == "image"

    llm = LLM(
        model="Qwen/Qwen-VL",
        trust_remote_code=True,
        max_num_seqs=5,
    )

    prompt = f"{question}Picture 1: <img></img>\n"
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Qwen2-VL
def run_qwen2_vl(question, modality):
    assert modality == "image"

    model_name = "Qwen/Qwen2-VL-7B-Instruct"

    llm = LLM(
        model=model_name,
        max_num_seqs=5,
    )

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Molmo
def run_molmo(question: str, modality: str):
    assert modality == "image"
    model_name = "allenai/Molmo-7B-D-0924"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    prompt = question
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LLama
def run_mllama(question, modality):
    assert modality == "image"
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.
    # The configuration below has been confirmed to launch on a
    # single H100 GPU.
    llm = LLM(
        model=model_name,
        max_num_seqs=16,
        enforce_eager=True,
        max_model_len=8192,
    )
    prompt = f"<|image|><|begin_of_text|>{question}"
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# GLM-4v
def run_glm4v(question: str, modality: str):
    assert modality == "image"
    model_name = "THUDM/glm-4v-9b"

    llm = LLM(model=model_name, max_model_len=2048, max_num_seqs=2, trust_remote_code=True, enforce_eager=True)
    prompt = question
    stop_token_ids = [151329, 151336, 151338]
    return llm, prompt, stop_token_ids


# Pixtral HF-format
def run_pixtral_hf(question: str, modality: str):
    assert modality == "image"
    model_name = "mistral-community/pixtral-12b"

    llm = LLM(
        model=model_name,
        max_model_len=8192,
    )

    prompt = f"<s>[INST]{question}\n[IMG][/INST]"
    stop_token_ids = None
    return llm, prompt, stop_token_ids


model_example_map = {
    "llava": run_llava,
    "llava-next": run_llava_next,
    "llava-next-video": run_llava_next_video,
    "llava-onevision": run_llava_onevision,
    "phi3_v": run_phi3v,
    "paligemma": run_paligemma,
    "chameleon": run_chameleon,
    "minicpmv": run_minicpmv,
    "blip-2": run_blip2,
    "internvl_chat": run_internvl,
    "mono_internvl": run_mono_internvl,
    "qwen_vl": run_qwen_vl,
    "qwen2_vl": run_qwen2_vl,
    "molmo": run_molmo,
    "mllama": run_mllama,
    "glm4v": run_glm4v,
    "pixtral_hf": run_pixtral_hf,
}


def get_multi_modal_input(args):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if args.modality == "image":
        return {
            "data": image,
            "question": img_question,
        }

    if args.modality == "video":
        video = VideoAsset(name="nadeko.mp4", num_frames=args.num_frames, local_path=video_path).np_ndarrays
        return {
            "data": video,
            "question": vid_question,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)


def main(args):
    if current_platform.is_rocm():
        os.environ["APHRODITE_WORKER_MULTIPROC_METHOD"] = "spawn"

    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    modality = args.modality
    mm_input = get_multi_modal_input(args)
    data = mm_input["data"]
    question = mm_input["question"]

    llm, prompt, stop_token_ids = model_example_map[model](question, modality)

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(temperature=0.2, max_tokens=512, stop_token_ids=stop_token_ids)

    assert args.num_prompts > 0
    if args.num_prompts == 1:
        # Single inference
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {modality: data},
        }

    else:
        # Batch inference
        inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {modality: data},
            }
            for _ in range(args.num_prompts)
        ]

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Demo on using Aphrodite for offline inference with vision language models"
    )
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        default="llava",
        choices=model_example_map.keys(),
        help='Huggingface "model_type".',
    )
    parser.add_argument("--num-prompts", type=int, default=1, help="Number of prompts to run.")
    parser.add_argument(
        "--modality", type=str, default="image", choices=["image", "video"], help="Modality of the input."
    )
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames to extract from the video.")
    args = parser.parse_args()
    main(args)
