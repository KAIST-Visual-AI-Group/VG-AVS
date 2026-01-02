# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import traceback
from typing import Optional

import base64
import os
from typing import List, Optional

import cv2
import openai

def set_openai_key(key: Optional[str] = None):
    if key is None:
        assert "OPENAI_API_KEY" in os.environ
        key = os.environ["OPENAI_API_KEY"]
    openai.api_key = key


def prepare_openai_messages(content: str):
    return [{"role": "user", "content": content}]


def call_openai_api(
    messages: list,
    model: str = "gpt-4",
    seed: Optional[int] = None,
    max_tokens: int = 32,
    temperature: float = 0.2,
    verbose: bool = False,
):
    client = openai.OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if verbose:
        print("openai api response: {}".format(completion))
    assert len(completion.choices) == 1
    return completion.choices[0].message.content

def parse_score(output: str, tag: str = "Your mark:") -> str:
    if output.isdigit():
        return int(output)
    start_idx = output.find(tag)
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return int(output[start_idx:].replace(tag, "").strip())
    return int(output[start_idx:end_idx].replace(tag, "").strip())

def get_llm_match_score(
    question: str,
    answer: str,
    prediction: str,
    extra_answers: Optional[list] = None,
    openai_key: Optional[str] = None,
    openai_model: str = "gpt-4-1106-preview",
    openai_seed: int = 1234,
    openai_max_tokens: int = 32,
    openai_temperature: float = 0.2,
    verbose: bool = False,
):
    if prediction is None:
        return 0

    prompt = "mmbench_llm_match.txt"

    try:
        set_openai_key(key=openai_key)
        messages = prepare_openai_messages(
            prompt.format(
                question=question,
                answer=answer,
                prediction=prediction,
                extra_answers=extra_answers,
            ),
        )
        output = call_openai_api(
            messages=messages,
            model=openai_model,
            seed=openai_seed,
            max_tokens=openai_max_tokens,
            temperature=openai_temperature,
            verbose=verbose,
        )
        return parse_score(output)
    except Exception as e:
        traceback.print_exc()
        raise e

