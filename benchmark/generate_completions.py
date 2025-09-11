#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "litellm",
#     "tqdm",
#     "datasets==4.0.*",
# ]
# ///
"""
This script evaluates models on the CanItEdit dataset using vLLM for hosting
and LiteLLM as the client for inference.
"""

import datasets
from pathlib import Path
from tqdm import tqdm
from typing import List, Literal, Optional, TypedDict, Callable, TypeVar
import gzip
import json
from litellm import atext_completion, acompletion
import itertools
import asyncio


def gunzip_json_write(path: Path, data: dict) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


T = TypeVar("T")

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str
    # the prefix for the assistant's response. this is only used for the
    # last message in a conversation, and is ignored otherwise.
    # NOTE: leaving commented for python 3.10 compatibility
    #  prefix_after: NotRequired[str]


# this is one edit request
class EditCommand(TypedDict):
    instruction: Optional[str]
    content: str


# this is model's output
class EditResponse(TypedDict):
    instruction: Optional[str]
    content: str


# (old, instr, new) -> prompt
PromptFormatFunction = Callable[[str, str, str], str]

# (old, instr) -> [messages]
MessagesFormatFunction = Callable[[str, str], List[Message]]

# (old, new) -> response
PostProcessFunction = Callable[[str, str], str]


def direct_edit_prompt(
    old,
    instr,
    codeblock_before: Optional[str] = None,
    codeblock_after: Optional[str] = None,
):
    """
    The codeblock_before and codeblock_after arguments are used to specify
    if there should be a codeblock surrounding the code before and after
    the instruction. If None, then no codeblock is used. The string is the
    extension of the codeblock, e.g. "py" or "md".
    """
    if codeblock_before is not None:
        old = f"```{codeblock_before}\n{old}\n```"
    if codeblock_after is not None:
        new = f"```{codeblock_after}\n{new}\n```"
    before = f"""## Code Before:\n{old}\n"""
    instr = f"""## Instruction:\n{instr}\n"""
    after = f"""## Code After:\n"""
    return before + instr + after


def chat_edit_prompt_zeroshot(old: str, instr: str) -> List[Message]:
    return [
        {
            "role": "system",
            "content": """
You are PythonEditGPT. You will be provided the original code snippet and an instruction that specifies the changes you need to make. You will produce the changed code, based on the original code and the instruction given. Only produce the code, do not include any additional prose.
            """.strip(),
        },
        {
            "role": "user",
            "content": f"""
## Code Before
```py
{old}
```

## Instruction
{instr}

## Code After""".strip(),
        },
    ]


def python_markdown_codeblock_extract(_: str, new: str) -> str:
    # print("prior to extracting codeblock:", new)
    lines = new.split("\n")
    buf = ""
    in_codeblock = False
    for ln in lines:
        if ln.startswith("```"):
            if in_codeblock:
                break
            else:
                in_codeblock = True
        elif in_codeblock:
            buf += ln + "\n"
    # print("after extracting codeblock:", buf)
    return buf


class EditModel:
    def __init__(self):
        pass

    async def generate(self, prompt: EditCommand, **kwargs) -> EditResponse:
        raise NotImplementedError

    def get_prompt_format(self):
        raise NotImplementedError


class DirectEditModel(EditModel):
    """
    The direct kind of edit model, this class is supposed to be used either with EditCoder or
    with non-chat models, like foundation models.
    """

    def __init__(
        self,
        model_name,
        prompt_format: PromptFormatFunction = direct_edit_prompt,
        post_process: PostProcessFunction = lambda old, new: new,
        stop_tokens=[
            # NOTE(arjun): These are the original stop tokens from the CanItEdit
            # code, which you can verify here:
            #
            # https://github.com/nuprl/CanItEdit/blob/1a87cb488e7ff801cce80550e20822228e1c88fe/benchmark/generate_completions.py#L507
            #
            # However, notice that in the paper and the few-shot prompt, there
            # may not be a ":" after the stop tokens. However, if you see
            # the training code, this is how EditCoder was trained:
            #
            # https://github.com/nuprl/CanItEdit/blob/1a87cb488e7ff801cce80550e20822228e1c88fe/editpackft/format.py#L20
            #
            # I am not going to remove these. But, I am adding new stop tokens below.
            "## Code After:",
            "## Instruction:",
            "## Code Before:",
            "## Test Case:",
            "## Explanation:",
            # NOTE(arjun): new stop tokens for compatibility with AgentPack
            "# Code Before# Code After# Instruction```",
        ],
    ):
        super().__init__()
        self.model_name = model_name
        self.prompt_format = prompt_format
        self.post_process = post_process
        self.stop_tokens = stop_tokens

    async def generate(self, prompt: EditCommand, **kwargs) -> EditResponse:
        assert prompt["instruction"] is not None, "Not implemented yet"
        str_prompt = self.prompt_format(prompt["content"], prompt["instruction"])

        kwargs = kwargs.copy()
        stop = kwargs.pop("stop", [])
        kwargs["stop"] = stop + self.stop_tokens

        # Generate using text_completion directly
        response = await atext_completion(
            model=self.model_name,
            prompt=str_prompt,
            **kwargs,
        )
        generated_text = response.choices[0].text

        # Process response
        try:
            processed = self.post_process(prompt["content"], generated_text)
        except Exception as e:
            # print full stack trace
            import traceback

            traceback.print_exc()
            print("Error in post processing:", e)
            processed = generated_text

        return {"content": processed, "instruction": None}

    def get_prompt_format(self):
        return self.prompt_format


class ChatAdaptorEditModel(EditModel):
    """
    This is an adaptor class to use ChatModels as EditModels.
    NOTE: This model class is only intended for inference, not training.
    """

    def __init__(
        self,
        model_name,
        prompt_format: MessagesFormatFunction = chat_edit_prompt_zeroshot,
        post_process: PostProcessFunction = python_markdown_codeblock_extract,
    ):
        super().__init__()
        self.model_name = model_name
        self.prompt_format = prompt_format
        self.post_process = post_process

    async def generate(self, prompt: EditCommand, **kwargs) -> EditResponse:
        assert prompt["instruction"] is not None, (
            "Every command must have an instruction in ChatAdaptorEditModel"
        )
        response = await acompletion(
            model=self.model_name,
            messages=self.prompt_format(prompt["content"], prompt["instruction"]),
            **kwargs,
        )
        gen = response.choices[0].message.content
        processed = self.post_process(prompt["content"], gen)
        return {"content": processed, "instruction": None}


async def process_example_and_instruction(
    ex: dict,
    instr_kind: str,
    model: EditModel,
    model_kwargs: dict,
    args,
    output_dir: Path,
    batch_sema: asyncio.Semaphore,
    pbar: tqdm,
) -> None:
    """
    Process a single example and instruction kind by generating completions and saving results.

    Args:
        ex: The dataset example containing the code and instructions
        instr_kind: The type of instruction to use ('instruction_descriptive' or 'instruction_lazy')
        model: The model to use for generating completions
        model_kwargs: Additional keyword arguments for model generation
        args: Command line arguments
        output_dir: Directory to save the results
        batch_sema: Semaphore to limit the number of concurrent tasks
    """
    # TODO(arjun): Not real resume. :)
    path = output_dir / f"{ex['full_name']}_{instr_kind}.json.gz"
    if path.exists():
        return  # this pretty much resumes from where it left off

    example = EditCommand(instruction=ex[instr_kind], content=ex["before"])

    async def gen(example):
        """
        Issues a request, but concurrency limited by batch_sema.
        """
        async with batch_sema:
            return await model.generate(example, **model_kwargs)

    completion_tasks = [gen(example) for _ in range(args.completion_limit)]
    completions = await asyncio.gather(*completion_tasks)

    # copy over the example
    result = {}
    for k in ex:
        result[k] = ex[k]

    result["instr_kind"] = instr_kind
    # this is for compatibility with the MultiPL-E evaluator
    result["prompt"] = ""
    result["completions"] = completions
    result["language"] = "py"
    result["temperature"] = args.temperature
    result["top_p"] = args.top_p
    result["max_tokens"] = args.max_tokens
    result["stop_tokens"] = getattr(model, "stop_tokens", [])
    result["script_args"] = args.__dict__.copy()

    gunzip_json_write(path, result)
    pbar.update(1)


async def main(args):
    dataset = datasets.load_dataset(args.dataset, args.subset, split=args.split)

    # Direct model instantiation based on model_type
    if args.model_type == "direct":
        model = DirectEditModel(args.model)
    elif args.model_type == "chat":
        model = ChatAdaptorEditModel(args.model)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    model_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    instr_kinds = ["instruction_descriptive", "instruction_lazy"]
    items = list(itertools.product(dataset, instr_kinds))

    batch_sema = asyncio.Semaphore(args.batch_size)

    pbar = tqdm(total=len(items))

    async with asyncio.TaskGroup() as tg:
        for ex, instr_kind in items:
            tg.create_task(
                process_example_and_instruction(
                    ex,
                    instr_kind,
                    model,
                    model_kwargs,
                    args,
                    Path(args.output_dir),
                    batch_sema=batch_sema,
                    pbar=pbar,
                )
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="nuprl/CanItEdit", help="dataset to use"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="split of the dataset to use"
    )
    parser.add_argument(
        "--subset", type=str, default=None, help="subset of the split to use"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="direct",
        choices=["direct", "chat", "chat_oneshot"],
        help="type of model to use for completions",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="path to model or hub name"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="output directory for completions"
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, help="batch size for completions"
    )
    parser.add_argument(
        "--completion-limit",
        type=int,
        default=20,
        help="number of completions per prompt",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=0.95, help="top-p sampling")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="max new tokens to generate per completion. 2048 works for CanItEdit",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
