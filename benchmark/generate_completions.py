import datasets
from pathlib import Path
from tqdm import tqdm
import torch
from typing import List, Literal, Optional, TypedDict, Callable, Union, TypeVar
import gzip
import json
from litellm import completion, text_completion, batch_completion

#its a batch generator for code-edit edits.

def gunzip_json_write(path: Path, data: dict) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


T = TypeVar("T")


def batch_prompts_from_example(example: T, batch_size: int, completion_limit: int) -> List[List[T]]:
    prompts = [example] * completion_limit
    num_batches = completion_limit // batch_size
    batches = [prompts[i * batch_size: (i + 1) * batch_size]
               for i in range(num_batches)]
    # the last batch may be smaller
    if len(prompts) % batch_size != 0:
        batches.append(prompts[num_batches * batch_size:])

    return batches


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str
    # the prefix for the assistant's response. this is only used for the
    # last message in a conversation, and is ignored otherwise.
    # NOTE: leaving commented for python 3.10 compatibility
    #  prefix_after: NotRequired[str]


#this is one edit request
class EditCommand(TypedDict):
    instruction: Optional[str]
    content: str

#this is model's output
class EditResponse(TypedDict):
    instruction: Optional[str]
    content: str


# (old, instr, new) -> prompt
PromptFormatFunction = Callable[[str, str, str], str]

# (old, instr) -> [messages]
MessagesFormatFunction = Callable[[str, str], List[Message]]

# (old, new) -> response
PostProcessFunction = Callable[[str, str], str]


CompletionEngine = Literal["litellm"]

def direct_edit_prompt(
    old,
    instr,
    new,
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
    after = f"""## Code After:\n{new}"""
    return before + instr + after

def openai_edit_prompt_1shot(old: str, instr: str) -> List[Message]:
    return [
        {
            "role": "system",
            "content": """
You are PythonEditGPT. You will be provided the original code snippet and an instruction that specifies the changes you need to make. You will produce the changed code, based on the original code and the instruction given. Only produce the code, do not include any additional prose.
             """.strip(),
        },
        {
            "role": "user",
            "content": """
## Code Before
```py
def add(a, b):
    return a + b
```

## Instruction
Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.
""".strip(),
        },
        {
            "role": "assistant",
            "content": """
## Code After
```py
def add(x, y):
    \"\"\"Adds two numbers.\"\"\"
    return x + y

def sub(x, y):
    \"\"\"Subtracts two numbers.\"\"\"
    return x - y
```
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
""".strip(),
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


def autodetect_dtype() -> str:
    if torch.cuda.is_bf16_supported():
        return "bfloat16"
    else:
        return "auto"

class LiteLLMChat:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    def generate(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[str]:
        responses = []
        response = batch_completion(
            model=self.model_name,
            api_base="http://localhost:8000/v1",
            messages=prompts,
            **kwargs,
        )

        for res in response:
            generated_text = res.choices[0]['message']['content']
            responses.append(generated_text)

        return responses

class LiteLLMBase:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    def generate(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[str]:
        responses = []
        for prompt in prompts:
            response = text_completion(
                model=self.model_name,
                api_base="http://localhost:8000/v1",
                prompt=prompt,
                **kwargs,
            )

            generated_text = response.choices[0].text
            responses.append(generated_text)

        return responses


class EditModel:
    def __init__(self, before_content_tok=None, instruction_tok=None, after_content_tok=None):
        self.before_content_tok = before_content_tok
        self.instruction_tok = instruction_tok
        self.after_content_tok = after_content_tok

    def edit_model_generate(
        self,
        model: Union[LiteLLMBase, LiteLLMChat],
        str_prompts: List[str], **kwargs
    ) -> List[str]:
        kwargs_gen = kwargs.copy()
        if "declaration" in kwargs_gen:
            del kwargs_gen["declaration"]
        use_tqdm = kwargs_gen.pop("use_tqdm", False)
        gens = model.generate(
            prompts=str_prompts,
            top_p=kwargs_gen.pop("top_p", 0.95),
            temperature=kwargs_gen.pop("temperature", 0.2),
            max_tokens=kwargs_gen.pop("max_tokens", 1024),
            use_tqdm=use_tqdm,
            **kwargs_gen,
        )
        return gens

    def get_before_content_tok(self) -> Optional[str]:
        return self.before_content_tok

    def get_instruction_tok(self) -> Optional[str]:
        return self.instruction_tok

    def get_after_content_tok(self) -> Optional[str]:
        return self.after_content_tok

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        raise NotImplementedError

    def bugfix_instr(self, prompt) -> Optional[str]:
        return None

    def get_prompt_format(self):
        raise NotImplementedError

    def get_tokenizer(self):
        raise NotImplementedError


class DirectEditModel(EditModel):
    """
    The direct kind of edit model, this class is supposed to be used either with EditCoder or
    with non-chat models, like foundation models.
    """

    def __init__(
        self,
        model_name,
        num_gpus=1,
        gpu_util=0.95,
        prompt_format: PromptFormatFunction = direct_edit_prompt,
        post_process: PostProcessFunction = lambda old, new: new,
        completion_engine: CompletionEngine = "litellm",
        stop_tokens: List[str] = ["## Code After:",
                                  "## Instruction:", "## Code Before:", "## Test Case:", "## Explanation:"],
        max_model_len=None,
    ):
        super().__init__()
        self.model = LiteLLMBase(
            model_name,
            dtype=autodetect_dtype()
        )
        self.prompt_format = prompt_format
        self.post_process = post_process
        self.stop_tokens = stop_tokens

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        str_prompts = []

        for prompt in prompts:
            declaration = kwargs["declaration"] if "declaration" in kwargs else ""
            assert prompt["instruction"] is not None, "Not implemented yet"
            str_prompts.append(
                self.prompt_format(
                    prompt["content"], prompt["instruction"], declaration
                )
            )

        kwargs = kwargs.copy()
        stop = kwargs.pop("stop", [])
        kwargs["stop"] = stop + self.stop_tokens

        # generate
        gens = self.edit_model_generate(self.model, str_prompts, **kwargs)

        responses = []
        for prompt, gen in zip(prompts, gens):
            out = gen
            try:
                processed = self.post_process(prompt["content"], out)
            except Exception as e:
                # print full stack trace
                import traceback
                traceback.print_exc()
                print("Error in post processing:", e)
                processed = out
            resp = {"content": processed, "instruction": None}
            responses.append(resp)

        return responses

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
        prompt_format: MessagesFormatFunction = openai_edit_prompt_1shot,
        post_process: PostProcessFunction = python_markdown_codeblock_extract,
    ):
        super().__init__()
        self.model = LiteLLMChat(
            model_name,
            dtype=autodetect_dtype(),
        )
        self.prompt_format = prompt_format
        self.post_process = post_process

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        kwargs = kwargs.copy()
        # TODO: can do something with declaration here
        kwargs.pop("declaration", None)

        kwargs.pop("use_tqdm", None)

        chat_prompts = []
        for prompt in prompts:
            assert (
                prompt["instruction"] is not None
            ), "Every command must have an instruction in ChatAdaptorEditModel"
            chat_prompts.append(
                self.prompt_format(prompt["content"], prompt["instruction"])
            )

        # generate
        gens = self.model.generate(chat_prompts, **kwargs)

        responses = []
        for prompt, gen in zip(prompts, gens):
            processed = self.post_process(prompt["content"], gen)
            resp = {"content": processed, "instruction": None}
            responses.append(resp)

        return responses


# NOTE: this is the factory for each model type. to add a new model type, add a new case here
# and implement it in models.py. Also, add a new case in the argument parser below.
def model_factory(
        model_type: str,
        quantize=False,
        num_gpus=1,
        system_supported=True,
        completion_engine: CompletionEngine = "litellm",
        max_model_len=None,
) -> Callable[[str], EditModel]:
    if model_type == "direct":
        return (lambda path: DirectEditModel(
            path,
            completion_engine=completion_engine,
            num_gpus=num_gpus,
            max_model_len=max_model_len,
        ))
    elif model_type == "chat":
        return (lambda path: ChatAdaptorEditModel(
            path,
        ))


def complete_problem(example: EditCommand, model: EditModel, batch_size: int, completion_limit: int, **kwargs) -> List[str]:
    batches = batch_prompts_from_example(example, batch_size, completion_limit)

    completions = []
    for batch in batches:
        resps = model.generate(batch, **kwargs)
        for resp in resps:
            completions.append(resp["content"])

    return completions


def main(args):
    dataset = datasets.load_dataset(
        args.dataset, args.subset, split=args.split)
    model = model_factory(
        args.model_type,
        quantize=args.quantize,
        num_gpus=args.num_gpus,
        system_supported=not args.no_system,
        completion_engine=args.completion_engine,
        max_model_len=args.max_model_len,
    )(args.model)
    model_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    # writing in this format such that we can use the MultiPL-E evaluation container :)
    for ex in tqdm(dataset, total=len(dataset)):  # type: ignore
        assert isinstance(ex, dict)

        if args.humanevalpack:
            instr_kinds = ['instruction']
        else:
            instr_kinds = ['instruction_descriptive', 'instruction_lazy']

        for instr_kind in instr_kinds:
            path = Path(args.output_dir) / \
                (f"{ex['full_name']}_{instr_kind}.json.gz")
            if path.exists():
                continue  # this pretty much resumes from where it left off

            instr = ex[instr_kind]
            example = EditCommand(
                instruction=instr,
                content=ex["before"],
            )

            if "declaration" in ex:
                model_kwargs["declaration"] = ex["declaration"]

            completions = complete_problem(
                example,
                model,
                args.batch_size,
                args.completion_limit,
                **model_kwargs,
            )

            # copy over the example
            result = {}
            for k in ex:
                result[k] = ex[k]

            result["instr_kind"] = instr_kind
            # this is for compatibility with the MultiPL-E evaluator
            result["prompt"] = ex["declaration"] if "declaration" in ex else ""
            result["completions"] = completions
            result["language"] = "py"
            result["temperature"] = args.temperature
            result["top_p"] = args.top_p
            result["max_tokens"] = args.max_tokens
            result["stop_tokens"] = getattr(model, "stop_tokens", [])
            result["script_args"] = args.__dict__.copy()

            gunzip_json_write(path, result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="nuprl/CanItEdit", help="dataset to use")
    parser.add_argument("--split", type=str, default="test",
                        help="split of the dataset to use")
    parser.add_argument("--subset", type=str, default=None,
                        help="subset of the split to use")
    parser.add_argument(
        "--model-type",
        type=str,
        default="direct",
        choices=["direct","chat"],
        help="type of model to use for completions",
    )
    parser.add_argument(
        "--completion-engine",
        type=str,
        default="litellm",
        choices=["litellm"],
    )
    parser.add_argument("--model", type=str, required=True,
                        help="path to model or hub name")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="output directory for completions")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="batch size for completions")
    parser.add_argument("--completion-limit", type=int,
                        default=20, help="number of completions per prompt")
    parser.add_argument("--temperature", type=float,
                        default=0.2, help="sampling temperature")
    parser.add_argument("--quantize", action="store_true",
                        help="quantize the model with AWQ")
    parser.add_argument("--humanevalpack", action="store_true",
                        help="run humanevalpack instead of CanItEdit")
    parser.add_argument("--top-p", type=float,
                        default=0.95, help="top-p sampling")
    parser.add_argument("--max-tokens", type=int,
                        default=2048, help="max new tokens to generate per completion. 2048 works for CanItEdit")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="max model length for batching with vLLM. only change if getting OOM")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="number of gpus for sharded model")
    parser.add_argument("--no-system", action="store_true",
                        help="disable system prompt for chat models")
    args = parser.parse_args()
    main(args)
