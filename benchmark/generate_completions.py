import datasets
from pathlib import Path
from typing import List, Callable, TypeVar, Literal, Optional, TypedDict
from tqdm import tqdm
import json
import gzip
import torch
import openai
import time
from vllm import LLM, RequestOutput, SamplingParams

T = TypeVar("T")
Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class EditCommand(TypedDict):
    instruction: Optional[str]
    content: str


class EditResponse(TypedDict):
    instruction: Optional[str]
    content: str


# (old, instr, new) -> prompt
PromptFormatFunction = Callable[[str, str, str], str]

# (old, instr) -> [messages]
MessagesFormatFunction = Callable[[str, str], List[Message]]

# (old, new) -> response
PostProcessFunction = Callable[[str, str], str]


def starcoder_edit_prompt(old, instr, new):
    # starcoder tokens
    OLD_CODE_TOKEN = "<commit_before>"
    REFLECTION_TOKEN = "<commit_msg>"
    NEW_CODE_TOKEN = "<commit_after>"
    return OLD_CODE_TOKEN + old + REFLECTION_TOKEN + instr + NEW_CODE_TOKEN + new


def codellama_edit_prompt(old, instr, new, codeblock_before: Optional[str] = None, codeblock_after: Optional[str] = None):
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
        {"role": "system", "content": """
You are PythonEditGPT. You will be provided the original code snippet and an instruction that specifies the changes you need to make. You will produce the changed code, based on the original code and the instruction given. Only produce the code, do not include any additional prose.
             """.strip()},
        {"role": "user", "content": """
## Code Before
```py
def add(a, b):
    return a + b
```

## Instruction
Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.
""".strip()},
        {"role": "assistant", "content": """
## Code After
```py
def add(x, y):
    \"\"\"Adds two numbers.\"\"\"
    return x + y

def sub(x, y):
    \"\"\"Subtracts two numbers.\"\"\"
    return x - y
```
         """.strip()},
        {"role": "user", "content": f"""
## Code Before
```py
{old}
```
## Instruction
{instr}
""".strip()},
    ]


def python_markdown_codeblock_extract(_: str, new: str) -> str:
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
    return buf


def autodetect_dtype() -> str:
    if torch.cuda.is_bf16_supported():
        return "bfloat16"
    else:
        return "auto"


class ChatModel:
    def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        raise NotImplementedError


class EditModel:
    def __init__(self, before_content_tok, instruction_tok, after_content_tok):
        self.before_content_tok = before_content_tok
        self.instruction_tok = instruction_tok
        self.after_content_tok = after_content_tok

    def edit_model_generate(self, model: LLM, str_prompts: List[str], **kwargs) -> List[RequestOutput]:
        kwargs_gen = kwargs.copy()
        if "declaration" in kwargs_gen:
            del kwargs_gen["declaration"]
        use_tqdm = kwargs_gen.pop("use_tqdm", False)
        gens = model.generate(
            prompts=str_prompts,
            sampling_params=SamplingParams(
                top_p=kwargs_gen.pop("top_p", 0.95),
                temperature=kwargs_gen.pop("temperature", 0.2),
                max_tokens=kwargs_gen.pop("max_tokens", 1024),
                **kwargs_gen
            ),
            use_tqdm=use_tqdm,
        )
        return gens

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        raise NotImplementedError


class StarCoderCommitEditModel(EditModel):
    def __init__(
        self,
        model_name="bigcode/starcoderbase",
        num_gpus=1,
        before_content_tok="<commit_before>",
        instruction_tok="<commit_msg>",
        after_content_tok="<commit_after>",
    ):
        super().__init__(before_content_tok, instruction_tok, after_content_tok)
        self.model = LLM(
            model_name,
            dtype=autodetect_dtype(),
            tensor_parallel_size=num_gpus,
        )
        self.tokenizer = self.model.get_tokenizer()
        self.instruction_tok_id = self.tokenizer.encode(instruction_tok)[0]
        self.after_content_tok_id = self.tokenizer.encode(after_content_tok)[0]

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        str_prompts = []

        for prompt in prompts:
            content = ("\n" + prompt["content"] +
                       "\n") if prompt["content"] != "" else ""
            if prompt["instruction"] is not None:
                str_prompt = f"{self.before_content_tok}{content}{self.instruction_tok}\n{prompt['instruction']}\n{self.after_content_tok}"
                if "declaration" in kwargs:
                    str_prompt += f"\n{kwargs['declaration']}"
            else:
                str_prompt = f"{self.before_content_tok}{content}{self.instruction_tok}"

            str_prompts.append(str_prompt)

        # generate
        gens = self.edit_model_generate(self.model, str_prompts, **kwargs)

        responses = []

        for prompt, gen in zip(prompts, gens):
            out = gen.outputs[0].token_ids

            resp = {"content": "", "instruction": None}
            # if we had an instruction, we are all good.
            # or, it could be that the model didn't generate anything useful
            if prompt["instruction"] is not None or self.after_content_tok_id not in out:
                resp["content"] = self.tokenizer.decode(
                    out, skip_special_tokens=True)
                responses.append(resp)
                continue

            # otherwise, find the end of the instruction
            new_content_idx = out.index(self.after_content_tok_id)
            resp["instruction"] = self.tokenizer.decode(
                out[:new_content_idx], skip_special_tokens=True)
            # and decode the content
            resp["content"] = self.tokenizer.decode(
                out[new_content_idx+1:], skip_special_tokens=True)
            responses.append(resp)

        return responses

class CodeLlamaEditModel(EditModel):
    # TODO: implement whole shebang for bugfix
    def __init__(
        self,
        model_name="codellama/CodeLlama-34b-hf",
        num_gpus=1,
        gpu_util=0.95,
        prompt_format: PromptFormatFunction = codellama_edit_prompt,
        post_process: PostProcessFunction = lambda old, new: new,
    ):
        super().__init__("", "", "")
        self.model = LLM(
            model_name,
            max_model_len=16384,  # NOTE: this is for compatibility with DeepSeek
            dtype=autodetect_dtype(),
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_util,
        )
        self.prompt_format = prompt_format
        self.post_process = post_process

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        str_prompts = []

        for prompt in prompts:
            declaration = kwargs["declaration"] if "declaration" in kwargs else ""
            assert prompt["instruction"] is not None, "Not implemented yet"
            str_prompts.append(self.prompt_format(
                prompt["content"], prompt["instruction"], declaration)
            )

        # generate
        gens = self.edit_model_generate(self.model, str_prompts, **kwargs)

        kwargs = kwargs.copy()
        stop = kwargs.pop("stop", [])
        kwargs["stop"] = stop + ["## Code Before:", "## Instruction:", "## Code After:"]
        responses = []
        for prompt, gen in zip(prompts, gens):
            out = gen.outputs[0].text
            try:
                processed = self.post_process(prompt["content"], out)
            except Exception as e:
                print("Error in post processing:", e)
                processed = out
            resp = {"content": processed, "instruction": None}
            responses.append(resp)

        return responses

class OctoCoderChatModel(ChatModel):

    def __init__(
        self,
        model_name="bigcode/octocoder",
        num_gpus=1,
        gpu_util=0.95,
        quantization=False,
    ):
        self.model = LLM(
            model_name,
            dtype=autodetect_dtype() if not quantization else "float16",
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_util,
            quantization="awq" if quantization else None,
        )
        self.tokenizer = self.model.get_tokenizer()

    def fmt_msg(self, message: List[Message]) -> str:

        fmt = []
        start = 0
        system = ""

        # check for system prompt
        if message[0]['role'] == "system":
            start = 1
            system = message[0]['content']
        
        for i in range(start, len(message)): 

            current = message[i]
            assert current['content'] is not None, "Content of a message cannot be null"
            assert current['role'] is not None, "Role of a message cannot be null"
            if current["role"] == "user":
                # if question, then add system prompt
                fmt.append(f"Question: {system}\n{current['content']}")
                # if last message and is a question, add an answer to it
                if i == len(message) - 1:
                    fmt.append(f"Answer:")
            else:
                # if answer, then no system prompt
                fmt.append(f"Answer: {current['content']}")

        return "\n\n".join(fmt)

    def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        
        kwargs_gen = kwargs.copy()

        msgs = [self.fmt_msg(msg) for msg in messages]

        stop = kwargs_gen.pop("stop", [])
        stop.append("\n\nAnswer:")
        stop.append("\n\nQuestion:")
        # stop.append("\n\n")

        gens = self.model.generate(
            prompts=msgs,
            sampling_params=SamplingParams(
                top_p=kwargs_gen.pop("top_p", 0.95),
                temperature=kwargs_gen.pop("temperature", 0.2),
                max_tokens=kwargs_gen.pop("max_tokens", 1024),
                stop=list(set(stop)),
                **kwargs_gen
            ),
            use_tqdm=True,
        )

        responses = []

        for gen in gens:
            print(gen)
            out = gen.outputs[0].text
            responses.append(out)

        return responses

class LlamaChatModel(ChatModel):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    def __init__(
        self,
        model_name="codellama/CodeLlama-34b-Instruct-hf",
        num_gpus=1,
        gpu_util=0.95,
        quantization=False,
    ):
        self.model = LLM(
            model_name,
            dtype=autodetect_dtype() if not quantization else "float16",
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_util,
            quantization="awq" if quantization else None,
        )
        self.tokenizer = self.model.get_tokenizer()

    def llama2_chat_generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        def tokenize_messages(ms) -> List[int]:
            toks = self.tokenizer.apply_chat_template(
                ms,  # type: ignore
                tokenize=True,
                truncation=True,
                max_length=16384 - max_new_tokens - 2,
            )
            assert isinstance(toks, list)
            return toks
        kwargs = kwargs.copy()
        max_new_tokens = kwargs.pop("max_tokens", 256)
        prompts = [
            tokenize_messages(ms)
            for ms in messages
        ]

        discard_lengthy = kwargs.pop("discard_lengthy", False)
        use_tqdm = kwargs.pop("use_tqdm", False)
        stop = kwargs.pop("stop", [])
        stop.append(self.E_INST)
        params = SamplingParams(
            top_p=kwargs.pop("top_p", 0.9),
            temperature=kwargs.pop("temperature", 0.75),
            max_tokens=max_new_tokens,
            stop=list(set(stop)),
            **kwargs
        )
        gens = self.model.generate(
            prompt_token_ids=prompts,
            sampling_params=params,
            use_tqdm=use_tqdm,
        )
        decoded = []
        for gen in gens:
            outs = gen.outputs[0]
            if discard_lengthy and outs.finish_reason == "length":
                continue
            toks = outs.token_ids
            dec = self.tokenizer.decode(toks, skip_special_tokens=True)

            found = dec.find(self.E_INST)
            if found != -1:
                dec = dec[:found]
            found = dec.find(self.B_INST)
            if found != -1:
                dec = dec[:found]

            stripped = dec.strip()
            decoded.append(stripped)
        return decoded

    def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        # make sure we have a list of lists
        assert isinstance(messages, list), "messages must be a list of lists"
        assert len(messages) > 0, "messages must have at least one list"
        assert isinstance(
            messages[0], list), "messages must be a list of lists"
        return self.llama2_chat_generate(messages, **kwargs)


class OpenAIChatModel(ChatModel):
    def __init__(
        self,
        model_name="gpt-4",
        endpoint=None,
    ):
        import os
        import openai
        if "ORG_ID" in os.environ:
            openai.organization = os.getenv("ORG_ID")
        assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY must be set"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        if endpoint is not None:
            # TODO: fix this
            #  openai.api_base = endpoint
            pass

    def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        # make sure we have a list of lists
        assert isinstance(messages, list), "messages must be a list of lists"
        assert len(messages) > 0, "messages must have at least one list"
        assert isinstance(
            messages[0], list), "messages must be a list of lists"
        # check that all messages are the same.
        # TODO: support heterogeneous messages
        message = messages[0]
        for m in messages[1:]:
            assert message == m, "OpenAI chat model only supports homogeneous messages"

        while True:
            _kwargs = kwargs.copy()
            discard_lengthy = _kwargs.pop("discard_lengthy", False)

            try:
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=message,  # type: ignore
                    n=len(messages),
                    stop=_kwargs.pop("stop", None),
                    temperature=_kwargs.pop("temperature", 0.75),
                    top_p=_kwargs.pop("top_p", 0.9),
                    max_tokens=_kwargs.pop("max_tokens", 256),
                    **_kwargs
                )
            except openai.RateLimitError as e:
                print("Rate limit error. Waiting two minutes:", e)
                time.sleep(120)
                continue

            break
        outs = []

        for choice in response.choices:
            if discard_lengthy and choice.finish_reason == "length":
                continue
            text = choice.message.content
            outs.append(text.strip())

        return outs


class ChatAdaptorEditModel(EditModel):
    """
    This is an adaptor class to use ChatModels as EditModels.
    NOTE: This model class is only intended for inference, not training.
    """
    def __init__(
        self,
        chat_model: ChatModel,
        prompt_format: MessagesFormatFunction = openai_edit_prompt_1shot,
        post_process: PostProcessFunction = python_markdown_codeblock_extract,
    ):
        super().__init__("", "", "")  # NOTE: hence why can be used only for inference
        self.model = chat_model
        self.prompt_format = prompt_format
        self.post_process = post_process

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        kwargs = kwargs.copy()
        kwargs.pop("declaration", None)

        kwargs.pop("use_tqdm", None)

        chat_prompts = []
        for prompt in prompts:
            assert prompt["instruction"] is not None, "Every command must have an instruction in ChatAdaptorEditModel"
            chat_prompts.append(self.prompt_format(
                prompt["content"], prompt["instruction"])
            )

        # generate
        gens = self.model.generate(chat_prompts, **kwargs)

        responses = []
        for prompt, gen in zip(prompts, gens):
            processed = self.post_process(prompt["content"], gen)
            resp = {"content": processed, "instruction": None}
            responses.append(resp)

        return responses

def batch_prompts_from_example(example: T, batch_size: int, completion_limit: int) -> List[List[T]]:
    prompts = [example] * completion_limit
    num_batches = completion_limit // batch_size
    batches = [prompts[i * batch_size: (i + 1) * batch_size]
               for i in range(num_batches)]
    # the last batch may be smaller
    if len(prompts) % batch_size != 0:
        batches.append(prompts[num_batches * batch_size:])

    return batches

def gunzip_json_write(path: Path, data: dict) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(data, f)

# NOTE: this is the factory for each model type. to add a new model type, add a new case here
# and implement it in models.py. Also, add a new case in the argument parser below.
def model_factory(model_type: str) -> Callable[[str], EditModel]:
    if model_type == "codellama" or model_type == "deepseek":
        return CodeLlamaEditModel
    elif model_type == "starcoder":
        return StarCoderCommitEditModel
    elif model_type == "openai":
        return (lambda path: ChatAdaptorEditModel(OpenAIChatModel(path)))
    elif model_type == "codellama-chat":
        return (lambda path: ChatAdaptorEditModel(LlamaChatModel(path)))
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def complete_problem(example: EditCommand, model: EditModel, batch_size: int, completion_limit: int, **kwargs) -> List[str]:
    batches = batch_prompts_from_example(example, batch_size, completion_limit)

    completions = []
    for batch in batches:
        resps = model.generate(batch, **kwargs)
        for resp in resps:
            completions.append(resp["content"])

    return completions


def main(args):
    dataset = datasets.load_dataset(args.dataset, split=args.split)
    model = model_factory(args.model_type)(args.model)
    model_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    # writing in this format such that we can use the MultiPL-E evaluation container :)
    for ex in tqdm(dataset, total=len(dataset)):  # type: ignore
        instr_kinds = ['instruction_descriptive', 'instruction_humane']
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
            result["prompt"] = ""  # this is for compatibility with MultiPL-E
            result["completions"] = completions
            result["language"] = "py"
            result["temperature"] = args.temperature
            result["top_p"] = args.top_p
            result["max_tokens"] = args.max_tokens
            result["stop_tokens"] = []
            result["script_args"] = args.__dict__.copy()

            gunzip_json_write(path, result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="nuprl/CanItEdit")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--model-type",
        type=str,
        default="codellama",
        choices=["codellama", "codellama-diff",
                 "codellama-chat", "openai", "starcoder"],
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--completion-limit", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    main(args)
