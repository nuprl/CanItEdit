from tqdm import tqdm
import datasets
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True)
parser.add_argument("--push", type=str, required=True)
parser.add_argument("--instr_col", type=str, default="instruction")
parser.add_argument("--old_col", type=str, default="old_contents")
parser.add_argument("--new_col", type=str, default="new_contents")
parser.add_argument("--codeblock", action="store_true")
args = parser.parse_args()

dataset = datasets.load_dataset(args.dataset, split="train")

def remove_windows_newlines(s):
    return s.replace("\r\n", "\n")

def edit_prompt(old, instr, new):
    before = f"""## Code Before:\n{old}\n"""
    instr = f"""## Instruction:\n{instr}\n"""
    after = f"""## Code After:\n{new}"""
    return before + instr + after



content = []
exs = []

for ex in tqdm(dataset, total=len(dataset)):
    old = remove_windows_newlines(ex[args.old_col])
    new = remove_windows_newlines(ex[args.new_col])
    instr = remove_windows_newlines(ex[args.instr_col])
    prompt = edit_prompt(old, instr, new)
    exs.append(ex)
    content.append(prompt)


ds = datasets.Dataset.from_list(exs)
if "content" in ds.column_names:
    ds = ds.remove_columns("content")
ds = ds.add_column("content", content)
print("#### Sample prompt: ####")
print(ds[0]["content"])
ds.push_to_hub(args.push, private=True)
