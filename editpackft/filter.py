import datasets
import editdistance
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--push", type=str, required=True, help="HuggingFace repo to push to")
args = parser.parse_args()

ds = datasets.load_dataset("bigcode/commitpackft", "python", split="train")

def cleanup(code):
    lines = code.split("\n")
    d = 0
    in_docstring = False
    for i, l in enumerate(lines):
        dsc = l.count('"""')
        if dsc == 1:
            if in_docstring:
                d = i + 1
                break
            else:
                in_docstring = True
        elif not in_docstring and l.startswith("#"):
            d = i + 1
        elif not in_docstring and l.strip() != "":
            break

    return "\n".join(lines[d:])

exs = []
blanks = 0
no_diff = 0
bad_comments = 0
syntax_errors = 0
badwords = ["TODO", "FIXME", "BUG", "FIX ME", "from setuptools import", "import setuptools"]
badwords_lower = [b.lower() for b in badwords]
badwords.extend(badwords_lower)
for ex in ds:
    ex["new_contents"] = cleanup(ex["new_contents"])
    ex["old_contents"] = cleanup(ex["old_contents"])
    if ex["old_contents"].strip() == "" or ex["new_contents"].strip() == "": # TODO: maybe we want to keep these?
        blanks += 1
        continue

    skip = False
    for b in badwords:
        if b in ex["new_contents"]:
            skip = True
            break

    if skip:
        bad_comments += 1
        continue
    
    dist = editdistance.eval(ex["old_contents"], ex["new_contents"])
    if dist < 10:
        no_diff += 1
        continue

    try:
        ast.parse(ex["new_contents"])
    except:
        syntax_errors += 1
        continue
    
    exs.append(ex)

print("blanks", blanks)
print("no_diff", no_diff)
print("bad_comments", bad_comments)
print("syntax_errors", syntax_errors)
print("no_diff", no_diff)
print("size", len(exs))
new_ds = datasets.Dataset.from_list(exs)
new_ds.push_to_hub(args.push)
