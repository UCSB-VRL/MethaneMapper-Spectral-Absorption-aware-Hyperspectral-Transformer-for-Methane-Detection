import numpy as np
import os
import json

path = "./train2020.json"

f = open(path)
data = json.load(f)


def main():
    annotations = data["annotations"].copy()
    list_len = len(annotations)
    idx = 0
    while idx < list_len:
        num_samples = len(annotations[idx]["category_id"])
        if num_samples == 0:
            del annotations[idx]
            list_len = len(annotations)
            continue
        idx = idx + 1
    del data["annotations"]
    data["annotations"] = annotations
    with open("train2022_2.json", "w") as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
    main()
