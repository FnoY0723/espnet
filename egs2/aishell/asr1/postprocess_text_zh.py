'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2024-08-28 15:47:51
FilePath: /espnet/egs2/aishell/asr1/postprocess_text_zh.py
'''
import sys
import re

in_f = sys.argv[1]
out_f = sys.argv[2]


with open(in_f, "r", encoding="utf-8") as f:
    lines = f.readlines()

with open(out_f, "w", encoding="utf-8") as f:
    for line in lines:
        outs = line.strip().split(" ", 1)
        if len(outs) == 2:
            idx, text = outs
            text = re.sub("</s>", "", text)
            text = re.sub("<s>", "", text)
            text = re.sub("@@", "", text)
            text = re.sub("@", "", text)
            text = re.sub("<unk>", "", text)
            text = re.sub(" ", "", text)
            text = text.lower()
        else:
            idx = outs[0]
            text = " "

        text = [x for x in text]
        text = " ".join(text)
        out = "{} {}\n".format(idx, text)
        f.write(out)