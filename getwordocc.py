## This script is used to gather the amount of occurences
## It uses the data from http://ai.stanford.edu/~amaas/data/sentiment/index.html
## The output is a csv file, with the amount of occurences and indexed

import os
import pandas as pd
characters_to_remove = ":.?!,'></()-\""
x = {}
def process(x, files):
    for filename in os.listdir(files):
        f = open(files + "/" + filename, "r", encoding="utf8")
        string = f.read()
        ssplit = string.split()
        for word in ssplit:
            if 'br' in word and ('<' in word or '>' in word):
                continue
            for character in characters_to_remove:
                word = word.replace(character, "")
            word = word.lower()
            if len(word) > 1:
                if word in x:
                    x[word] = x[word] + 1
                else:
                    x[word] = 1
    return x
x = process(x, 't/test/neg')
x = process(x, 't/train/neg')
x = process(x, 't/test/pos')
x = process(x, 't/train/pos')
print(x)
x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
print(x)
df = pd.DataFrame.from_dict(x, orient='index')
df.to_csv("occ.csv")