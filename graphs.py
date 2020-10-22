# Plots figure 4 en 5

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os import path
def fromFiles():
    result = []
    for epoch in range(1,4):
        for model in ['cnn', 'cnn_lstm', 'gru', 'lstm']:
            for it in range(0,6):
                pathstr = "data/" + str(model) + "-" + str(epoch) + "-" + str(it) + "-output.txt"
                if path.exists(pathstr):
                    f = open(pathstr, "r")
                    string = f.read().split()
                    result.append((model, str(it), float(string[0]), float(string[1]), float(float(string[0])/float(string[1])), str(epoch)))
    print(result)
    df = pd.DataFrame(result)
    df.columns = ["model_type", "it", "accuracy", "time", "accuracy/time", "epoch"]
    return df

def fromResults():
    df = pd.read_csv("results.csv", delimiter=";")
    df.columns = ["accuracy", "time", "model_type", "epochs", "iteration_number", "n_test", "n_train", "max_words"]
    return df

def addDivision(df):
    df['accuracy/time'] = df['accuracy'] / df['time']
    return df

df = fromResults()
df = addDivision(df)

ax1 = sns.swarmplot(x="n_train", y="accuracy", hue="epochs", data=df, size=4, dodge=True)
# ax = sns.boxplot(x="n_train", y="accuracy", hue="epochs",
#                   data=df, palette="Set3")
plt.grid(axis='both')
plt.savefig("3.png")
plt.show()

ax2 = sns.swarmplot(x="n_train", y="time", hue="epochs", data=df, size=4)
plt.grid(axis='both')
plt.ylabel("time (s)")
plt.savefig("4.png")
plt.show()

ax3 = sns.swarmplot(x="n_train", y="accuracy/time", hue="epochs", data=df, size=4)
plt.grid(axis='both')

plt.show()

ax4 = sns.regplot(x="n_train", y="accuracy", data=df[df.epochs.eq(1)],
                  x_jitter=.1, scatter_kws={'s':14}, order=4, ci=None)
ax4 = sns.regplot(x="n_train", y="accuracy", data=df[df.epochs.eq(2)],
                  x_jitter=.1, scatter_kws={'s':14}, order=4, ci=None)
ax4 = sns.regplot(x="n_train", y="accuracy", data=df[df.epochs.eq(3)],
                  x_jitter=.1, scatter_kws={'s':14}, order=4, ci=None)
ax4 = sns.regplot(x="n_train", y="accuracy", data=df[df.epochs.eq(4)],
                  x_jitter=.1, scatter_kws={'s':14}, order=4, ci=None)
plt.grid(axis='both')
plt.show()

ax4 = sns.regplot(x="n_train", y="time", data=df[df.epochs.eq(1)],
                  x_jitter=.1, scatter_kws={'s':14}, order=1, ci=None)
ax4 = sns.regplot(x="n_train", y="time", data=df[df.epochs.eq(2)],
                  x_jitter=.1, scatter_kws={'s':14}, order=1, ci=None)
ax4 = sns.regplot(x="n_train", y="time", data=df[df.epochs.eq(3)],
                  x_jitter=.1, scatter_kws={'s':14}, order=1, ci=None)
ax4 = sns.regplot(x="n_train", y="time", data=df[df.epochs.eq(4)],
                  x_jitter=.1, scatter_kws={'s':14}, order=1, ci=None)
plt.grid(axis='both')
plt.show()