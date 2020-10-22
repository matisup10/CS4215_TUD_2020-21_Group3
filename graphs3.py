# Plots figure 7 and an additional plot using time on the y axis

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def fromResults():
    df = pd.read_csv("data2/maxwordsimage-instance-2.csv", delimiter=";")
    df.columns = ["accuracy", "time", "model_type", "epochs", "iteration_number", "n_test", "n_train", "max_words", "min_words"]
    df1 = pd.read_csv("data2/maxwordsimage-instance-3.csv", delimiter=";")
    df1.columns = ["accuracy", "time", "model_type", "epochs", "iteration_number", "n_test", "n_train", "max_words", "min_words"]
    df2 = pd.read_csv("data2/testmaxwords-instance-1.csv", delimiter=";")
    df2.columns = ["accuracy", "time", "model_type", "epochs", "iteration_number", "n_test", "n_train", "max_words", "min_words"]
    print("df")
    df = pd.concat([df, df1, df2], ignore_index=True)
    return df

def addDivision(df):
    df['accuracy/time'] = df['accuracy'] / df['time']
    return df

df = fromResults()
df = addDivision(df)
fig = plt.figure()
ax = plt.subplot(111)
ax1 = sns.swarmplot(x="max_words", y="accuracy", hue="n_train", data=df, size=4, dodge=True)
# ax = sns.boxplot(x="n_train", y="accuracy", hue="epochs",
#                   data=df, palette="Set3")
plt.grid(axis='both')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
# ax2 = sns.swarmplot(x="max_words", y="time", hue="n_train", data=df, size=4)
# plt.grid(axis='both')
# plt.show()
#
# ax3 = sns.swarmplot(x="n_train", y="accuracy/time", hue="max_words", data=df, size=4)
# plt.grid(axis='both')
# plt.show()

# ax4 = sns.regplot(x="n_train", y="accuracy", data=df[df.epochs.eq(1)],
#                   x_jitter=.1, scatter_kws={'s':14}, order=4, ci=None)
# ax4 = sns.regplot(x="n_train", y="accuracy", data=df[df.epochs.eq(2)],
#                   x_jitter=.1, scatter_kws={'s':14}, order=4, ci=None)
# ax4 = sns.regplot(x="n_train", y="accuracy", data=df[df.epochs.eq(3)],
#                   x_jitter=.1, scatter_kws={'s':14}, order=4, ci=None)
# ax4 = sns.regplot(x="n_train", y="accuracy", data=df[df.epochs.eq(4)],
#                   x_jitter=.1, scatter_kws={'s':14}, order=4, ci=None)
# plt.grid(axis='both')
# plt.show()
#
# ax4 = sns.regplot(x="n_train", y="time", data=df[df.epochs.eq(1)],
#                   x_jitter=.1, scatter_kws={'s':14}, order=1, ci=None)
# ax4 = sns.regplot(x="n_train", y="time", data=df[df.epochs.eq(2)],
#                   x_jitter=.1, scatter_kws={'s':14}, order=1, ci=None)
# ax4 = sns.regplot(x="n_train", y="time", data=df[df.epochs.eq(3)],
#                   x_jitter=.1, scatter_kws={'s':14}, order=1, ci=None)
# ax4 = sns.regplot(x="n_train", y="time", data=df[df.epochs.eq(4)],
#                   x_jitter=.1, scatter_kws={'s':14}, order=1, ci=None)
# plt.grid(axis='both')
# plt.show()