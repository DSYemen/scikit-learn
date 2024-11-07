"""
=======================================================================
رسم سطح القرار لأشجار القرار المدربة على مجموعة بيانات الزهرة الآرغوانية
=======================================================================

ارسم سطح القرار لشجرة قرار مدربة على أزواج
من ميزات مجموعة بيانات الزهرة الآرغوانية.

راجع :ref:`decision tree <tree>` لمزيد من المعلومات حول أداة التقدير.

بالنسبة لكل زوج من ميزات الزهرة الآرغوانية، تتعلم شجرة القرار حدود القرار
المكونة من مجموعات من قواعد العتبة البسيطة المستنبطة من
عينات التدريب.

نحن أيضًا نعرض بنية الشجرة لنموذج مبني على جميع الميزات.
"""

# المؤلفون: مطوري سكايلرن
# معرف الترخيص: BSD-3-Clause

# %%
# قم أولاً بتحميل نسخة مجموعة بيانات الزهرة الآرغوانية المرفقة مع سكايلرن:
from sklearn.datasets import load_iris

iris = load_iris()


# %%
# عرض وظائف القرار للأشجار المدربة على جميع أزواج الميزات.
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier

# المعاملات
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02


for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    # نأخذ فقط الميزتين المقابلتين
    X = iris.data[:, pair]
    y = iris.target

    # التدريب
    clf = DecisionTreeClassifier().fit(X, y)

    # رسم حدود القرار
    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=iris.feature_names[pair[0]],
        ylabel=iris.feature_names[pair[1]],
    )

    # رسم نقاط التدريب
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=iris.target_names[i],
            edgecolor="black",
            s=15,
        )

plt.suptitle("سطح القرار لأشجار القرار المدربة على أزواج من الميزات")
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
_ = plt.axis("tight")

# %%
# عرض بنية شجرة قرار واحدة مدربة على جميع الميزات
# معًا.
from sklearn.tree import plot_tree

plt.figure()
clf = DecisionTreeClassifier().fit(iris.data, iris.target)
plot_tree(clf, filled=True)
plt.title("شجرة القرار المدربة على جميع ميزات الزهرة الآرغوانية")
plt.show()