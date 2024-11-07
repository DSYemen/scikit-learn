"""
============================
مثال على هوامش SVM
============================
توضح المخططات أدناه تأثير المعامل `C`
على خط الفصل. تشير قيمة كبيرة من `C` بشكل أساسي إلى
نموذجنا أننا لا نثق كثيراً في توزيع البيانات، ولن نأخذ في الاعتبار سوى النقاط القريبة من خط
الفصل.

تتضمن قيمة صغيرة من `C` المزيد/جميع الملاحظات، مما يسمح
بحساب الهوامش باستخدام جميع البيانات في المنطقة.

"""

# المؤلفون: مطوري سكايلرن
# معرف الترخيص: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm

# نقوم بإنشاء 40 نقطة قابلة للفصل
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

# رقم الشكل
fignum = 1

# ملاءمة النموذج
for name, penalty in (("unreg", 1), ("reg", 0.05)):
    clf = svm.SVC(kernel="linear", C=penalty)
    clf.fit(X, Y)
    # الحصول على الفاصل الفائق
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # رسم المتوازيات للفاصل الفائق التي تمر عبر
    # المتجهات الداعمة (الهامش بعيدًا عن الفاصل الفائق في الاتجاه
    # عمودي على الفاصل الفائق). هذا بعيدًا عموديًا في
    # 2-d.
    margin = 1 / np.sqrt(np.sum(clf.coef_**2))
    yy_down = yy - np.sqrt(1 + a**2) * margin
    yy_up = yy + np.sqrt(1 + a**2) * margin

    # رسم الخط، والنقاط، وأقرب المتجهات إلى المستوى
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    plt.plot(xx, yy, "k-")
    plt.plot(xx, yy_down, "k--")
    plt.plot(xx, yy_up, "k--")

    plt.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=80,
        facecolors="none",
        zorder=10,
        edgecolors="k",
    )
    plt.scatter(
        X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.get_cmap("RdBu"), edgecolors="k"
    )

    plt.axis("tight")
    x_min = -4.8
    x_max = 4.2
    y_min = -6
    y_max = 6

    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # وضع النتيجة في مخطط المُحيط
    plt.contourf(XX, YY, Z, cmap=plt.get_cmap("RdBu"), alpha=0.5, linestyles=["-"])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1

    plt.show()