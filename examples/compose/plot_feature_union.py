"""
=================================================
دمج طرق استخراج ميزات متعددة
=================================================

في العديد من الأمثلة الواقعية، هناك العديد من الطرق لاستخراج الميزات من مجموعة بيانات. غالبًا ما يكون من المفيد الجمع بين عدة طرق للحصول على أداء جيد. يوضح هذا المثال كيفية استخدام ``FeatureUnion`` لدمج الميزات التي تم الحصول عليها بواسطة PCA والاختيار أحادي المتغير.

يُتيح دمج الميزات باستخدام هذا المحول ميزة أنه يسمح بالتحقق المتبادل والبحث الشبكي خلال العملية بأكملها.

إن التركيبة المستخدمة في هذا المثال ليست مفيدة بشكل خاص في مجموعة البيانات هذه ولا تُستخدم إلا لتوضيح استخدام FeatureUnion.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC

iris = load_iris()

X, y = iris.data, iris.target

# مجموعة البيانات هذه عالية الأبعاد للغاية. من الأفضل القيام بـ PCA:
pca = PCA(n_components=2)

# ربما كانت بعض الميزات الأصلية جيدة أيضًا؟
selection = SelectKBest(k=1)

# بناء مقدر من PCA والاختيار أحادي المتغير:

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# استخدام الميزات المدمجة لتحويل مجموعة البيانات:
X_features = combined_features.fit(X, y).transform(X)
print("Combined space has", X_features.shape[1], "features")

svm = SVC(kernel="linear")

# إجراء بحث شبكي على k و n_components و C:

pipeline = Pipeline([("features", combined_features), ("svm", svm)])

param_grid = dict(
    features__pca__n_components=[1, 2, 3],
    features__univ_select__k=[1, 2],
    svm__C=[0.1, 1, 10],
)

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(X, y)
print(grid_search.best_estimator_)
