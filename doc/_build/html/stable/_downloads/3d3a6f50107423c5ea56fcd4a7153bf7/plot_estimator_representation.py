"""
===========================================
عرض المنمذجات وأنابيب التوصيل المعقدة
===========================================

يوضح هذا المثال طرقًا مختلفة لعرض المنمذجات وأنابيب التوصيل.
"""

# المؤلفون: مطوري scikit-learn
# معرف الترخيص: BSD-3-Clause

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# %%
# تمثيل نصي مضغوط
# ---------------------------
#
# ستعرض المنمذجات فقط المعاملات التي تم ضبطها على قيم غير افتراضية
# عندما يتم عرضها كسلسلة نصية. هذا يقلل من التشويش البصري ويجعل من السهل
# ملاحظة الاختلافات عند مقارنة الحالات.

lr = LogisticRegression(penalty="l1")
print(lr)

# %%
# تمثيل HTML غني
# ------------------------
# في دفاتر الملاحظات، ستستخدم المنمذجات وأنابيب التوصيل تمثيل HTML غني.
# وهذا مفيد بشكل خاص لتلخيص
# بنية أنابيب التوصيل والمنمذجات المركبة الأخرى، مع التفاعل لتوفير التفاصيل.  انقر على الصورة التوضيحية أدناه لتوسيع عناصر أنبوب التوصيل.  راجع: ref:`visualizing_composite_estimators` لمعرفة كيفية استخدام
# هذه الميزة.

num_proc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

cat_proc = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OneHotEncoder(handle_unknown="ignore"),
)

preprocessor = make_column_transformer(
    (num_proc, ("feat1", "feat3")), (cat_proc, ("feat0", "feat2"))
)

clf = make_pipeline(preprocessor, LogisticRegression())
clf