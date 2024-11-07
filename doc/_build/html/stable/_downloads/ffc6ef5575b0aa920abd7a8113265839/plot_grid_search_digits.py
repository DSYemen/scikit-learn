"""
============================================================
استراتيجية إعادة الضبط المخصصة للبحث الشبكي مع التحقق المتقاطع
============================================================

يُظهر هذا المثال كيفية تحسين التصنيف من خلال التحقق المتقاطع،
والذي يتم باستخدام كائن :class:`~sklearn.model_selection.GridSearchCV`
على مجموعة تطوير تتكون من نصف بيانات التصنيف المتاحة فقط.

يتم بعد ذلك قياس أداء المعلمات فائقة التحديد والنموذج المدرب
على مجموعة تقييم مخصصة لم يتم استخدامها أثناء
خطوة اختيار النموذج.

يمكن العثور على مزيد من التفاصيل حول الأدوات المتاحة لاختيار النموذج في
الأقسام الخاصة بـ :ref:`cross_validation` و :ref:`grid_search`.
"""

# المؤلفون: مطوري scikit-learn
# معرف SPDX-License: BSD-3-Clause

# %%
# مجموعة البيانات
# -----------
#
# سنعمل مع مجموعة بيانات `digits`. الهدف هو تصنيف صور الأرقام المكتوبة بخط اليد.
# نحن نحول المشكلة إلى تصنيف ثنائي من أجل الفهم الأسهل: الهدف هو تحديد ما إذا كان الرقم هو `8` أم لا.
from sklearn import datasets

digits = datasets.load_digits()

# %%
# من أجل تدريب مصنف على الصور، نحتاج إلى تسطيحها إلى متجهات.
# تحتاج كل صورة من 8 بكسل في 8 بكسل إلى تحويلها إلى متجه من 64 بكسل.
# وبالتالي، سنحصل على مصفوفة بيانات نهائية ذات شكل `(n_images, n_pixels)`.
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target == 8
print(
    f"عدد الصور هو {X.shape[0]} وتحتوي كل صورة على {X.shape[1]} بكسل"
)

# %%
# كما هو موضح في المقدمة، سيتم تقسيم البيانات إلى مجموعة تدريب
# ومجموعة اختبار بنفس الحجم.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# %%
# تحديد استراتيجية البحث الشبكي
# -------------------------------
#
# سنختار مصنفًا من خلال البحث عن أفضل المعلمات فائقة التحديد على طيات
# مجموعة التدريب. للقيام بذلك، نحتاج إلى تحديد
# الدرجات لاختيار أفضل مرشح.

scores = ["precision", "recall"]

# %%
# يمكننا أيضًا تحديد دالة لتمريرها إلى معلمة `refit` الخاصة بـ
# :class:`~sklearn.model_selection.GridSearchCV`. ستقوم بتنفيذ
# الاستراتيجية المخصصة لاختيار أفضل مرشح من سمة `cv_results_`
# الخاصة بـ :class:`~sklearn.model_selection.GridSearchCV`. بمجرد اختيار المرشح،
# يتم إعادة ضبطه تلقائيًا بواسطة
# :class:`~sklearn.model_selection.GridSearchCV`.
#
# هنا، الاستراتيجية هي وضع قائمة مختصرة للنماذج التي تكون الأفضل من حيث
# الدقة والاستدعاء. من النماذج المختارة، نختار أخيرًا النموذج الأسرع
# في التنبؤ. لاحظ أن هذه الخيارات المخصصة تعسفية تمامًا.

import pandas as pd


def print_dataframe(filtered_cv_results):
    """طباعة جميلة لمصفوفة البيانات المفلترة"""
    for mean_precision, std_precision, mean_recall, std_recall, params in zip(
        filtered_cv_results["mean_test_precision"],
        filtered_cv_results["std_test_precision"],
        filtered_cv_results["mean_test_recall"],
        filtered_cv_results["std_test_recall"],
        filtered_cv_results["params"],
    ):
        print(
            f"الدقة: {mean_precision:0.3f} (±{std_precision:0.03f}),"
            f" الاستدعاء: {mean_recall:0.3f} (±{std_recall:0.03f}),"
            f" للـ {params}"
        )
    print()


def refit_strategy(cv_results):
    """تحديد الاستراتيجية لاختيار أفضل مقدر.

    الاستراتيجية المحددة هنا هي استبعاد جميع النتائج التي تقل عن عتبة دقة
    تبلغ 0.98، وترتيب النتائج المتبقية حسب الاستدعاء والاحتفاظ بجميع النماذج
    مع انحراف معياري واحد من الأفضل من حيث الاستدعاء. بمجرد اختيار هذه النماذج،
    يمكننا اختيار النموذج الأسرع في التنبؤ.

    المعلمات
    ----------
    cv_results : dict of numpy (masked) ndarrays
        نتائج CV كما أعادتها `GridSearchCV`.

    الإرجاع
    -------
    best_index : int
        فهرس أفضل مقدر كما يظهر في `cv_results`.
    """
    # طباعة المعلومات حول البحث الشبكي للدرجات المختلفة
    precision_threshold = 0.98

    cv_results_ = pd.DataFrame(cv_results)
    print("جميع نتائج البحث الشبكي:")
    print_dataframe(cv_results_)

    # استبعاد جميع النتائج التي تقل عن العتبة
    high_precision_cv_results = cv_results_[
        cv_results_["mean_test_precision"] > precision_threshold
    ]

    print(f"النماذج ذات الدقة الأعلى من {precision_threshold}:")
    print_dataframe(high_precision_cv_results)

    high_precision_cv_results = high_precision_cv_results[
        [
            "mean_score_time",
            "mean_test_recall",
            "std_test_recall",
            "mean_test_precision",
            "std_test_precision",
            "rank_test_recall",
            "rank_test_precision",
            "params",
        ]
    ]

    # اختيار النماذج الأكثر أداءً من حيث الاستدعاء
    # (ضمن انحراف معياري واحد من الأفضل)
    best_recall_std = high_precision_cv_results["mean_test_recall"].std()
    best_recall = high_precision_cv_results["mean_test_recall"].max()
    best_recall_threshold = best_recall - best_recall_std

    high_recall_cv_results = high_precision_cv_results[
        high_precision_cv_results["mean_test_recall"] > best_recall_threshold
    ]
    print(
        "من النماذج المختارة ذات الدقة العالية، نحتفظ بجميع\n"
        "النماذج ضمن انحراف معياري واحد من النموذج الأعلى استدعاءً:"
    )
    print_dataframe(high_recall_cv_results)

    # من بين أفضل المرشحين، اختيار النموذج الأسرع في التنبؤ
    fastest_top_recall_high_precision_index = high_recall_cv_results[
        "mean_score_time"
    ].idxmin()

    print(
        "\nالنموذج المختار النهائي هو الأسرع في التنبؤ من بين\n"
        "المجموعة الفرعية المختارة مسبقًا من أفضل النماذج بناءً على الدقة والاستدعاء.\n"
        "وقت تسجيله هو:\n\n"
        f"{high_recall_cv_results.loc[fastest_top_recall_high_precision_index]}"
    )

    return fastest_top_recall_high_precision_index


# %%
#
# ضبط المعلمات فائقة التحديد
# -----------------------
#
# بمجرد تحديد استراتيجيتنا لاختيار أفضل نموذج، نقوم بتحديد قيم
# المعلمات فائقة التحديد وإنشاء مثيل البحث الشبكي:
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

tuned_parameters = [
    {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
    {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
]

grid_search = GridSearchCV(
    SVC(), tuned_parameters, scoring=scores, refit=refit_strategy
)
grid_search.fit(X_train, y_train)

# %%
#
# المعلمات التي اختارها البحث الشبكي باستراتيجيتنا المخصصة هي:
grid_search.best_params_

# %%
#
# أخيرًا، نقوم بتقييم النموذج المضبوط بدقة على مجموعة التقييم المتبقية:
# تم إعادة ضبط كائن `grid_search` **تلقائيًا** على مجموعة التدريب
# الكاملة بالمعلمات التي اختارتها استراتيجية إعادة الضبط المخصصة لدينا.
#
# يمكننا استخدام تقرير التصنيف لحساب مقاييس التصنيف القياسية على المجموعة المتبقية:
from sklearn.metrics import classification_report

y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
# .. note::
#    المشكلة سهلة للغاية: هضبة المعلمات فائقة التحديد مسطحة للغاية والنموذج
#    الناتج هو نفسه بالنسبة للدقة والاستدعاء مع تعادلات في الجودة.
