
.. _permutation_importance:

أهمية التبديل
===============

.. currentmodule:: sklearn.inspection

أهمية التبديل هي تقنية فحص النموذج التي تقيس مساهمة كل ميزة في الأداء الإحصائي للنموذج :term:`fitted` على مجموعة بيانات جدولي معينة. هذه التقنية مفيدة بشكل خاص للمقدّرات غير الخطية أو غير الشفافة، وتتضمن خلط قيم ميزة واحدة بشكل عشوائي ومراقبة التدهور الناتج في درجة النموذج [1]_. من خلال كسر العلاقة بين الميزة والهدف، نحدد مدى اعتماد النموذج على هذه الميزة المحددة.

في الأشكال التالية، نلاحظ تأثير تبديل الميزات على الارتباط
بين الميزة والهدف وبالتالي على الأداء الإحصائي للنموذج.

.. image:: ../images/permuted_predictive_feature.png
   :align: center

.. image:: ../images/permuted_non_predictive_feature.png
   :align: center

في الشكل العلوي، نلاحظ أن تبديل ميزة تنبؤية يكسر
الارتباط بين الميزة والهدف، وبالتالي ينخفض الأداء الإحصائي للنموذج. في الشكل السفلي، نلاحظ أن تبديل
ميزة غير تنبؤية لا يتسبب في تدهور كبير في الأداء الإحصائي للنموذج.

تتمثل إحدى المزايا الرئيسية لأهمية التبديل في أنها
لا تعتمد على النموذج، أي يمكن تطبيقها على أي نموذج مدرب. علاوة على ذلك، يمكن
حسابها عدة مرات مع تبديلات مختلفة للميزة، مما يوفر المزيد
من قياس التباين في أهمية الميزات المقدرة للنموذج المدرب المحدد.

يوضح الشكل أدناه أهمية التبديل ل
:class:`~sklearn.ensemble.RandomForestClassifier` المدرب على نسخة موسعة
من مجموعة بيانات التيتانيك تحتوي على `random_cat` و `random_num`
ميزات، أي ميزة تصنيف وميزة رقمية غير مرتبطة بأي شكل من الأشكال مع المتغير المستهدف:

.. figure:: ../auto_examples/inspection/images/sphx_glr_plot_permutation_importance_002.png
   :target: ../auto_examples/inspection/plot_permutation_importance.html
   :align: center
   :scale: 70

.. warning::

  الميزات التي تعتبر ذات **أهمية منخفضة لنموذج سيء** (درجة التحقق المتقاطع منخفضة) قد تكون **مهمة للغاية لنموذج جيد**.
  لذلك من المهم دائمًا تقييم القوة التنبؤية للنموذج
  باستخدام مجموعة محجوزة (أو أفضل مع التحقق المتقاطع) قبل حساب
  الأهميات. لا تعكس أهمية التبديل القيمة التنبؤية الجوهرية لميزة بحد ذاتها ولكن **مدى أهمية هذه الميزة
  لنموذج معين**.

تقوم دالة :func:`permutation_importance` بحساب أهمية الميزة
من :term:`estimators` لمجموعة بيانات معينة. يحدد معلمة ``n_repeats``
عدد المرات التي يتم فيها خلط ميزة عشوائيًا وإرجاع عينة من أهمية الميزة.

لنأخذ في الاعتبار نموذج الانحدار المدرب التالي::

  >>> from sklearn.datasets import load_diabetes
  >>> from sklearn.model_selection import train_test_split
  >>> from sklearn.linear_model import Ridge
  >>> diabetes = load_diabetes()
  >>> X_train, X_val, y_train, y_val = train_test_split(
  ...     diabetes.data, diabetes.target, random_state=0)
  ...
  >>> model = Ridge(alpha=1e-2).fit(X_train, y_train)
  >>> model.score(X_val, y_val)
  0.356...

أداؤه التحقق، المقاس عبر درجة :math:`R^2`،
أكبر بكثير من مستوى الصدفة. هذا يجعل من الممكن استخدام
:func:`permutation_importance` وظيفة للتحقق من الميزات الأكثر تنبؤية::

  >>> from sklearn.inspection import permutation_importance
  >>> r = permutation_importance(model, X_val, y_val,
  ...                            n_repeats=30,
  ...                            random_state=0)
  ...
  >>> for i in r.importances_mean.argsort()[::-1]:
  ...     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
  ...         print(f"{diabetes.feature_names[i]:<8}"
  ...               f"{r.importances_mean[i]:.3f}"
  ...               f" +/- {r.importances_std[i]:.3f}")
  ...
  s5      0.204 +/- 0.050
  bmi     0.176 +/- 0.048
  bp      0.088 +/- 0.033
  sex     0.056 +/- 0.023

لاحظ أن قيم الأهمية للميزات الأعلى تمثل جزءًا كبيرًا من درجة المرجع 0.356.

يمكن حساب أهمية التبديل إما على مجموعة التدريب أو على
مجموعة اختبار أو تحقق محجوزة. باستخدام مجموعة محجوزة يجعل من الممكن
تسليط الضوء على الميزات التي تساهم أكثر في قوة التعميم للنموذج
المفحوص. الميزات المهمة في مجموعة التدريب ولكن ليس على
المجموعة المحجوزة قد تتسبب في الإفراط في تناسب النموذج.

تعتمد أهمية التبديل على دالة الدرجة التي يتم تحديدها
مع حجة `scoring`. تقبل هذه الحجة العديد من الدلائل،
والتي تكون أكثر كفاءة من الناحية الحسابية من الاستدعاء التسلسلي
:func:`permutation_importance` عدة مرات مع درجة مختلفة، حيث يعيد استخدام تنبؤات النموذج.

.. dropdown:: مثال على أهمية التبديل باستخدام العديد من الدلائل

  في المثال أدناه، نستخدم قائمة من المقاييس، ولكن هناك تنسيقات إدخال أكثر،
  كما هو موثق في :ref:`multimetric_scoring`.

    >>> scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
    >>> r_multi = permutation_importance(
    ...     model, X_val, y_val, n_repeats=30, random_state=0, scoring=scoring)
    ...
    >>> for metric in r_multi:
    ...     print(f"{metric}")
    ...     r = r_multi[metric]
    ...     for i in r.importances_mean.argsort()[::-1]:
    ...         if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
    ...             print(f"    {diabetes.feature_names[i]:<8}"
    ...                   f"{r.importances_mean[i]:.3f}"
    ...                   f" +/- {r.importances_std[i]:.3f}")
    ...
    r2
        s5      0.204 +/- 0.050
        bmi     0.176 +/- 0.048
        bp      0.088 +/- 0.033
        sex     0.056 +/- 0.023
    neg_mean_absolute_percentage_error
        s5      0.081 +/- 0.020
        bmi     0.064 +/- 0.015
        bp      0.029 +/- 0.010
    neg_mean_squared_error
        s5      1013.866 +/- 246.445
        bmi     872.726 +/- 240.298
        bp      438.663 +/- 163.022
        sex     277.376 +/- 115.123

  ترتيب الميزات هو نفسه تقريبًا لمقاييس مختلفة حتى إذا كانت مقاييس الأهمية مختلفة جدًا. ومع ذلك، هذا غير
  مضمون وقد تؤدي المقاييس المختلفة إلى أهمية ميزة مختلفة بشكل كبير، خاصة بالنسبة للنماذج المدربة على مشاكل التصنيف غير المتوازنة،
  حيث **قد يكون اختيار مقياس التصنيف حاسمًا**.

مخطط خوارزمية أهمية التبديل
-----------------------------------------------

- المدخلات: نموذج تنبؤي مدرب :math:`m`، مجموعة بيانات جدولي (تدريب أو
  التحقق) :math:`D`.
- احسب درجة المرجع :math:`s` للنموذج :math:`m` على البيانات
  :math:`D` (على سبيل المثال الدقة لمصنّف أو :math:`R^2` لمصنّف).
- لكل ميزة :math:`j` (عمود من :math:`D`):

  - لكل تكرار :math:`k` في :math:`{1, ..., K}`:

    - قم بخلط عمود :math:`j` من مجموعة البيانات :math:`D` بشكل عشوائي لإنشاء
      نسخة فاسدة من البيانات باسم :math:`\tilde{D}_{k,j}`.
    - احسب درجة :math:`s_{k,j}` للنموذج :math:`m` على البيانات الفاسدة
      :math:`\tilde{D}_{k,j}`.

  - احسب الأهمية :math:`i_j` للميزة :math:`f_j` المحددة على النحو التالي:

    .. math:: i_j = s - \frac{1}{K} \sum_{k=1}^{K} s_{k,j}

العلاقة بأهمية الشجرة القائمة على الشوائب
----------------------------------------------

توفر النماذج القائمة على الشجرة مقياسًا بديلاً لـ :ref:`feature importances
based on the mean decrease in impurity <random_forest_feature_importance>`
(MDI). يتم تحديد الشوائب بواسطة معيار التقسيم لشجرة القرار (Gini أو Log Loss أو Mean Squared Error). ومع ذلك، يمكن لهذه الطريقة أن تعطي أهمية عالية للميزات التي قد لا تكون تنبؤية على البيانات غير المرئية عندما يكون النموذج مفرطًا في الملاءمة. من ناحية أخرى، تتجنب أهمية الميزة القائمة على التبديل هذه المشكلة، حيث يمكن حسابها على البيانات غير المرئية.

علاوة على ذلك، فإن أهمية الميزة القائمة على الشوائب للشجرة **متحيزة بشدة** و **تفضل الميزات ذات التعداد المرتفع** (عادة الميزات الرقمية)
على الميزات ذات التعداد المنخفض مثل الميزات الثنائية أو المتغيرات التصنيفية
مع عدد صغير من الفئات المحتملة.

لا تظهر أهمية الميزات القائمة على التبديل مثل هذا التحيز. بالإضافة إلى ذلك،
يمكن حساب أهمية التبديل بأي مقياس للأداء على تنبؤات النموذج ويمكن استخدامها لتحليل أي فئة من النماذج (ليس فقط النماذج القائمة على الشجرة).

يسلط المثال التالي الضوء على قيود أهمية الميزة القائمة على الشوائب على عكس أهمية الميزة القائمة على التبديل:
:ref:`sphx_glr_auto_examples_inspection_plot_permutation_importance.py`.

القيم المضللة على الميزات ذات الارتباط القوي
-------------------------------------------------

عندما تكون الميزتان مترابطتين ويتم تبديل إحدى الميزات، فإن النموذج
لا يزال لديه حق الوصول إلى الأخيرة من خلال ميزته المرتبطة. يؤدي هذا إلى قيمة أهمية أقل للإبلاغ عن كلتا الميزتين، على الرغم من أنهما قد يكونان *في الواقع*
مهمة.

يوضح الشكل أدناه أهمية التبديل ل
:class:`~sklearn.ensemble.RandomForestClassifier` المدرب باستخدام
:ref:`breast_cancer_dataset`، الذي يحتوي على ميزات مترابطة بقوة. سيوحي التفسير الساذج بأن جميع الميزات غير مهمة:

.. figure:: ../auto_examples/inspection/images/sphx_glr_plot_permutation_importance_multicollinear_002.png
   :target: ../auto_examples/inspection/plot_permutation_importance_multicollinear.html
   :align: center
   :scale: 70

تتمثل إحدى طرق التعامل مع المشكلة في تجميع الميزات المترابطة والاحتفاظ بميزة واحدة فقط من كل مجموعة.

.. figure:: ../auto_examples/inspection/images/sphx_glr_plot_permutation_importance_multicollinear_004.png
   :target: ../auto_examples/inspection/plot_permutation_importance_multicollinear.html
   :align: center
   :scale: 70

للحصول على مزيد من التفاصيل حول هذه الاستراتيجية، راجع المثال
:ref:`sphx_glr_auto_examples_inspection_plot_permutation_importance_multicollinear.py`.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_inspection_plot_permutation_importance.py`
* :ref:`sphx_glr_auto_examples_inspection_plot_permutation_importance_multicollinear.py`

.. rubric:: المراجع

.. [1] L. Breiman, :doi:`"Random Forests" <10.1023/A:1010933404324>`,
  Machine Learning, 45(1), 5-32, 2001.