
.. _visualizations:

============
التصورات
============

يُعرّف Scikit-learn واجهة برمجة تطبيقات بسيطة لإنشاء تصورات للتعلم الآلي.
الميزة الرئيسية لواجهة برمجة التطبيقات هذه هي السماح بالتخطيط السريع والتعديلات المرئية دون إعادة الحساب.
نحن نقدم فئات `Display` التي تعرض طريقتين لإنشاء الرسوم البيانية: `from_estimator` و `from_predictions`.
ستأخذ طريقة `from_estimator` مقدرًا مناسبًا وبعض البيانات (`X` و `y`) وتنشئ كائن `Display`.
في بعض الأحيان، نرغب في حساب التنبؤات مرة واحدة فقط ويجب على المرء استخدام `from_predictions` بدلاً من ذلك.
في المثال التالي، نرسم منحنى ROC لجهاز متجه دعم مناسب:

.. plot::
   :context: close-figs
   :align: center

    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import RocCurveDisplay
    from sklearn.datasets import load_wine

    X, y = load_wine(return_X_y=True)
    y = y == 2  # جعل ثنائي
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    svc = SVC(random_state=42)
    svc.fit(X_train, y_train)

    svc_disp = RocCurveDisplay.from_estimator(svc, X_test, y_test)

يسمح لنا الكائن `svc_disp` الذي تم إرجاعه بمواصلة استخدام منحنى ROC المحسوب بالفعل لـ SVC في الرسوم البيانية المستقبلية.
في هذه الحالة، يكون `svc_disp` هو :class:`~sklearn.metrics.RocCurveDisplay` الذي يخزن القيم المحسوبة كسمات تسمى `roc_auc` و `fpr` و `tpr`.
انتبه إلى أنه يمكننا الحصول على التنبؤات من جهاز متجه الدعم ثم استخدام `from_predictions` بدلاً من `from_estimator`.
بعد ذلك، نقوم بتدريب مصنف غابة عشوائي ورسم منحنى roc المحسوب مسبقًا مرة أخرى باستخدام طريقة `plot` لكائن `Display`.

.. plot::
   :context: close-figs
   :align: center

    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier

    rfc = RandomForestClassifier(n_estimators=10, random_state=42)
    rfc.fit(X_train, y_train)

    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=ax, alpha=0.8)
    svc_disp.plot(ax=ax, alpha=0.8)

لاحظ أننا نمرر `alpha=0.8` إلى وظائف التخطيط لضبط قيم ألفا للمنحنيات.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_miscellaneous_plot_roc_curve_visualization_api.py`
* :ref:`sphx_glr_auto_examples_miscellaneous_plot_partial_dependence_visualization_api.py`
* :ref:`sphx_glr_auto_examples_miscellaneous_plot_display_object_visualization.py`
* :ref:`sphx_glr_auto_examples_calibration_plot_compare_calibration.py`

أدوات التخطيط المتاحة
=========================

عرض الكائنات
---------------

.. currentmodule:: sklearn

.. autosummary::

   calibration.CalibrationDisplay
   inspection.PartialDependenceDisplay
   inspection.DecisionBoundaryDisplay
   metrics.ConfusionMatrixDisplay
   metrics.DetCurveDisplay
   metrics.PrecisionRecallDisplay
   metrics.PredictionErrorDisplay
   metrics.RocCurveDisplay
   model_selection.LearningCurveDisplay
   model_selection.ValidationCurveDisplay


