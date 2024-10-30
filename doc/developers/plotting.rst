.. _plotting_api:

======================================================
التطوير باستخدام واجهة برمجة تطبيقات تصور البيانات
======================================================

يُحدد Scikit-learn واجهة برمجة تطبيقات بسيطة لإنشاء تصورات لـ
التعلم الآلي. الميزات الرئيسية لهذه الواجهة البرمجية هي تشغيل العمليات الحسابية مرة واحدة والحصول على
مرونة ضبط التصورات بعد وقوع الحدث. هذا القسم مخصص للمطورين الذين يرغبون في تطوير أو صيانة أدوات تصور البيانات.
بالنسبة للاستخدام، يجب على المستخدمين الرجوع إلى :ref:`دليل المستخدم <visualizations>`.

نظرة عامة على واجهة برمجة تطبيقات تصور البيانات
----------------------------------------------------

يتم تغليف هذا المنطق في كائن عرض حيث يتم تخزين البيانات المحسوبة ويتم
تصور البيانات في أسلوب `plot`. يحتوي أسلوب `__init__` لكائن العرض
على البيانات اللازمة لإنشاء التصور فقط.
يأخذ أسلوب `plot` معلمات تتعلق فقط بالتصور،
مثل محاور matplotlib. سيخزن أسلوب `plot` فناني matplotlib
كسمات تسمح بضبط النمط من خلال كائن العرض.
يجب أن تُحدد فئة `Display` أسلوب فئة واحد أو كليهما: `from_estimator` و
`from_predictions`. تسمح هذه الأساليب بإنشاء كائن `Display` من
المقدر وبعض البيانات أو من القيم الحقيقية والمتوقعة. بعد هذه
أساليب الفئة التي تُنشئ كائن العرض بالقيم المحسوبة، ثم استدعاء
أسلوب plot للعرض. لاحظ أن أسلوب `plot` يُحدد السمات المتعلقة
بـ matplotlib، مثل فنان الخط. يسمح هذا بالتخصيصات بعد
استدعاء أسلوب `plot`.

على سبيل المثال، يُحدد `RocCurveDisplay` الأساليب والسمات
التالية::

   class RocCurveDisplay:
       def __init__(self, fpr, tpr, roc_auc, estimator_name):
           ...
           self.fpr = fpr
           self.tpr = tpr
           self.roc_auc = roc_auc
           self.estimator_name = estimator_name

       @classmethod
       def from_estimator(cls, estimator, X, y):
           # الحصول على التوقعات
           y_pred = estimator.predict_proba(X)[:, 1]
           return cls.from_predictions(y, y_pred, estimator.__class__.__name__)

       @classmethod
       def from_predictions(cls, y, y_pred, estimator_name):
           # إجراء حساب ROC من y و y_pred
           fpr, tpr, roc_auc = ...
           viz = RocCurveDisplay(fpr, tpr, roc_auc, estimator_name)
           return viz.plot()

       def plot(self, ax=None, name=None, **kwargs):
           ...
           self.line_ = ...
           self.ax_ = ax
           self.figure_ = ax.figure_

اقرأ المزيد في :ref:`sphx_glr_auto_examples_miscellaneous_plot_roc_curve_visualization_api.py`
و :ref:`دليل المستخدم <visualizations>`.

تصور البيانات باستخدام محاور متعددة
---------------------------------------

تدعم بعض أدوات تصور البيانات مثل
:func:`~sklearn.inspection.PartialDependenceDisplay.from_estimator` و
:class:`~sklearn.inspection.PartialDependenceDisplay` تصور البيانات على
محاور متعددة. يتم دعم سيناريوهين مختلفين:

1. إذا تم تمرير قائمة محاور، فسيُحدد `plot` ما إذا كان عدد المحاور
   متوافقًا مع عدد المحاور الذي يتوقعه ثم يرسم على تلك المحاور. 2.
   إذا تم تمرير محور واحد، فإن هذا المحور يُحدد مساحة لوضع محاور متعددة
   فيها. في هذه الحالة، نقترح استخدام
   `~matplotlib.gridspec.GridSpecFromSubplotSpec` الخاص بـ matplotlib لتقسيم المساحة::

   import matplotlib.pyplot as plt
   from matplotlib.gridspec import GridSpecFromSubplotSpec

   fig, ax = plt.subplots()
   gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=ax.get_subplotspec())

   ax_top_left = fig.add_subplot(gs[0, 0])
   ax_top_right = fig.add_subplot(gs[0, 1])
   ax_bottom = fig.add_subplot(gs[1, :])


افتراضيًا، تكون الكلمة الرئيسية `ax` في `plot` هي `None`. في هذه الحالة، يتم إنشاء
المحور الفردي ويتم استخدام واجهة برمجة تطبيقات gridspec لإنشاء المناطق المراد تصور البيانات فيها.

انظر، على سبيل المثال، :meth:`~sklearn.inspection.PartialDependenceDisplay.from_estimator`
الذي يرسم خطوطًا وخطوط كفاف متعددة باستخدام واجهة برمجة التطبيقات هذه. يتم حفظ المحور الذي يُحدد
المربع المحيط في سمة `bounding_ax_`. يتم تخزين المحاور الفردية
التي تم إنشاؤها في ndarray `axes_`، المطابق لموضع المحاور على
الشبكة. يتم تعيين المواضع التي لا يتم استخدامها إلى `None`. علاوة على ذلك،
يتم تخزين فناني matplotlib في `lines_` و `contours_` حيث يكون المفتاح هو
الموضع على الشبكة. عند تمرير قائمة محاور، يكون `axes_` و `lines_`
و `contours_` عبارة عن ndarray أحادي الأبعاد يتوافق مع قائمة المحاور التي تم تمريرها.


