.. currentmodule:: sklearn.preprocessing

.. _preprocessing_targets:

==========================================
تحويل هدف التنبؤ (``y``)
==========================================

هذه هي المحولات التي لا يقصد استخدامها على الميزات، فقط على
أهداف التعلم المشرف. راجع أيضًا :ref:`transformed_target_regressor` إذا
كنت ترغب في تحويل هدف التنبؤ للتعلم، ولكن تقييم
النموذج في المساحة الأصلية (غير المحولة).

تثنية التصنيف
==================

LabelBinarizer
--------------

:class:`LabelBinarizer` هي فئة مساعدة للمساعدة في إنشاء مصفوفة مؤشر التصنيف من قائمة التصنيفات متعددة الفئات::

    >>> from sklearn import preprocessing
    >>> lb = preprocessing.LabelBinarizer()
    >>> lb.fit([1, 2, 6, 4, 2])
    LabelBinarizer()
    >>> lb.classes_
    array([1, 2, 4, 6])
    >>> lb.transform([1, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

يسمح استخدام هذا التنسيق بالتصنيف متعدد الفئات في المقدرات
التي تدعم تنسيق مصفوفة مؤشر التصنيف.

.. warning::

    لا يلزم استخدام LabelBinarizer إذا كنت تستخدم مقدرًا
    يدعم بالفعل بيانات متعددة الفئات.

لمزيد من المعلومات حول التصنيف متعدد الفئات، راجع
:ref:`multiclass_classification`.

MultiLabelBinarizer
-------------------

في التعلم متعدد التصنيفات، يتم التعبير عن مجموعة مهام التصنيف الثنائية المشتركة بمؤشر مصفوفة ثنائية التصنيف: كل عينة هي صف واحد من مصفوفة ثنائية الأبعاد ذات الشكل (n_samples، n_classes) بقيم ثنائية حيث الواحد، أي العناصر غير الصفرية، يقابل مجموعة التصنيفات لهذه العينة. مصفوفة مثل ``np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]])`` تمثل التصنيف 0 في العينة الأولى، والتصنيفات 1 و 2 في العينة الثانية، ولا تصنيفات في العينة الثالثة.

قد يكون إنتاج بيانات متعددة التصنيفات على شكل قائمة من مجموعات التصنيفات أكثر بديهية.
يمكن استخدام المحول :class:`MultiLabelBinarizer <sklearn.preprocessing.MultiLabelBinarizer>`
لتحويل بين مجموعة من مجموعات التصنيفات وتنسيق المؤشر::

    >>> from sklearn.preprocessing import MultiLabelBinarizer
    >>> y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]
    >>> MultiLabelBinarizer().fit_transform(y)
    array([[0, 0, 1, 1, 1],
           [0, 0, 1, 0, 0],
           [1, 1, 0, 1, 0],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 0, 0]])

لمزيد من المعلومات حول التصنيف متعدد التصنيفات، راجع
:ref:`multilabel_classification`.

الترميز التصنيفي
==============

:class:`LabelEncoder` هي فئة مساعدة للمساعدة في تطبيع التصنيفات بحيث
تحتوي فقط على قيم بين 0 و n_classes-1. هذا مفيد أحيانًا
لكتابة روتينات Cython الفعالة. يمكن استخدام :class:`LabelEncoder` على النحو التالي::

    >>> from sklearn import preprocessing
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit([1, 2, 2, 6])
    LabelEncoder()
    >>> le.classes_
    array([1, 2, 6])
    >>> le.transform([1, 1, 2, 6])
    array([0, 0, 1, 2])
    >>> le.inverse_transform([0, 0, 1, 2])
    array([1, 1, 2, 6])

يمكن استخدامه أيضًا لتحويل التصنيفات غير الرقمية (طالما أنها
قابلة للتجزئة والقابلة للمقارنة) إلى تصنيفات رقمية::

    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"])
    array([2, 2, 1])
    >>> list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']