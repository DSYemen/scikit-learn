.. _loading_other_datasets:

تحميل مجموعات بيانات أخرى
============================

.. currentmodule:: sklearn.datasets

.. _sample_images:

صور عينة
---------

يحتوي Scikit-learn أيضًا على بعض صور JPEG العينة المنشورة بموجب ترخيص المشاع الإبداعي من قبل مؤلفيها. يمكن أن تكون هذه الصور مفيدة لاختبار الخوارزميات وخطوط الأنابيب على البيانات ثنائية الأبعاد.

.. autosummary::

   load_sample_images
   load_sample_image

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_color_quantization_001.png
   :target: ../auto_examples/cluster/plot_color_quantization.html
   :scale: 30
   :align: right


.. warning::

  يعتمد الترميز الافتراضي للصور على نوع البيانات ``uint8`` لتوفير الذاكرة. غالبًا ما تعمل خوارزميات التعلم الآلي بشكل أفضل إذا تم تحويل المدخلات إلى تمثيل نقطة عائمة أولاً. أيضًا، إذا كنت تخطط لاستخدام ``matplotlib.pyplpt.imshow``، فلا تنسَ تغيير النطاق إلى 0-1 كما هو موضح في المثال التالي.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_cluster_plot_color_quantization.py`

.. _libsvm_loader:

مجموعات البيانات بتنسيق svmlight / libsvm
--------------------------------------------

يتضمن scikit-learn وظائف مساعدة لتحميل مجموعات البيانات بتنسيق svmlight / libsvm. في هذا التنسيق، يأخذ كل سطر الشكل ``<label> <feature-id>:<feature-value> <feature-id>:<feature-value> ...``. هذا التنسيق مناسب بشكل خاص لمجموعات البيانات المتفرقة. في هذه الوحدة، يتم استخدام مصفوفات CSR المتفرقة من scipy لـ ``X`` ومصفوفات numpy لـ ``y``.

يمكنك تحميل مجموعة بيانات كما يلي::

  >>> from sklearn.datasets import load_svmlight_file
  >>> X_train, y_train = load_svmlight_file("/path/to/train_dataset.txt")
  ...                                                         # doctest: +SKIP

يمكنك أيضًا تحميل مجموعتي بيانات (أو أكثر) في وقت واحد::

  >>> X_train, y_train, X_test, y_test = load_svmlight_files(
  ...     ("/path/to/train_dataset.txt", "/path/to/test_dataset.txt"))
  ...                                                         # doctest: +SKIP

في هذه الحالة، من المضمون أن يكون لـ ``X_train`` و ``X_test`` نفس عدد الميزات. هناك طريقة أخرى لتحقيق نفس النتيجة وهي تحديد عدد الميزات::

  >>> X_test, y_test = load_svmlight_file(
  ...     "/path/to/test_dataset.txt", n_features=X_train.shape[1])
  ...                                                         # doctest: +SKIP

.. rubric:: روابط ذات صلة

- `مجموعات البيانات العامة بتنسيق svmlight / libsvm`: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets
- `تنفيذ أسرع متوافق مع API`: https://github.com/mblondel/svmlight-loader

..
    لاختبارات الوثائق:

    >>> import numpy as np
    >>> import os

.. _openml:

تنزيل مجموعات البيانات من مستودع openml.org
--------------------------------------------

`openml.org <https://openml.org>`_ هو مستودع عام لبيانات وتجارب التعلم الآلي، والذي يسمح للجميع بتحميل مجموعات البيانات المفتوحة.

حزمة ``sklearn.datasets`` قادرة على تنزيل مجموعات البيانات من المستودع باستخدام الدالة :func:`sklearn.datasets.fetch_openml`.

على سبيل المثال، لتنزيل مجموعة بيانات عن تعبيرات الجينات في أدمغة الفئران::

  >>> from sklearn.datasets import fetch_openml
  >>> mice = fetch_openml(name='miceprotein', version=4)

لتحديد مجموعة بيانات بالكامل، تحتاج إلى توفير اسم وإصدار، على الرغم من أن الإصدار اختياري، انظر :ref:`openml_versions` أدناه. تحتوي مجموعة البيانات على ما مجموعه 1080 مثالًا ينتمي إلى 8 فئات مختلفة::

  >>> mice.data.shape
  (1080, 77)
  >>> mice.target.shape
  (1080,)
  >>> np.unique(mice.target)
  array(['c-CS-m', 'c-CS-s', 'c-SC-m', 'c-SC-s', 't-CS-m', 't-CS-s', 't-SC-m', 't-SC-s'], dtype=object)

يمكنك الحصول على مزيد من المعلومات حول مجموعة البيانات من خلال النظر إلى السمتين ``DESCR`` و ``details``::

  >>> print(mice.DESCR) # doctest: +SKIP
  **المؤلف**: Clara Higuera, Katheleen J. Gardiner, Krzysztof J. Cios
  **المصدر**: [UCI](https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression) - 2015
  **الرجاء الاستشهاد**: Higuera C, Gardiner KJ, Cios KJ (2015) Self-Organizing
  Feature Maps Identify Proteins Critical to Learning in a Mouse Model of Down
  Syndrome. PLoS ONE 10(6): e0129126...

  >>> mice.details # doctest: +SKIP
  {'id': '40966', 'name': 'MiceProtein', 'version': '4', 'format': 'ARFF',
  'upload_date': '2017-11-08T16:00:15', 'licence': 'Public',
  'url': 'https://www.openml.org/data/v1/download/17928620/MiceProtein.arff',
  'file_id': '17928620', 'default_target_attribute': 'class',
  'row_id_attribute': 'MouseID',
  'ignore_attribute': ['Genotype', 'Treatment', 'Behavior'],
  'tag': ['OpenML-CC18', 'study_135', 'study_98', 'study_99'],
  'visibility': 'public', 'status': 'active',
  'md5_checksum': '3c479a6885bfa0438971388283a1ce32'}


يحتوي ``DESCR`` على وصف نصي حر للبيانات، بينما يحتوي ``details`` على قاموس بيانات وصفية مخزنة بواسطة openml، مثل معرّف مجموعة البيانات. لمزيد من التفاصيل، انظر `وثائق OpenML <https://docs.openml.org/#data>`_. معرّف البيانات لمجموعة بيانات بروتين الفئران هو 40966، ويمكنك استخدام هذا (أو الاسم) للحصول على مزيد من المعلومات حول مجموعة البيانات على موقع openml::

  >>> mice.url
  'https://www.openml.org/d/40966'

يحدد ``data_id`` أيضًا بشكل فريد مجموعة بيانات من OpenML::

  >>> mice = fetch_openml(data_id=40966)
  >>> mice.details # doctest: +SKIP
  {'id': '4550', 'name': 'MiceProtein', 'version': '1', 'format': 'ARFF',
  'creator': ...,
  'upload_date': '2016-02-17T14:32:49', 'licence': 'Public', 'url':
  'https://www.openml.org/data/v1/download/1804243/MiceProtein.ARFF', 'file_id':
  '1804243', 'default_target_attribute': 'class', 'citation': 'Higuera C,
  Gardiner KJ, Cios KJ (2015) Self-Organizing Feature Maps Identify Proteins
  Critical to Learning in a Mouse Model of Down Syndrome. PLoS ONE 10(6):
  e0129126. [Web Link] journal.pone.0129126', 'tag': ['OpenML100', 'study_14',
  'study_34'], 'visibility': 'public', 'status': 'active', 'md5_checksum':
  '3c479a6885bfa0438971388283a1ce32'}

.. _openml_versions:

إصدارات مجموعة البيانات
~~~~~~~~~~~~~~~~~~~~~~

يتم تحديد مجموعة البيانات بشكل فريد بواسطة ``data_id``، ولكن ليس بالضرورة باسمها. يمكن أن توجد عدة "إصدارات" مختلفة لمجموعة بيانات بنفس الاسم والتي يمكن أن تحتوي على مجموعات بيانات مختلفة تمامًا.
إذا تم العثور على إصدار معين من مجموعة بيانات يحتوي على مشكلات كبيرة، فقد يتم إلغاء تنشيطه. سيؤدي استخدام اسم لتحديد مجموعة بيانات إلى الحصول على الإصدار الأقدم من مجموعة البيانات الذي لا يزال نشطًا. هذا يعني أن ``fetch_openml(name="miceprotein")`` يمكن أن ينتج عنه نتائج مختلفة في أوقات مختلفة إذا أصبحت الإصدارات السابقة غير نشطة.
يمكنك أن ترى أن مجموعة البيانات ذات ``data_id`` 40966 التي قمنا باستردادها أعلاه هي الإصدار الأول من مجموعة بيانات "miceprotein"::

  >>> mice.details['version']  #doctest: +SKIP
  '1'

في الواقع، تحتوي مجموعة البيانات هذه على إصدار واحد فقط. من ناحية أخرى، تحتوي مجموعة بيانات iris على إصدارات متعددة::

  >>> iris = fetch_openml(name="iris")
  >>> iris.details['version']  #doctest: +SKIP
  '1'
  >>> iris.details['id']  #doctest: +SKIP
  '61'

  >>> iris_61 = fetch_openml(data_id=61)
  >>> iris_61.details['version']
  '1'
  >>> iris_61.details['id']
  '61'

  >>> iris_969 = fetch_openml(data_id=969)
  >>> iris_969.details['version']
  '3'
  >>> iris_969.details['id']
  '969'

يؤدي تحديد مجموعة البيانات باسم "iris" إلى الحصول على الإصدار الأدنى، الإصدار 1، مع ``data_id`` 61. للتأكد من حصولك دائمًا على هذه المجموعة الدقيقة من البيانات، من الأكثر أمانًا تحديدها بواسطة ``data_id`` مجموعة البيانات. مجموعة البيانات الأخرى، مع ``data_id`` 969، هي الإصدار 3 (الإصدار 2 أصبح غير نشط)، وتحتوي على إصدار ثنائي من البيانات::

  >>> np.unique(iris_969.target)
  array(['N', 'P'], dtype=object)

يمكنك أيضًا تحديد كل من الاسم والإصدار، اللذين يحددان أيضًا بشكل فريد مجموعة البيانات::

  >>> iris_version_3 = fetch_openml(name="iris", version=3)
  >>> iris_version_3.details['version']
  '3'
  >>> iris_version_3.details['id']
  '969'


.. rubric:: المراجع

* :arxiv:`Vanschoren, van Rijn, Bischl and Torgo. "OpenML: networked science in
  machine learning" ACM SIGKDD Explorations Newsletter, 15(2), 49-60, 2014.
  <1407.7722>`

.. _openml_parser:

محلل ARFF
~~~~~~~~~

من الإصدار 1.2، يوفر scikit-learn وسيطة كلمة رئيسية جديدة `parser` التي توفر العديد من الخيارات لتحليل ملفات ARFF التي يوفرها OpenML. يعتمد المحلل القديم (أي `parser="liac-arff"`) على مشروع `LIAC-ARFF <https://github.com/renatopp/liac-arff>`_. ومع ذلك، فإن هذا المحلل بطيء ويستهلك ذاكرة أكثر من المطلوب. المحلل الجديد الذي يعتمد على pandas (أي `parser="pandas"`) أسرع وأكثر كفاءة في استخدام الذاكرة. ومع ذلك، لا يدعم هذا المحلل البيانات المتفرقة. لذلك، نوصي باستخدام `parser="auto"` الذي سيستخدم أفضل محلل متاح لمجموعة البيانات المطلوبة.

يمكن أن يؤدي المحللان `"pandas"` و `"liac-arff"` إلى أنواع بيانات مختلفة في المخرجات. الاختلافات الملحوظة هي التالية:

- يقوم المحلل `"liac-arff"` دائمًا بترميز الميزات الفئوية ككائنات `str`. على العكس من ذلك، يستنتج المحلل `"pandas"` النوع أثناء القراءة وسيتم تحويل الفئات الرقمية إلى أعداد صحيحة كلما أمكن ذلك.
- يستخدم المحلل `"liac-arff"` float64 لترميز الميزات الرقمية الموسومة باسم 'REAL' و 'NUMERICAL' في البيانات الوصفية. يستنتج المحلل `"pandas"` بدلاً من ذلك ما إذا كانت هذه الميزات الرقمية تتوافق مع الأعداد الصحيحة ويستخدم نوع بيانات امتداد Integer الخاص بـ panda.
- على وجه الخصوص، يتم تحميل مجموعات بيانات التصنيف ذات الفئات الصحيحة عادةً على هذا النحو `(0، 1، ...)` مع المحلل `"pandas"` بينما سيفرض `"liac-arff"` استخدام تسميات الفئات المشتركة كسلاسل مثل `"0"` و `"1"` وما إلى ذلك.
- لن يقوم المحلل `"pandas"` بإزالة علامات الاقتباس المفردة - أي `'` - من أعمدة السلسلة. على سبيل المثال، سيتم الاحتفاظ بسلسلة `'my string'` كما هي بينما سيقوم المحلل `"liac-arff"` بإزالة علامات الاقتباس المفردة. بالنسبة لأعمدة الفئات، تتم إزالة علامات الاقتباس المفردة من القيم.

بالإضافة إلى ذلك، عند استخدام `as_frame = False`، يُرجع المحلل `"liac-arff"` بيانات مشفرة ترتيبيًا حيث يتم توفير الفئات في السمة `categories` لنموذج `Bunch`. بدلاً من ذلك، يُرجع `"pandas"` مصفوفة NumPy حيث الفئات. ثم يعود الأمر للمستخدم لتصميم خط أنابيب لهندسة الميزات مع نموذج من `OneHotEncoder` أو `OrdinalEncoder` ملفوف عادةً في `ColumnTransformer` لمعالجة الأعمدة الفئوية مسبقًا بشكل صريح. انظر على سبيل المثال: :ref:`sphx_glr_auto_examples_compose_plot_column_transformer_mixed_types.py`.

.. _external_datasets:

التحميل من مجموعات بيانات خارجية
---------------------------------

يعمل scikit-learn على أي بيانات رقمية مخزنة كمصفوفات numpy أو مصفوفات scipy المتفرقة. الأنواع الأخرى القابلة للتحويل إلى مصفوفات رقمية مثل pandas DataFrame مقبولة أيضًا.

فيما يلي بعض الطرق الموصى بها لتحميل البيانات العمودية القياسية بتنسيق قابل للاستخدام بواسطة scikit-learn:

* `pandas.io <https://pandas.pydata.org/pandas-docs/stable/io.html>`_
  يوفر أدوات لقراءة البيانات من التنسيقات الشائعة بما في ذلك CSV و Excel و JSON و SQL. يمكن أيضًا إنشاء DataFrames من قوائم tuples أو dicts. يتعامل Pandas مع البيانات غير المتجانسة بسلاسة ويوفر أدوات للمعالجة والتحويل إلى مصفوفة رقمية مناسبة لـ scikit-learn.
* `scipy.io <https://docs.scipy.org/doc/scipy/reference/io.html>`_
  متخصص في التنسيقات الثنائية التي غالبًا ما تستخدم في سياق الحوسبة العلمية مثل .mat و .arff
* `numpy/routines.io <https://docs.scipy.org/doc/numpy/reference/routines.io.html>`_
  للتحميل القياسي للبيانات العمودية في مصفوفات numpy
* :func:`load_svmlight_file` من scikit-learn لتنسيق svmlight أو libSVM المتفرق
* :func:`load_files` من scikit-learn لأدلة ملفات النص حيث يكون اسم كل دليل هو اسم كل فئة ويتوافق كل ملف داخل كل دليل مع عينة واحدة من تلك الفئة

بالنسبة لبعض البيانات المتنوعة مثل الصور ومقاطع الفيديو والصوت، قد ترغب في الرجوع إلى:

* `skimage.io <https://scikit-image.org/docs/dev/api/skimage.io.html>`_ أو
  `Imageio <https://imageio.readthedocs.io/en/stable/reference/core_v3.html>`_
  لتحميل الصور ومقاطع الفيديو في مصفوفات numpy
* `scipy.io.wavfile.read <https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html>`_
 لقراءة ملفات WAV في مصفوفة numpy

ستحتاج الميزات الفئوية (أو الاسمية) المخزنة كسلاسل (شائعة في pandas DataFrames) إلى التحويل إلى ميزات رقمية باستخدام :class:`~sklearn.preprocessing.OneHotEncoder` أو :class:`~sklearn.preprocessing.OrdinalEncoder` أو ما شابه. انظر :ref:`preprocessing`.

ملاحظة: إذا كنت تدير بياناتك الرقمية الخاصة، فمن المستحسن استخدام تنسيق ملفات محسن مثل HDF5 لتقليل أوقات تحميل البيانات. توفر مكتبات مختلفة مثل H5Py و PyTables و pandas واجهة Python لقراءة وكتابة البيانات بهذا التنسيق.
