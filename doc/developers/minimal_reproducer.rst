
.. _minimal_reproducer:

=================================
صياغة مُكرِّر بسيط لـ scikit-learn
=================================


سواء كان إرسال تقرير خطأ، أو تصميم مجموعة من الاختبارات، أو مجرد نشر سؤال
في المناقشات، فإن القدرة على صياغة أمثلة بسيطة وقابلة للتكرار
(أو أمثلة بسيطة وعاملة) هي المفتاح للتواصل الفعال و
بكفاءة مع المجتمع.

هناك إرشادات جيدة جدًا على الإنترنت مثل `وثيقة StackOverflow
هذه <https://stackoverflow.com/help/mcve>`_ أو `هذه المدونة بواسطة Matthew
Rocklin <https://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports>`_
حول صياغة أمثلة قابلة للتحقق كاملة وبسيطة (يشار إليها أدناه باسم MCVE).
هدفنا ليس أن نكون متكررين مع هذه المراجع، ولكن بالأحرى تقديم
دليل خطوة بخطوة حول كيفية تضييق نطاق الخطأ حتى تصل إلى
أقصر كود ممكن لإعادة إنتاجه.

الخطوة الأولى قبل إرسال تقرير خطأ إلى scikit-learn هي قراءة
`قالب المشكلة
<https://github.com/scikit-learn/scikit-learn/blob/main/.github/ISSUE_TEMPLATE/bug_report.yml>`_.
إنه غني بالمعلومات بالفعل حول المعلومات التي سيُطلب منك
تقديمها.


.. _good_practices:

الممارسات الجيدة
==================

في هذا القسم، سنركز على قسم **الخطوات/الشفرة لإعادة الإنتاج** من
`قالب المشكلة
<https://github.com/scikit-learn/scikit-learn/blob/main/.github/ISSUE_TEMPLATE/bug_report.yml>`_.
سنبدأ بمقتطف شفرة يوفر بالفعل مثالاً فاشلاً ولكنه
يحتوي على مجال لتحسين قابلية القراءة. ثم نصنع MCVE منه.

**مثال**

.. code-block:: python

    # أعمل حاليًا في مشروع ML وعندما حاولت ملاءمة
    # نموذج GradientBoostingRegressor لـ my_data.csv، تلقيت UserWarning:
    # "X لديه أسماء ميزات، ولكن تم ملاءمة DecisionTreeRegressor بدون
    # أسماء ميزات". يمكنك الحصول على نسخة من مجموعة البيانات الخاصة بي من
    # https://example.com/my_data.csv والتحقق من أن ميزاتي تحتوي بالفعل
    # على أسماء. يبدو أن المشكلة تنشأ أثناء التوفيق عندما أمرر عددًا صحيحًا
    # إلى معلمة n_iter_no_change.

    df = pd.read_csv('my_data.csv')
    X = df[["feature_name"]] # ميزاتي تحتوي بالفعل على أسماء
    y = df["target"]

    # نقوم بتعيين random_state=42 لـ train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # نموذج مع n_iter_no_change الافتراضي لا يُثير أي خطأ أو تحذيرات
    gbdt = GradientBoostingRegressor(random_state=0)
    gbdt.fit(X_train, y_train)
    default_score = gbdt.score(X_test, y_test)

    # يظهر الخطأ عندما أغير قيمة n_iter_no_change
    gbdt = GradientBoostingRegressor(random_state=0, n_iter_no_change=5)
    gbdt.fit(X_train, y_train)
    other_score = gbdt.score(X_test, y_test)

    other_score = gbdt.score(X_test, y_test)


تقديم مثال شفرة فاشل مع الحد الأدنى من التعليقات
----------------------------------------------------

غالبًا ما تكون كتابة التعليمات لإعادة إنتاج المشكلة باللغة الإنجليزية غامضة.
من الأفضل التأكد من أن جميع التفاصيل اللازمة لإعادة إنتاج المشكلة
موضحة في مقتطف شفرة Python لتجنب أي غموض. بالإضافة إلى ذلك، في هذه
المرحلة، قدمت بالفعل وصفًا موجزًا ​​في قسم **وصف الخطأ** من
`قالب المشكلة
<https://github.com/scikit-learn/scikit-learn/blob/main/.github/ISSUE_TEMPLATE/bug_report.yml>`_.

الشفرة التالية، على الرغم من **أنها لا تزال غير بسيطة**، فهي **أفضل بكثير**
لأنه يمكن نسخها ولصقها في محطة Python لإعادة إنتاج المشكلة في
خطوة واحدة. على وجه الخصوص:

- تحتوي على **جميع عبارات الاستيراد الضرورية**؛
- يمكنها جلب مجموعة البيانات العامة دون الحاجة إلى تنزيل
  ملف يدويًا ووضعه في الموقع المتوقع على القرص.

**مثال مُحسَّن**

.. code-block:: python

    import pandas as pd

    df = pd.read_csv("https://example.com/my_data.csv")
    X = df[["feature_name"]]
    y = df["target"]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.ensemble import GradientBoostingRegressor

    gbdt = GradientBoostingRegressor(random_state=0)
    gbdt.fit(X_train, y_train)  # لا يوجد تحذير
    default_score = gbdt.score(X_test, y_test)

    gbdt = GradientBoostingRegressor(random_state=0, n_iter_no_change=5)
    gbdt.fit(X_train, y_train)  # يُثير تحذيرًا
    other_score = gbdt.score(X_test, y_test)
    other_score = gbdt.score(X_test, y_test)


اختزل البرنامج النصي الخاص بك إلى شيء صغير قدر الإمكان
-----------------------------------------------------------

عليك أن تسأل نفسك أي أسطر من الشفرة ذات صلة وأيها غير ذات صلة
لإعادة إنتاج الخطأ. سيؤدي حذف أسطر التعليمات البرمجية غير الضرورية أو تبسيط
استدعاءات الدوال عن طريق حذف الخيارات غير الافتراضية غير ذات الصلة إلى مساعدتك أنت
والمساهمين الآخرين على تضييق نطاق سبب الخطأ.

على وجه الخصوص، لهذا المثال المحدد:

- التحذير لا علاقة له بـ `train_test_split` لأنه يظهر بالفعل في خطوة
  التدريب، قبل أن نستخدم مجموعة الاختبار.
- وبالمثل، فإن الأسطر التي تحسب الدرجات في مجموعة الاختبار ليست
  ضرورية؛
- يمكن إعادة إنتاج الخطأ لأي قيمة لـ `random_state` لذا اتركه إلى قيمته
  الافتراضية؛
- يمكن إعادة إنتاج الخطأ دون معالجة البيانات مسبقًا باستخدام
  `StandardScaler`.

**مثال مُحسَّن**

.. code-block:: python

    import pandas as pd
    df = pd.read_csv("https://example.com/my_data.csv")
    X = df[["feature_name"]]
    y = df["target"]

    from sklearn.ensemble import GradientBoostingRegressor

    gbdt = GradientBoostingRegressor()
    gbdt.fit(X, y)  # لا يوجد تحذير

    gbdt = GradientBoostingRegressor(n_iter_no_change=5)
    gbdt.fit(X, y)  # يُثير تحذيرًا


**لا** تُبلغ عن بياناتك إلا إذا كانت ضرورية للغاية
-------------------------------------------------------

الفكرة هي جعل الشفرة مكتفية ذاتيًا قدر الإمكان. للقيام بذلك، يمكنك
استخدام :ref:`synth_data`. يمكن إنشاؤها باستخدام numpy أو pandas أو
وحدة :mod:`sklearn.datasets`. في معظم الأوقات، لا يرتبط الخطأ
ببنية معينة لبياناتك. حتى لو كان الأمر كذلك، فحاول العثور على مجموعة بيانات متاحة
لها خصائص مشابهة لخصائصك وتعيد إنتاج
المشكلة. في هذه الحالة بالذات، نحن مهتمون بالبيانات التي تحتوي على
أسماء ميزات مُعلَّمة.

**مثال مُحسَّن**

.. code-block:: python

    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor

    df = pd.DataFrame(
        {
            "feature_name": [-12.32, 1.43, 30.01, 22.17],
            "target": [72, 55, 32, 43],
        }
    )
    X = df[["feature_name"]]
    y = df["target"]

    gbdt = GradientBoostingRegressor()
    gbdt.fit(X, y) # لا يوجد تحذير
    gbdt = GradientBoostingRegressor(n_iter_no_change=5)
    gbdt.fit(X, y) # يُثير تحذيرًا

كما ذكرنا سابقًا، فإن مفتاح التواصل هو قابلية قراءة الكود، والتنسيق الجيد
يمكن أن يكون إضافة حقيقية. لاحظ أنه في المقتطف السابق، قمنا بما يلي:

- حاول تحديد جميع الأسطر بحد أقصى 79 حرفًا لتجنب أشرطة التمرير الأفقية في
  كتل مقتطفات الشفرة التي يتم عرضها في مشكلة GitHub؛
- استخدم أسطرًا فارغة لفصل مجموعات الدوال ذات الصلة؛
- ضع جميع عمليات الاستيراد في مجموعتها الخاصة في البداية.

يمكن تنفيذ خطوات التبسيط المقدمة في هذا الدليل بترتيب
مختلف عن التقدم الذي عرضناه هنا. النقاط المهمة
هي:

- يجب أن يكون المُكرِّر البسيط قابلاً للتشغيل عن طريق النسخ واللصق البسيط في
  محطة python؛
- يجب تبسيطه قدر الإمكان عن طريق إزالة أي خطوات شفرة
  ليست ضرورية تمامًا لإعادة إنتاج المشكلة الأصلية؛
- من الناحية المثالية، يجب أن يعتمد فقط على مجموعة بيانات بسيطة تم إنشاؤها أثناء
  التشغيل عن طريق تشغيل الشفرة بدلاً من الاعتماد على بيانات خارجية، إن أمكن.


استخدم تنسيق markdown
-----------------------

لتنسيق الشفرة أو النص في كتلة مميزة خاصة به، استخدم علامات اقتباس خلفية ثلاثية.
يدعم `Markdown
<https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax>`_
معرف لغة اختياريًا لتمكين تمييز بناء الجملة في كتلة
الشفرة المسيَّجة الخاصة بك. فمثلا::

    ```python
    from sklearn.datasets import make_blobs

    n_samples = 100
    n_components = 3
    X, y = make_blobs(n_samples=n_samples, centers=n_components)
    ```

سيعرض مقتطفًا منسقًا بلغة python على النحو التالي

.. code-block:: python

    from sklearn.datasets import make_blobs

    n_samples = 100
    n_components = 3
    X, y = make_blobs(n_samples=n_samples, centers=n_components)


ليس من الضروري إنشاء عدة كتل من الشفرة عند إرسال تقرير خطأ.
تذكر أن المراجعين الآخرين سينسخون الشفرة الخاصة بك ولصقها، وسيؤدي وجود
خلية واحدة إلى تسهيل مهمتهم.

في القسم المسمى **النتائج الفعلية** من `قالب المشكلة
<https://github.com/scikit-learn/scikit-learn/blob/main/.github/ISSUE_TEMPLATE/bug_report.yml>`_
، يُطلب منك تقديم رسالة الخطأ بما في ذلك التتبع الكامل للاستثناء.
في هذه الحالة، استخدم مُؤهِّل `python-traceback`. فمثلا::

    ```python-traceback
    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    <ipython-input-1-a674e682c281> in <module>
        4 vectorizer = CountVectorizer(input=docs, analyzer='word')
        5 lda_features = vectorizer.fit_transform(docs)
    ----> 6 lda_model = LatentDirichletAllocation(
        7     n_topics=10,
        8     learning_method='online',

    TypeError: __init__() got an unexpected keyword argument 'n_topics'
    ```

ينتج ما يلي عند العرض:

.. code-block:: python

    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    <ipython-input-1-a674e682c281> in <module>
        4 vectorizer = CountVectorizer(input=docs, analyzer='word')
        5 lda_features = vectorizer.fit_transform(docs)
    ----> 6 lda_model = LatentDirichletAllocation(
        7     n_topics=10,
        8     learning_method='online',

    TypeError: __init__() got an unexpected keyword argument 'n_topics'


.. _synth_data:

مجموعة بيانات تركيبية
========================

قبل اختيار مجموعة بيانات تركيبية معينة، عليك أولاً تحديد
نوع المشكلة التي تحلها: هل هي تصنيف، انحدار،
تجميع، إلخ؟

بمجرد تضييق نطاق نوع المشكلة، تحتاج إلى توفير مجموعة بيانات تركيبية
وفقًا لذلك. في معظم الأوقات، تحتاج فقط إلى مجموعة بيانات بسيطة.
إليك قائمة غير شاملة بالأدوات التي قد تساعدك.

NumPy
-----

يمكن استخدام أدوات NumPy مثل `numpy.random.randn
<https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html>`_
و `numpy.random.randint
<https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html>`_
لإنشاء بيانات رقمية وهمية.

- الانحدار

  يأخذ الانحدار بيانات رقمية متصلة كميزات وهدف.

  .. code-block:: python

      import numpy as np

      rng = np.random.RandomState(0)
      n_samples, n_features = 5, 5
      X = rng.randn(n_samples, n_features)
      y = rng.randn(n_samples)

يمكن استخدام مقتطف مشابه كبيانات تركيبية عند اختبار أدوات القياس
مثل :class:`sklearn.preprocessing.StandardScaler`.

- التصنيف

  إذا لم يتم طرح الخطأ أثناء تشفير متغير فئوي، فيمكنك
  تغذية البيانات الرقمية لمصنف. فقط تذكر التأكد من أن الهدف
  هو بالفعل عدد صحيح.

  .. code-block:: python

      import numpy as np

      rng = np.random.RandomState(0)
      n_samples, n_features = 5, 5
      X = rng.randn(n_samples, n_features)
      y = rng.randint(0, 2, n_samples)  # هدف ثنائي بقيم في {0، 1}


  إذا حدث الخطأ فقط مع تسميات الفئات غير الرقمية، فقد ترغب في
  إنشاء هدف عشوائي باستخدام `numpy.random.choice
  <https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html>`_.

  .. code-block:: python

      import numpy as np

      rng = np.random.RandomState(0)
      n_samples, n_features = 50, 5
      X = rng.randn(n_samples, n_features)
      y = np.random.choice(
          ["male", "female", "other"], size=n_samples, p=[0.49, 0.49, 0.02]
      )


Pandas
------

تتوقع بعض كائنات scikit-learn إطارات بيانات pandas كمدخلات. في هذه الحالة، يمكنك
تحويل مصفوفات numpy إلى كائنات pandas باستخدام `pandas.DataFrame
<https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_، أو
`pandas.Series
<https://pandas.pydata.org/docs/reference/api/pandas.Series.html>`_.

.. code-block:: python

    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(0)
    n_samples, n_features = 5, 5
    X = pd.DataFrame(
        {
            "continuous_feature": rng.randn(n_samples),
            "positive_feature": rng.uniform(low=0.0, high=100.0, size=n_samples),
            "categorical_feature": rng.choice(["a", "b", "c"], size=n_samples),
        }
    )
    y = pd.Series(rng.randn(n_samples))

بالإضافة إلى ذلك، يتضمن scikit-learn العديد من :ref:`sample_generators` التي يمكن
استخدامها لبناء مجموعات بيانات اصطناعية ذات حجم وتعقيد متحكم بهما.

`make_regression`
-----------------

كما يوحي الاسم، ينتج :class:`sklearn.datasets.make_regression`
أهداف انحدار مع ضوضاء كمزيج خطي عشوائي اختياريًا متناثر
من ميزات عشوائية.

.. code-block:: python

    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=20)

`make_classification`
---------------------

ينشئ :class:`sklearn.datasets.make_classification` مجموعات بيانات متعددة الفئات مع مجموعات غاوسية
متعددة لكل فئة. يمكن إدخال الضوضاء عن طريق ميزات مترابطة أو زائدة عن الحاجة أو
غير مفيدة.

.. code-block:: python

    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1
    )

`make_blobs`
------------

على غرار `make_classification`، ينشئ :class:`sklearn.datasets.make_blobs`
مجموعات بيانات متعددة الفئات باستخدام مجموعات من النقاط موزعة بشكل طبيعي. يوفر
تحكمًا أكبر فيما يتعلق بمراكز وانحرافات معيارية لكل مجموعة،
وبالتالي فهو مفيد لعرض التجميع.

.. code-block:: python

    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=10, centers=3, n_features=2)

أدوات مساعدة لتحميل مجموعة البيانات
--------------------------------------

يمكنك استخدام :ref:`datasets` لتحميل وجلب العديد من مجموعات البيانات
المرجعية الشائعة. يكون هذا الخيار مفيدًا عندما يرتبط الخطأ ببنية معينة
للبيانات، على سبيل المثال التعامل مع القيم المفقودة أو التعرف على الصور.

.. code-block:: python

    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)


