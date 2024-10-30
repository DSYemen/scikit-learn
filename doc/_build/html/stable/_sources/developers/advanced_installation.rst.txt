.. _advanced-installation:

.. include:: ../min_dependency_substitutions.rst

==================================================
تثبيت نسخة التطوير من scikit-learn
==================================================

يقدم هذا القسم كيفية تثبيت **الفرع الرئيسي** من scikit-learn.
يمكن القيام بذلك إما عن طريق تثبيت نسخة ليلية أو البناء من المصدر.

.. _install_nightly_builds:

تثبيت النسخ الليلية
=========================

تقوم خوادم التكامل المستمر لمشروع scikit-learn ببناء واختبار وتحميل حزم wheel لأحدث إصدار من Python على أساس ليلي.

يُعد تثبيت نسخة ليلية أسرع طريقة لـ:

- تجربة ميزة جديدة سيتم شحنها في الإصدار التالي (أي ميزة من طلب سحب تم دمجه مؤخرًا في الفرع الرئيسي)؛

- التحقق مما إذا كان قد تم إصلاح خطأ واجهته منذ الإصدار الأخير.

يمكنك تثبيت النسخة الليلية من scikit-learn باستخدام فهرس `scientific-python-nightly-wheels`
من سجل PyPI لـ `anaconda.org`:

.. prompt:: bash $

  pip install --pre --extra-index https://pypi.anaconda.org/scientific-python-nightly-wheels/simple scikit-learn

لاحظ أنه قد يلزم إلغاء تثبيت scikit-learn أولاً لتتمكن من تثبيت النسخ الليلية من scikit-learn.

.. _install_bleeding_edge:

البناء من المصدر
====================

البناء من المصدر مطلوب للعمل على مساهمة (إصلاح خطأ، ميزة جديدة، تحسين التعليمات البرمجية أو الوثائق).

.. _git_repo:

#. استخدم `Git <https://git-scm.com/>`_ للتحقق من أحدث مصدر من
   `مستودع scikit-learn <https://github.com/scikit-learn/scikit-learn>`_ على
   Github.:

   .. prompt:: bash $

     git clone git@github.com:scikit-learn/scikit-learn.git  # add --depth 1 if your connection is slow
     cd scikit-learn

   إذا كنت تخطط لتقديم طلب سحب، فيجب عليك الاستنساخ من الشوكة الخاصة بك
   بدلاً من ذلك.

#. ثبّت إصدارًا حديثًا من Python (3.9 أو أحدث وقت كتابة هذا التقرير) على سبيل المثال باستخدام Miniforge3_. يوفر Miniforge توزيعًا قائمًا على conda لـ
   Python وأكثر المكتبات العلمية شيوعًا.

   إذا قمت بتثبيت Python باستخدام conda، فإننا نوصي بإنشاء
   `بيئة conda`_ مخصصة مع جميع تبعيات البناء الخاصة بـ scikit-learn
   (أي NumPy_ و SciPy_ و Cython_ و meson-python_ و Ninja_):

   .. prompt:: bash $

     conda create -n sklearn-env -c conda-forge python numpy scipy cython meson-python ninja

   ليس من الضروري دائمًا ولكن من الأكثر أمانًا فتح موجه أوامر جديد قبل
   تنشيط بيئة conda المنشأة حديثًا.

   .. prompt:: bash $

     conda activate sklearn-env

#. **بديل لـ conda:** يمكنك استخدام عمليات تثبيت بديلة لـ Python
   بشرط أن تكون حديثة بما فيه الكفاية (3.9 أو أعلى وقت كتابة هذا التقرير).
   فيما يلي مثال حول كيفية إنشاء بيئة بناء لـ Python على نظام Linux. يتم تثبيت تبعيات البناء باستخدام `pip` في virtualenv_ مخصص
   لتجنب تعطيل برامج Python الأخرى المثبتة على النظام:

   .. prompt:: bash $

     python3 -m venv sklearn-env
     source sklearn-env/bin/activate
     pip install wheel numpy scipy cython meson-python ninja

#. ثبّت مترجمًا مع دعم OpenMP_ لمنصتك. انظر التعليمات الخاصة بـ :ref:`compiler_windows` و :ref:`compiler_macos` و :ref:`compiler_linux`
   و :ref:`compiler_freebsd`.

#. بناء المشروع باستخدام pip:

   .. prompt:: bash $

     pip install --editable . \
        --verbose --no-build-isolation \
        --config-settings editable-verbose=true

#. تحقق من أن scikit-learn المثبت لديه رقم إصدار ينتهي بـ
   `.dev0`:

   .. prompt:: bash $

     python -c "import sklearn; sklearn.show_versions()"

#. يرجى الرجوع إلى :ref:`developers_guide` و :ref:`pytest_tips` لتشغيل
   الاختبارات على الوحدة التي تختارها.

.. note::

    `--config-settings editable-verbose=true` اختياري ولكنه موصى به
    لتجنب المفاجآت عند استيراد `sklearn`. يُنفِّذ `meson-python`
    عمليات التثبيت القابلة للتحرير عن طريق إعادة بناء `sklearn` عند تنفيذ `import sklearn`.
    مع الإعداد الموصى به، سترى رسالة عندما يحدث هذا،
    بدلاً من الانتظار المحتمل بدون تغذية مرتدة والتساؤل
    عما يستغرق وقتًا طويلاً. مكافأة: هذا يعني أنك تحتاج فقط إلى تشغيل الأمر `pip
    install` مرة واحدة، سيتم إعادة بناء `sklearn` تلقائيًا عند
    استيراد `sklearn`.

التبعيات
------------

تبعيات وقت التشغيل
~~~~~~~~~~~~~~~~~~~~

يتطلب Scikit-learn التبعيات التالية في وقت البناء وفي وقت التشغيل:

- Python (>= 3.8)
- NumPy (>= |NumpyMinVersion|)
- SciPy (>= |ScipyMinVersion|)
- Joblib (>= |JoblibMinVersion|)
- threadpoolctl (>= |ThreadpoolctlMinVersion|).

تبعيات البناء
~~~~~~~~~~~~~~~~~~

يتطلب بناء Scikit-learn أيضًا:

..
    # يجب أن تكون الأماكن التالية متزامنة فيما يتعلق بإصدار Cython:
    # - ملف تكوين .circleci
    # - sklearn/_build_utils/__init__.py
    # - دليل التثبيت المتقدم

- Cython >= |CythonMinVersion|
- مترجم C/C++ ومكتبة وقت تشغيل OpenMP_ متوافقة. انظر
  :ref:`التعليمات الخاصة بنظام المنصة
  <platform_specific_instructions>` لمزيد من التفاصيل.

.. note::

   إذا لم يكن OpenMP مدعومًا من قبل المترجم، فسيتم البناء مع
   تعطيل وظائف OpenMP. لا يوصى بذلك لأنه سيجبر
   بعض المقدرات على التشغيل في الوضع التسلسلي بدلاً من الاستفادة من التوازي القائم على الخيوط. سيؤدي تعيين متغير البيئة ``SKLEARN_FAIL_NO_OPENMP``
   (قبل cythonization) إلى فشل البناء إذا لم يكن OpenMP
   مدعومًا.

منذ الإصدار 0.21، يكتشف scikit-learn تلقائيًا ويستخدم مكتبة الجبر الخطي التي يستخدمها SciPy **في وقت التشغيل**. لذلك ليس لدى Scikit-learn
تبعية بناء على تطبيقات BLAS/LAPACK مثل OpenBlas أو Atlas أو Blis
أو MKL.

تبعيات الاختبار
~~~~~~~~~~~~~~~~~

يتطلب تشغيل الاختبارات:

- pytest >= |PytestMinVersion|

تتطلب بعض الاختبارات أيضًا `pandas <https://pandas.pydata.org>`_.


بناء إصدار معين من علامة
--------------------------------------

إذا كنت ترغب في بناء إصدار ثابت، يمكنك ``git checkout <VERSION>``
للحصول على التعليمات البرمجية لهذا الإصدار المحدد، أو تنزيل أرشيف مضغوط لـ
الإصدار من github.

.. _platform_specific_instructions:

تعليمات خاصة بالمنصة
==============================

فيما يلي تعليمات لتثبيت مترجم C/C++ عامل مع دعم OpenMP
لبناء ملحقات Cython الخاصة بـ scikit-learn لكل منصة مدعومة.

.. _compiler_windows:

Windows
-------

أولاً، قم بتنزيل `أدوات البناء لـ Visual Studio 2019 المثبت
<https://aka.ms/vs/17/release/vs_buildtools.exe>`_.

قم بتشغيل ملف `vs_buildtools.exe` الذي تم تنزيله، أثناء التثبيت ستحتاج إلى التأكد من تحديد "تطوير سطح المكتب باستخدام C++"، على غرار لقطة الشاشة هذه:

.. image:: ../images/visual-studio-build-tools-selection.png

ثانيًا، اكتشف ما إذا كنت تقوم بتشغيل Python 64 بت أو 32 بت. يعتمد أمر البناء على بنية مترجم Python. يمكنك التحقق
من البنية عن طريق تشغيل ما يلي في وحدة تحكم ``cmd`` أو ``powershell``:

.. prompt:: bash $

    python -c "import struct; print(struct.calcsize('P') * 8)"

بالنسبة لـ Python 64 بت، قم بتكوين بيئة البناء عن طريق تشغيل الأوامر التالية في ``cmd`` أو موجه أوامر Anaconda (إذا كنت تستخدم Anaconda):

.. sphinx-prompt 1.3.0 (المستخدم في مهمة CI doc-min-dependencies) لا يدعم نوع موجه `batch`،
.. لذلك نتجاوز ذلك باستخدام نوع موجه معروف ونص موجه صريح.
..
.. prompt:: bash C:\>

    SET DISTUTILS_USE_SDK=1
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

استبدل ``x64`` بـ ``x86`` للبناء لـ Python 32 بت.

يرجى العلم أن المسار أعلاه قد يختلف من مستخدم لآخر. الهدف هو الإشارة إلى ملف "vcvarsall.bat" الذي سيُعيِّن
متغيرات البيئة اللازمة في موجه الأوامر الحالي.

أخيرًا، قم ببناء scikit-learn باستخدام موجه الأوامر هذا:

.. prompt:: bash $

    pip install --editable . \
        --verbose --no-build-isolation \
        --config-settings editable-verbose=true

.. _compiler_macos:

macOS
-----

لا يدعم مترجم C الافتراضي على macOS، Apple clang (المُلقَّب بشكل مُربِك باسم
`/usr/bin/gcc`)، OpenMP مباشرةً. نقدم بديلين
لتفعيل دعم OpenMP:

- إما تثبيت `conda-forge::compilers` باستخدام conda؛

- أو تثبيت `libomp` باستخدام Homebrew لتوسيع مترجم Apple clang الافتراضي.

بالنسبة لأجهزة Apple Silicon M1، من المعروف أن طريقة conda-forge أدناه فقط هي التي تعمل وقت كتابة هذا التقرير (يناير 2021). يمكنك تثبيت توزيع `macos/arm64`
لـ conda باستخدام `مثبت miniforge
<https://github.com/conda-forge/miniforge#miniforge>`_

مترجمات macOS من conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

إذا كنت تستخدم مدير حزم conda (الإصدار >= 4.7)، يمكنك تثبيت
حزمة تعريف ``compilers`` من قناة conda-forge، والتي توفر
مترجمات C/C++ تدعم OpenMP استنادًا إلى سلسلة أدوات llvm.

قم أولاً بتثبيت أدوات سطر أوامر macOS:

.. prompt:: bash $

    xcode-select --install

يوصى باستخدام `بيئة conda`_ مخصصة لـ
بناء scikit-learn من المصدر:

.. prompt:: bash $

    conda create -n sklearn-dev -c conda-forge python numpy scipy cython \
        joblib threadpoolctl pytest compilers llvm-openmp meson-python ninja

ليس من الضروري دائمًا ولكن من الأكثر أمانًا فتح موجه أوامر جديد قبل
تنشيط بيئة conda المنشأة حديثًا.

.. prompt:: bash $

    conda activate sklearn-dev
    make clean
    pip install --editable . \
        --verbose --no-build-isolation \
        --config-settings editable-verbose=true

.. note::

    إذا تلقيت أي رسالة خطأ تتعلق بتعارض التبعيات، فحاول التعليق على
    أي تكوين conda مخصص في ملف ``$HOME/.condarc``. على وجه الخصوص، من المعروف أن التوجيه ``channel_priority: strict`` يسبب
    مشاكل لهذا الإعداد.


يمكنك التحقق من تثبيت المترجمات المخصصة بشكل صحيح من conda
forge باستخدام الأمر التالي:

.. prompt:: bash $

    conda list

الذي يجب أن يتضمن ``compilers`` و ``llvm-openmp``.

ستقوم حزمة تعريف المترجمات بتعيين متغيرات بيئة مخصصة تلقائيًا:

.. prompt:: bash $

    echo $CC
    echo $CXX
    echo $CFLAGS
    echo $CXXFLAGS
    echo $LDFLAGS

تشير إلى الملفات والمجلدات من بيئة conda ``sklearn-dev`` الخاصة بك
(على وجه الخصوص في مجلدات bin/ و include/ و lib/ الفرعية). على سبيل المثال
``-L/path/to/conda/envs/sklearn-dev/lib`` يجب أن يظهر في ``LDFLAGS``.

في السجل، يجب أن ترى الامتداد المترجم الذي يتم بناؤه باستخدام
مترجمي clang و clang++ المثبتين بواسطة conda مع علامة سطر الأوامر ``-fopenmp``.

مترجمات macOS من Homebrew
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

هناك حل آخر وهو تمكين دعم OpenMP لمترجم clang الذي يأتي
افتراضيًا على macOS.

قم أولاً بتثبيت أدوات سطر أوامر macOS:

.. prompt:: bash $

    xcode-select --install

ثبّت مدير حزم Homebrew_ لـ macOS.

ثبّت مكتبة LLVM OpenMP:

.. prompt:: bash $

    brew install libomp

عيِّن متغيرات البيئة التالية:

.. prompt:: bash $

    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
    export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
    export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
    export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib -L/usr/local/opt/libomp/lib -lomp"

أخيرًا، قم ببناء scikit-learn في الوضع المطول (للتحقق من وجود
علامة ``-fopenmp`` في أوامر المترجم):

.. prompt:: bash $

    make clean
    pip install --editable . \
        --verbose --no-build-isolation \
        --config-settings editable-verbose=true

.. _compiler_linux:

Linux
-----

مترجمات Linux من النظام
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

يتطلب تثبيت scikit-learn من المصدر بدون استخدام conda تثبيت
رؤوس تطوير Python الخاصة بـ scikit-learn ومترجم C/C++ عامل
مع دعم OpenMP (عادةً سلسلة أدوات GCC).

ثبّت تبعيات البناء لأنظمة التشغيل القائمة على Debian، على سبيل المثال
Ubuntu:

.. prompt:: bash $

    sudo apt-get install build-essential python3-dev python3-pip

ثم تابع كالمعتاد:

.. prompt:: bash $

    pip3 install cython
    pip3 install --editable . \
        --verbose --no-build-isolation \
        --config-settings editable-verbose=true

يجب تثبيت Cython والعجلات المترجمة مسبقًا لتبعيات وقت التشغيل (numpy و scipy
و joblib) تلقائيًا في
``$HOME/.local/lib/pythonX.Y/site-packages``. بدلاً من ذلك، يمكنك تشغيل
الأوامر أعلاه من virtualenv_ أو `بيئة conda`_ للحصول على عزل كامل عن حزم Python المثبتة عبر مدير حزم النظام. عند استخدام بيئة معزولة، يجب استبدال ``pip3`` بـ ``pip`` في
الأوامر أعلاه.

عندما لا تتوفر عجلات التبعيات المترجمة مسبقًا لبنيتك
(مثل ARM)، يمكنك تثبيت إصدارات النظام:

.. prompt:: bash $

    sudo apt-get install cython3 python3-numpy python3-scipy

على Red Hat والمستنسخات (مثل CentOS)، قم بتثبيت التبعيات باستخدام:

.. prompt:: bash $

    sudo yum -y install gcc gcc-c++ python3-devel numpy scipy

مترجمات Linux من conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

بدلاً من ذلك، قم بتثبيت إصدار حديث من سلسلة أدوات مترجم GNU C (GCC)
في مجلد المستخدم باستخدام conda:

.. prompt:: bash $

    conda create -n sklearn-dev -c conda-forge python numpy scipy cython \
        joblib threadpoolctl pytest compilers meson-python ninja

ليس من الضروري دائمًا ولكن من الأكثر أمانًا فتح موجه أوامر جديد قبل
تنشيط بيئة conda المنشأة حديثًا.

.. prompt:: bash $

    conda activate sklearn-dev
    pip install --editable . \
        --verbose --no-build-isolation \
        --config-settings editable-verbose=true

.. _compiler_freebsd:

FreeBSD
-------

لا يتضمن مترجم clang المضمن في أنظمة FreeBSD 12.0 و 11.2 الأساسية
دعم OpenMP. تحتاج إلى تثبيت مكتبة `openmp` من الحزم
(أو المنافذ):

.. prompt:: bash $

    sudo pkg install openmp

سيؤدي ذلك إلى تثبيت ملفات الرأس في ``/usr/local/include`` والملفات lib في
``/usr/local/lib``. نظرًا لأنه لا يتم البحث في هذه الأدلة افتراضيًا، يمكنك
تعيين متغيرات البيئة لهذه المواقع:

.. prompt:: bash $

    export CFLAGS="$CFLAGS -I/usr/local/include"
    export CXXFLAGS="$CXXFLAGS -I/usr/local/include"
    export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/lib -L/usr/local/lib -lomp"


أخيرًا، قم ببناء الحزمة باستخدام الأمر القياسي:

.. prompt:: bash $

    pip install --editable . \
        --verbose --no-build-isolation \
        --config-settings editable-verbose=true

بالنسبة لإصدارات FreeBSD 12.1 و 11.3 القادمة، سيتم تضمين OpenMP في
النظام الأساسي ولن تكون هذه الخطوات ضرورية.

.. _OpenMP: https://en.wikipedia.org/wiki/OpenMP
.. _Cython: https://cython.org
.. _meson-python: https://mesonbuild.com/meson-python
.. _Ninja: https://ninja-build.org/
.. _NumPy: https://numpy.org
.. _SciPy: https://www.scipy.org
.. _Homebrew: https://brew.sh
.. _virtualenv: https://docs.python.org/3/tutorial/venv.html
.. _conda environment: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
.. _Miniforge3: https://github.com/conda-forge/miniforge#miniforge3
