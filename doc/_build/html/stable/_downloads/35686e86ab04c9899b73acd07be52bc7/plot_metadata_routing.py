"""
==========================
توجيه البيانات الوصفية
==========================

.. currentmodule:: sklearn

توضح هذه الوثيقة كيفية استخدام آلية توجيه البيانات الوصفية
<metadata_routing> في scikit-learn لتوجيه البيانات الوصفية إلى
المقدرات، والمقيمين، ومقسمات CV التي تستهلكها.

لفهم الوثيقة التالية بشكل أفضل، نحتاج إلى تقديم مفهومين:
الموجهات والمستهلكين. الموجه هو كائن يقوم بتوجيه بعض البيانات والبيانات
الوصفية المعطاة إلى كائنات أخرى. في معظم الحالات، يكون الموجه عبارة عن
:term:`meta-estimator`، أي مقدر يأخذ مقدرًا آخر كمعلمة. وظيفة مثل
:func:`sklearn.model_selection.cross_validate` التي تأخذ مقدرًا كمعلمة
وتقوم بتوجيه البيانات والبيانات الوصفية، هي أيضًا موجه.

من ناحية أخرى، المستهلك هو كائن يقبل ويستخدم بعض البيانات الوصفية
المعطاة. على سبيل المثال، مقدر يأخذ في الاعتبار "sample_weight" في
طريقته :term:`fit` هو مستهلك "sample_weight".

من الممكن أن يكون الكائن موجهًا ومستهلكًا في نفس الوقت. على سبيل المثال،
قد يأخذ الميتا-مقدر في الاعتبار "sample_weight" في حسابات معينة، ولكنه
قد يقوم أيضًا بتوجيهه إلى المقدر الأساسي.

أولاً بعض الاستيرادات وبعض البيانات العشوائية لبقية البرنامج النصي.
"""

# المؤلفون: مطوري scikit-learn
# معرف SPDX-License: BSD-3-Clause

# %%

import warnings
from pprint import pprint

import numpy as np

from sklearn import set_config
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
)
from sklearn.linear_model import LinearRegression
from sklearn.utils import metadata_routing
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    get_routing_for_object,
    process_routing,
)
from sklearn.utils.validation import check_is_fitted

n_samples, n_features = 100, 4
rng = np.random.RandomState(42)
X = rng.rand(n_samples, n_features)
y = rng.randint(0, 2, size=n_samples)
my_groups = rng.randint(0, 10, size=n_samples)
my_weights = rng.rand(n_samples)
my_other_weights = rng.rand(n_samples)

# %%
# توجيه البيانات الوصفية متاح فقط إذا تم تمكينه بشكل صريح:
set_config(enable_metadata_routing=True)


# %%
# هذه الدالة المساعدة هي دمية للتحقق مما إذا تم تمرير البيانات الوصفية:
def check_metadata(obj, **kwargs):
    for key, value in kwargs.items():
        if value is not None:
            print(
                f"Received {key} of length = {len(value)} in {obj.__class__.__name__}."
            )
        else:
            print(f"{key} is None in {obj.__class__.__name__}.")


# %%
# دالة مساعدة لطباعة معلومات التوجيه بشكل جميل:
def print_routing(obj):
    pprint(obj.get_metadata_routing()._serialize())


# %%
# المُقدر المستهلك
# -------------------
# هنا نُظهر كيف يمكن لمقدر أن يعرض واجهة برمجة التطبيقات المطلوبة لدعم
# توجيه البيانات الوصفية كمستهلك. تخيل مصنفًا بسيطًا يقبل
# "sample_weight" كبيانات وصفية على "fit" و "groups" في
# طريقة "predict":


class ExampleClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        check_metadata(self, sample_weight=sample_weight)
        # جميع المصنفات تحتاج إلى عرض سمة classes_ بمجرد ملائمتها.
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X, groups=None):
        check_metadata(self, groups=groups)
        # إرجاع قيمة ثابتة 1، ليس مصنفًا ذكيًا جدًا!
        return np.ones(len(X))


# %%
# الآن يمتلك المُقدر أعلاه كل ما يحتاجه لاستهلاك البيانات الوصفية. هذا
# يتم إنجازه ببعض السحر الذي يتم في :class:`~base.BaseEstimator`. هناك
# الآن ثلاث طرق معروضة بواسطة الفئة أعلاه: "set_fit_request"،
# "set_predict_request"، و "get_metadata_routing". هناك أيضًا
# "set_score_request" لـ "sample_weight" الموجودة منذ
# :class:`~base.ClassifierMixin` تنفذ طريقة "score" التي تقبل
# "sample_weight". وينطبق الشيء نفسه على المقدرات التي ترث من
# :class:`~base.RegressorMixin`.
#
# بشكل افتراضي، لا يتم طلب أي بيانات وصفية، والتي يمكننا رؤيتها على النحو التالي:

print_routing(ExampleClassifier())

# %%
# يعني الإخراج أعلاه أن "sample_weight" و "groups" غير مطلوبة
# بواسطة `ExampleClassifier`، وإذا تم إعطاء موجه هذه البيانات الوصفية، فإنه
# يجب أن يرفع خطأ، حيث لم يحدد المستخدم صراحة ما إذا كانوا مطلوبين أم لا.
# وينطبق الشيء نفسه على "sample_weight" في طريقة "score"، والتي يتم
# وراثتها من :class:`~base.ClassifierMixin`. من أجل تحديد قيم الطلب
# صراحة لهذه البيانات الوصفية، يمكننا استخدام هذه الطرق:

est = (
    ExampleClassifier()
    .set_fit_request(sample_weight=False)
    .set_predict_request(groups=True)
    .set_score_request(sample_weight=False)
)
print_routing(est)

# %%
# .. note ::
#     يرجى ملاحظة أنه طالما أن المُقدر أعلاه لا يتم استخدامه في
#     ميتا-مقدر، لا يحتاج المستخدم إلى تحديد أي طلبات للبيانات
#     الوصفية والقيم المحددة يتم تجاهلها، حيث أن المستهلك لا
#     يتحقق من صحة البيانات الوصفية المعطاة أو يقوم بتوجيهها. سيؤدي
#     الاستخدام البسيط للمُقدر أعلاه إلى العمل كما هو متوقع.

est = ExampleClassifier()
est.fit(X, y, sample_weight=my_weights)
est.predict(X[:3, :], groups=my_groups)

# %%
# المُقدر الموجه
# ----------------------
# الآن، نُظهر كيفية تصميم ميتا-مقدر ليكون موجهًا. كمثال مبسط،
# هنا ميتا-مقدر، لا يفعل الكثير غير توجيه البيانات الوصفية.


class MetaClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def get_metadata_routing(self):
        # هذه الطريقة تحدد التوجيه لهذا الميتا-مقدر.
        # من أجل القيام بذلك، يتم إنشاء مثيل `MetadataRouter`، ويتم إضافة
        # التوجيه إليه. تليها المزيد من التوضيحات أدناه.
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.estimator,
            method_mapping=MethodMapping()
            .add(caller="fit", callee="fit")
            .add(caller="predict", callee="predict")
            .add(caller="score", callee="score"),
        )
        return router

    def fit(self, X, y, **fit_params):
        # `get_routing_for_object` يعيد نسخة من `MetadataRouter`
        # التي تم إنشاؤها بواسطة طريقة `get_metadata_routing` أعلاه، والتي يتم
        # استدعاؤها داخليًا.
        request_router = get_routing_for_object(self)
        # الميتا-مقدرات مسؤولة عن التحقق من صحة البيانات الوصفية المعطاة.
        # `method` تشير إلى طريقة الوالد، أي `fit` في هذا المثال.
        request_router.validate_metadata(params=fit_params, method="fit")
        # `MetadataRouter.route_params` يقوم برسم خريطة للبيانات الوصفية المعطاة
        # إلى البيانات الوصفية المطلوبة بواسطة المُقدر الأساسي بناءً على
        # معلومات التوجيه المحددة بواسطة MetadataRouter. الإخراج من النوع
        # `Bunch` يحتوي على مفتاح لكل كائن مستهلك وتلك التي تحتوي على مفاتيح
        # لطرقهم المستهلكة، والتي تحتوي بعد ذلك على مفاتيح للبيانات الوصفية
        # التي يجب توجيهها إليهم.
        routed_params = request_router.route_params(params=fit_params, caller="fit")

        # يتم ملاءمة مُقدر فرعي وتعيين فئاته إلى الميتا-مقدر.
        self.estimator_ = clone(self.estimator).fit(X, y, **routed_params.estimator.fit)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X, **predict_params):
        check_is_fitted(self)
        # كما في `fit`، نحصل على نسخة من MetadataRouter للكائن،
        request_router = get_routing_for_object(self)
        # ثم نقوم بالتحقق من صحة البيانات الوصفية المعطاة،
        request_router.validate_metadata(params=predict_params, method="predict")
        # ثم نعد الإدخال إلى طريقة "predict" الأساسية.
        routed_params = request_router.route_params(
            params=predict_params, caller="predict"
        )
        return self.estimator_.predict(X, **routed_params.estimator.predict)


# %%
# دعنا نحلل الأجزاء المختلفة من الكود أعلاه.
#
# أولاً، :meth:`~utils.metadata_routing.get_routing_for_object` يأخذ
# الميتا-مقدر (``self``) ويعيد
# :class:`~utils.metadata_routing.MetadataRouter` أو،
# :class:`~utils.metadata_routing.MetadataRequest` إذا كان الكائن مستهلكًا،
# بناءً على إخراج طريقة "get_metadata_routing" للمُقدر.
#
# ثم في كل طريقة، نستخدم طريقة "route_params" لإنشاء قاموس على شكل
# ``{"object_name": {"method_name": {"metadata": value}}}`` لتمريره إلى
# طريقة المُقدر الأساسي. "object_name" (``estimator`` في مثال
# "routed_params.estimator.fit" أعلاه) هو نفسه الذي تمت إضافته في
# "get_metadata_routing". "validate_metadata" تتأكد من أن جميع البيانات
# الوصفية المعطاة مطلوبة لتجنب الأخطاء الصامتة.
#
# بعد ذلك، نوضح السلوكيات المختلفة، وخاصة نوع الأخطاء التي تم
# إثارتها.

meta_est = MetaClassifier(
    estimator=ExampleClassifier().set_fit_request(sample_weight=True)
)
meta_est.fit(X, y, sample_weight=my_weights)

# %%
# لاحظ أن المثال أعلاه يستدعي دالة المساعدة الخاصة بنا
# `check_metadata()` عبر `ExampleClassifier`. يتحقق من أن
# "sample_weight" يتم تمريره بشكل صحيح إليه. إذا لم يكن كذلك، مثل
# في المثال التالي، فإنه سيطبع أن "sample_weight" هو "None":

meta_est.fit(X, y)

# %%
# إذا قمنا بتمرير بيانات وصفية غير معروفة، يتم إثارة خطأ:
try:
    meta_est.fit(X, y, test=my_weights)
except TypeError as e:
    print(e)

# %%
# وإذا قمنا بتمرير بيانات وصفية غير مطلوبة صراحة:
try:
    meta_est.fit(X, y, sample_weight=my_weights).predict(X, groups=my_groups)
except ValueError as e:
    print(e)

# %%
# أيضًا، إذا قمنا بتحديد صراحة أنه غير مطلوب، ولكنه يتم توفيره:
meta_est = MetaClassifier(
    estimator=ExampleClassifier()
    .set_fit_request(sample_weight=True)
    .set_predict_request(groups=False)
)
try:
    meta_est.fit(X, y, sample_weight=my_weights).predict(X[:3, :], groups=my_groups)
except TypeError as e:
    print(e)

# %%
# مفهوم آخر يجب تقديمه هو **البيانات الوصفية المُستعارة**. هذا عندما
# يطلب مُقدر بيانات وصفية باسم متغير مختلف عن اسم المتغير الافتراضي.
# على سبيل المثال، في إعداد حيث يوجد مُقدران في خط أنابيب، يمكن لأحدهما
# أن يطلب "sample_weight1" والآخر "sample_weight2". لاحظ أن هذا لا
# يغير ما يتوقعه المُقدر، فهو يخبر الميتا-مقدر فقط كيفية رسم خريطة
# للبيانات الوصفية المقدمة إلى ما هو مطلوب. إليك مثال، حيث نقوم بتمرير
# "aliased_sample_weight" إلى الميتا-مقدر، ولكن الميتا-مقدر يفهم أن
# "aliased_sample_weight" هو مستعار لـ "sample_weight"، ويقوم بتمريره
# كـ "sample_weight" إلى المُقدر الأساسي:
meta_est = MetaClassifier(
    estimator=ExampleClassifier().set_fit_request(sample_weight="aliased_sample_weight")
)
meta_est.fit(X, y, aliased_sample_weight=my_weights)

# %%
# تمرير "sample_weight" هنا سيفشل لأنه يتم طلبها بمستعار و"sample_weight"
# بهذا الاسم غير مطلوب:
try:
    meta_est.fit(X, y, sample_weight=my_weights)
except TypeError as e:
    print(e)

# %%
# هذا يقودنا إلى "get_metadata_routing". الطريقة التي يعمل بها التوجيه في
# scikit-learn هي أن المستهلكين يطلبون ما يحتاجون إليه، والموجهات
# تمرر ذلك. بالإضافة إلى ذلك، يكشف الموجه عما يحتاجه بنفسه حتى
# يمكن استخدامه داخل موجه آخر، على سبيل المثال خط أنابيب داخل كائن بحث
# الشبكة. إخراج "get_metadata_routing" الذي هو تمثيل قاموس
# لـ :class:`~utils.metadata_routing.MetadataRouter`، يتضمن شجرة كاملة
# من البيانات الوصفية المطلوبة من جميع الكائنات المضمنة وتوجيهات
# الطرق الخاصة بهم، أي أي طريقة لمُقدر فرعي يتم استخدامها في أي
# طريقة لميتا-مقدر:

print_routing(meta_est)

# %%
# كما ترى، البيانات الوصفية الوحيدة المطلوبة لطريقة "fit" هي
# "sample_weight" مع "aliased_sample_weight" كمستعار. فئة
# "MetadataRouter" تسمح لنا بسهولة إنشاء كائن التوجيه الذي من شأنه
# أن يخلق الإخراج الذي نحتاجه لـ "get_metadata_routing".
#
# لفهم كيفية عمل المستعارات في الميتا-مقدرات، تخيل الميتا-مقدر
# داخل آخر:

meta_meta_est = MetaClassifier(estimator=meta_est).fit(
    X, y, aliased_sample_weight=my_weights
)

# %%
# في المثال أعلاه، هذه هي الطريقة التي ستستدعي بها طريقة "fit"
# للمُقدرات الفرعية الخاصة بـ `meta_meta_est` طرق "fit" الخاصة بهم::
#
#     # يقوم المستخدم بتغذية `my_weights` كـ `aliased_sample_weight` إلى `meta_meta_est`:
#     meta_meta_est.fit(X, y, aliased_sample_weight=my_weights):
#         ...
#
#         # يتوقع المُقدر الفرعي الأول (`meta_est`) `aliased_sample_weight`
#         self.estimator_.fit(X, y, aliased_sample_weight=aliased_sample_weight):
#             ...
#
#             # يتوقع المُقدر الفرعي الثاني (`est`) `sample_weight`
#             self.estimator_.fit(X, y, sample_weight=aliased_sample_weight):
#                 ...

# %%
# المُقدر المستهلك والموجه
# ------------------------------------
# لمثال أكثر تعقيدًا قليلاً، ضع في اعتبارك ميتا-مقدر يقوم بتوجيه
# البيانات الوصفية إلى مُقدر أساسي كما كان من قبل، ولكنه يستخدم أيضًا
# بعض البيانات الوصفية في طرقه الخاصة. هذا الميتا-مقدر هو مستهلك
# وموجه في نفس الوقت. تنفيذ واحد هو مشابه جدًا لما كان لدينا من قبل،
# ولكن مع بعض التعديلات.


class RouterConsumerClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def get_metadata_routing(self):
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            # تعريف قيم طلب توجيه البيانات الوصفية للاستخدام في المقدر التلوي
            .add_self_request(self)
            # تعريف قيم طلب توجيه البيانات الوصفية للاستخدام في المقدر الفرعي
            .add(
                estimator=self.estimator,
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="predict", callee="predict")
                .add(caller="score", callee="score"),
            )
        )
        return router

    # نظرًا لأنه يتم استخدام `sample_weight` واستهلاكه هنا، فيجب تعريفه كـ
    # وسيطة صريحة في توقيع الأسلوب. سيتم تمرير جميع البيانات الوصفية الأخرى التي
    # يتم توجيهها فقط كـ `**fit_params`:
    def fit(self, X, y, sample_weight, **fit_params):
        if self.estimator is None:
            raise ValueError("لا يمكن أن يكون المقدر فارغًا!")

        check_metadata(self, sample_weight=sample_weight)

        # نضيف `sample_weight` إلى قاموس `fit_params`.
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        request_router = get_routing_for_object(self)
        request_router.validate_metadata(params=fit_params, method="fit")
        routed_params = request_router.route_params(params=fit_params, caller="fit")
        self.estimator_ = clone(self.estimator).fit(X, y, **routed_params.estimator.fit)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X, **predict_params):
        check_is_fitted(self)
        # كما في `fit`، نحصل على نسخة من MetadataRouter للكائن،
        request_router = get_routing_for_object(self)
        # نقوم بالتحقق من صحة البيانات الوصفية المحددة،
        request_router.validate_metadata(params=predict_params, method="predict")
        # ثم نقوم بإعداد المدخلات لأسلوب ``predict`` الأساسي.
        routed_params = request_router.route_params(
            params=predict_params, caller="predict"
        )
        return self.estimator_.predict(X, **routed_params.estimator.predict)


# %%
# الأجزاء الرئيسية التي يختلف فيها المقدر التلوي أعلاه عن المقدر التلوي السابق
# لدينا هو قبول ``sample_weight`` بشكل صريح في ``fit`` و
# تضمينه في ``fit_params``. نظرًا لأن ``sample_weight`` وسيطة
# صريحة، يمكننا التأكد من وجود ``set_fit_request(sample_weight=...)``
# لهذا الأسلوب. المقدر التلوي هو مستهلك، بالإضافة إلى كونه
# موجهًا لـ ``sample_weight``.
#
# في ``get_metadata_routing``، نضيف ``self`` إلى التوجيه باستخدام
# ``add_self_request`` للإشارة إلى أن هذا المقدر يستهلك
# ``sample_weight`` بالإضافة إلى كونه موجهًا؛ والذي يضيف أيضًا
# مفتاح ``$self_request`` إلى معلومات التوجيه كما هو موضح أدناه. الآن دعونا
# نلقي نظرة على بعض الأمثلة:

# %%
# - لم يتم طلب بيانات وصفية
meta_est = RouterConsumerClassifier(estimator=ExampleClassifier())
print_routing(meta_est)


# %%
# - ``sample_weight`` مطلوب بواسطة المقدر الفرعي
meta_est = RouterConsumerClassifier(
    estimator=ExampleClassifier().set_fit_request(sample_weight=True)
)
print_routing(meta_est)

# %%
# - ``sample_weight`` مطلوب بواسطة المقدر التلوي
meta_est = RouterConsumerClassifier(estimator=ExampleClassifier()).set_fit_request(
    sample_weight=True
)
print_routing(meta_est)

# %%
# لاحظ الفرق في تمثيلات البيانات الوصفية المطلوبة أعلاه.
#
# - يمكننا أيضًا تسمية البيانات الوصفية لتمرير قيم مختلفة إلى أساليب الملاءمة
#   للمقدر التلوي والمقدر الفرعي:

meta_est = RouterConsumerClassifier(
    estimator=ExampleClassifier().set_fit_request(sample_weight="clf_sample_weight"),
).set_fit_request(sample_weight="meta_clf_sample_weight")
print_routing(meta_est)

# %%
# ومع ذلك، فإن ``fit`` للمقدر التلوي يحتاج فقط إلى الاسم المستعار للمقدر
# الفرعي ويعالج وزن عينته كـ `sample_weight`، لأنه
# لا يتحقق من صحة بياناته الوصفية المطلوبة ولا يقوم بتوجيهها:
meta_est.fit(X, y, sample_weight=my_weights, clf_sample_weight=my_other_weights)

# %%
# - الاسم المستعار فقط على المقدر الفرعي:
#
# هذا مفيد عندما لا نريد أن يستخدم المقدر التلوي البيانات الوصفية، ولكن
# يجب على المقدر الفرعي استخدامها.
meta_est = RouterConsumerClassifier(
    estimator=ExampleClassifier().set_fit_request(sample_weight="aliased_sample_weight")
)
print_routing(meta_est)
# %%
# لا يمكن للمقدر التلوي استخدام `aliased_sample_weight`، لأنه يتوقع
# تمريره كـ `sample_weight`. سينطبق هذا حتى لو
# تم تعيين `set_fit_request(sample_weight=True)` عليه.


# %%
# خط أنابيب بسيط
# ---------------
# حالة استخدام أكثر تعقيدًا قليلاً هي مقدر تلوي يشبه
# :class:`~pipeline.Pipeline`. هنا مقدر تلوي، يقبل
# محولًا ومصنفًا. عند استدعاء أسلوب `fit` الخاص به، فإنه يطبق
# `fit` و `transform` للمحول قبل تشغيل المصنف على
# البيانات المحولة. عند `predict`، فإنه يطبق `transform` للمحول
# قبل التنبؤ باستخدام أسلوب `predict` للمصنف على البيانات
# الجديدة المحولة.

class SimplePipeline(ClassifierMixin, BaseEstimator):
    def __init__(self, transformer, classifier):
        self.transformer = transformer
        self.classifier = classifier

    def get_metadata_routing(self):
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            # نضيف التوجيه للمحول.
            .add(
                transformer=self.transformer,
                method_mapping=MethodMapping()
                # يتم توجيه البيانات الوصفية بحيث تتبع كيفية
                # استدعاء `SimplePipeline` داخليًا لأساليب `fit` و
                # `transform` للمحول في أساليبه الخاصة (`fit` و `predict`).
                .add(caller="fit", callee="fit")
                .add(caller="fit", callee="transform")
                .add(caller="predict", callee="transform"),
            )
            # نضيف التوجيه للمصنف.
            .add(
                classifier=self.classifier,
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="predict", callee="predict"),
            )
        )
        return router

    def fit(self, X, y, **fit_params):
        routed_params = process_routing(self, "fit", **fit_params)

        self.transformer_ = clone(self.transformer).fit(
            X, y, **routed_params.transformer.fit
        )
        X_transformed = self.transformer_.transform(
            X, **routed_params.transformer.transform
        )

        self.classifier_ = clone(self.classifier).fit(
            X_transformed, y, **routed_params.classifier.fit
        )
        return self

    def predict(self, X, **predict_params):
        routed_params = process_routing(self, "predict", **predict_params)

        X_transformed = self.transformer_.transform(
            X, **routed_params.transformer.transform
        )
        return self.classifier_.predict(
            X_transformed, **routed_params.classifier.predict
        )


# %%
# لاحظ استخدام :class:`~utils.metadata_routing.MethodMapping` لـ
# إعلان الأساليب التي يستخدمها المقدر التابع (المستدعى) في أي
# أساليب للمقدر التلوي (المستدعي). كما ترى، يستخدم `SimplePipeline`
# أساليب ``transform`` و ``fit`` للمحول في ``fit``، وأسلوب
# ``transform`` الخاص به في ``predict``، وهذا ما تراه مطبقًا في
# بنية توجيه فئة خط الأنابيب.
#
# هناك اختلاف آخر في المثال أعلاه مع الأمثلة السابقة وهو استخدام
# :func:`~utils.metadata_routing.process_routing`، والذي يعالج معلمات
# الإدخال، ويقوم بالتحقق المطلوب، ويعيد `routed_params`
# الذي أنشأناه في الأمثلة السابقة. هذا يقلل من التعليمات البرمجية المعيارية
# التي يحتاج المطور لكتابتها في كل أسلوب مقدر تلوي. يوصى بشدة
# للمطورين باستخدام هذه الوظيفة ما لم يكن هناك سبب وجيه
# ضدها.
#
# لاختبار خط الأنابيب أعلاه، دعنا نضيف محولًا نموذجيًا.


class ExampleTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        check_metadata(self, sample_weight=sample_weight)
        return self

    def transform(self, X, groups=None):
        check_metadata(self, groups=groups)
        return X

    def fit_transform(self, X, y, sample_weight=None, groups=None):
        return self.fit(X, y, sample_weight).transform(X, groups)


# %%
# لاحظ أنه في المثال أعلاه، قمنا بتطبيق ``fit_transform`` الذي
# يستدعي ``fit`` و ``transform`` مع البيانات الوصفية المناسبة. هذا مطلوب فقط
# إذا كان ``transform`` يقبل البيانات الوصفية، لأن تطبيق ``fit_transform`` الافتراضي
# في :class:`~base.TransformerMixin` لا يمرر البيانات الوصفية إلى
# ``transform``.
#
# الآن يمكننا اختبار خط الأنابيب الخاص بنا، ومعرفة ما إذا كانت البيانات الوصفية
# يتم تمريرها بشكل صحيح. يستخدم هذا المثال `SimplePipeline` الخاص بنا، و
# `ExampleTransformer` الخاص بنا، و `RouterConsumerClassifier` الخاص بنا
# الذي يستخدم `ExampleClassifier` الخاص بنا.

pipe = SimplePipeline(
    transformer=ExampleTransformer()
    # قمنا بتعيين ملاءمة المحول لتلقي sample_weight
    .set_fit_request(sample_weight=True)
    # قمنا بتعيين تحويل المحول لتلقي المجموعات
    .set_transform_request(groups=True),
    classifier=RouterConsumerClassifier(
        estimator=ExampleClassifier()
        # نريد أن يتلقى هذا المقدر الفرعي sample_weight في الملاءمة
        .set_fit_request(sample_weight=True)
        # ولكن ليس المجموعات في التنبؤ
        .set_predict_request(groups=False),
    )
    # ونريد أن يتلقى المقدر التلوي sample_weight أيضًا
    .set_fit_request(sample_weight=True),
)
pipe.fit(X, y, sample_weight=my_weights, groups=my_groups).predict(
    X[:3], groups=my_groups
)

# %%
# إهمال / تغيير القيمة الافتراضية
# ----------------------------------
# في هذا القسم، نظهر كيفية تعامل المرء مع الحالة التي يصبح فيها الموجه
# أيضًا مستهلكًا، خاصةً عندما يستهلك نفس البيانات الوصفية مثل المقدر
# الفرعي الخاص به، أو يبدأ المستهلك في استهلاك بيانات وصفية لم تكن موجودة
# في إصدار أقدم. في هذه الحالة، يجب إصدار تحذير لفترة من الوقت،
# لإعلام المستخدمين بأن السلوك قد تغير عن الإصدارات السابقة.


class MetaRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        routed_params = process_routing(self, "fit", **fit_params)
        self.estimator_ = clone(self.estimator).fit(X, y, **routed_params.estimator.fit)

    def get_metadata_routing(self):
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.estimator,
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
        return router


# %%
# كما هو موضح أعلاه، هذا استخدام صالح إذا لم يكن من المفترض تمرير `my_weights`
# كـ `sample_weight` إلى `MetaRegressor`:

reg = MetaRegressor(estimator=LinearRegression().set_fit_request(sample_weight=True))
reg.fit(X, y, sample_weight=my_weights)


# %%
# الآن تخيل أننا نطور ``MetaRegressor`` بشكل أكبر وأنه *يستهلك* الآن
# أيضًا ``sample_weight``:


class WeightedMetaRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    # إظهار تحذير لتذكير المستخدم بتعيين القيمة صراحةً باستخدام
    # `.set_{method}_request(sample_weight={boolean})`
    __metadata_request__fit = {"sample_weight": metadata_routing.WARN}

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None, **fit_params):
        routed_params = process_routing(
            self, "fit", sample_weight=sample_weight, **fit_params
        )
        check_metadata(self, sample_weight=sample_weight)
        self.estimator_ = clone(self.estimator).fit(X, y, **routed_params.estimator.fit)

    def get_metadata_routing(self):
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(
                estimator=self.estimator,
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),
            )
        )
        return router


# %%
# التطبيق أعلاه هو نفسه تقريبًا ``MetaRegressor``، و
# بسبب قيمة الطلب الافتراضية المحددة في ``__metadata_request__fit``
# يتم إصدار تحذير عند الملاءمة.


with warnings.catch_warnings(record=True) as record:
    WeightedMetaRegressor(
        estimator=LinearRegression().set_fit_request(sample_weight=False)
    ).fit(X, y, sample_weight=my_weights)
for w in record:
    print(w.message)


# %%
# عندما يستهلك مقدر بيانات وصفية لم يستهلكها من قبل،
# يمكن استخدام النمط التالي لتحذير المستخدمين بشأنها.



class ExampleRegressor(RegressorMixin, BaseEstimator):
    __metadata_request__fit = {"sample_weight": metadata_routing.WARN}

    def fit(self, X, y, sample_weight=None):
        check_metadata(self, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return np.zeros(shape=(len(X)))


with warnings.catch_warnings(record=True) as record:
    MetaRegressor(estimator=ExampleRegressor()).fit(X, y, sample_weight=my_weights)
for w in record:
    print(w.message)

# %%
# في النهاية نقوم بتعطيل علامة التكوين لتوجيه البيانات الوصفية:

set_config(enable_metadata_routing=False)

# %%
# تطوير الطرف الثالث وتبعية scikit-learn
# ---------------------------------------------------
#
# كما هو موضح أعلاه، يتم توصيل المعلومات بين الفئات باستخدام
# :class:`~utils.metadata_routing.MetadataRequest` و
# :class:`~utils.metadata_routing.MetadataRouter`. لا يُنصح بشدة،
# ولكن من الممكن بيع الأدوات المتعلقة بتوجيه البيانات الوصفية إذا كنت تريد
# بشكل صارم الحصول على مقدر متوافق مع scikit-learn، دون الاعتماد على
# حزمة scikit-learn. إذا تم استيفاء جميع الشروط التالية، فلن تحتاج
# إلى تعديل التعليمات البرمجية الخاصة بك على الإطلاق:
#
# - يرث المقدر الخاص بك من :class:`~base.BaseEstimator`
# - المعلمات التي تستهلكها أساليب المقدر الخاص بك، على سبيل المثال ``fit``،
#   مُعرّفة صراحةً في توقيع الأسلوب، على عكس كونها
#   ``*args`` أو ``*kwargs``.
# - لا يقوم المقدر الخاص بك بتوجيه أي بيانات وصفية إلى الكائنات الأساسية، أي
#   أنه ليس *موجهًا*.
