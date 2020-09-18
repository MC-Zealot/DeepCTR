import tensorflow as tf
from deepctr.estimator.inputs import input_fn_tfrecord
from deepctr.estimator.models import WDLEstimator
import functools

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

dnn_feature_columns = []
linear_feature_columns = []

for i, feat in enumerate(sparse_features):
    dnn_feature_columns.append(tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_identity(feat, 1000), 4))
    linear_feature_columns.append(tf.feature_column.categorical_column_with_identity(feat, 1000))
for feat in dense_features:
    dnn_feature_columns.append(tf.feature_column.numeric_column(feat))
    linear_feature_columns.append(tf.feature_column.numeric_column(feat))

print(dnn_feature_columns)

feature_description = {k: tf.io.FixedLenFeature(dtype=tf.int64, shape=1) for k in sparse_features}
feature_description.update(
    {k: tf.io.FixedLenFeature(dtype=tf.float32, shape=1) for k in dense_features}
)
feature_description['label'] = tf.io.FixedLenFeature(dtype=tf.float32, shape=1)

train_model_input = input_fn_tfrecord('./criteo_sample.tr.tfrecords', feature_description, 'label', batch_size=256, num_epochs=1, shuffle_factor=10)
test_model_input = input_fn_tfrecord('./criteo_sample.te.tfrecords', feature_description, 'label', batch_size=2 ** 14, num_epochs=1, shuffle_factor=0)

estimator = WDLEstimator(linear_feature_columns, dnn_feature_columns)

train_input_fn = functools.partial(
    input_fn_tfrecord,
    './data/criteo_sample.tr.tfrecords',
    256,
    1)

train_spec = tf.estimator.TrainSpec(
    input_fn=train_input_fn, max_steps=10,
    hooks=[])

eval_input_fn = functools.partial(
    input_fn_tfrecord,
    './data/criteo_sample.te.tfrecords',
    256,
    1)

eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=10,
                                  start_delay_secs=10,
                                  throttle_secs=60, hooks=[])
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
