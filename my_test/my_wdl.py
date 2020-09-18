import tensorflow as tf
from deepctr.estimator.inputs import input_fn_tfrecord
from deepctr.estimator.models import WDLEstimator

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

train_model_input = input_fn_tfrecord('./data/criteo_sample.tr.tfrecords', feature_description, 'label', batch_size=256, num_epochs=1, shuffle_factor=10)
test_model_input = input_fn_tfrecord('./data/criteo_sample.te.tfrecords', feature_description, 'label', batch_size=2 ** 14, num_epochs=1, shuffle_factor=0)

# model = WDLEstimator(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=[4, 4], dnn_dropout=0.5)
model = WDLEstimator(linear_feature_columns, dnn_feature_columns)

model.train(train_model_input)
eval_result = model.evaluate(test_model_input)

print(eval_result)