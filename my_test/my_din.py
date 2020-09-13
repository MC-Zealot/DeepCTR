import numpy as np

from deepctr.models import DIN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat,get_feature_names
from tensorflow import keras
import pandas as pd
import tensorflow as tf
#文中说，用item


def get_xy_fd():

    feature_columns = [SparseFeat('user',3,         embedding_dim=10),
                       SparseFeat('gender', 2,      embedding_dim=4),
                       SparseFeat('item_id', 3 + 1, embedding_dim=8),
                       SparseFeat('cate_id', 2 + 1, embedding_dim=4),
                       DenseFeat('pay_score', 1)]
    feature_columns += [VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=3 + 1,embedding_dim=8, embedding_name='item_id'), maxlen=4),
                        VarLenSparseFeat(SparseFeat('hist_cate_id', 2 + 1,                embedding_dim=4, embedding_name='cate_id'), maxlen=4)]

    behavior_feature_list = ["item_id", "cate_id"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    cate_id = np.array([1, 2, 2])  # 0 is mask value
    pay_score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [3, 2, 1, 0], [1, 2, 0, 0]])
    hist_cate_id = np.array([[1, 2, 2, 0], [2, 2, 1, 0], [1, 2, 0, 0]])

    feature_dict = {'user': uid, 'gender': ugender, 'item_id': iid, 'cate_id': cate_id,
                    'hist_item_id': hist_iid, 'hist_cate_id': hist_cate_id, 'pay_score': pay_score}
    x = {name:feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1])
    return x, y, feature_columns, behavior_feature_list

def example_din():
    """
    1. 生成训练数据为txt格式的，逗号分割字段
    2. 转换成tfrecord
    3. 读取数据，区分dense, sparse, VarLenSparse, 用户行为序列特征
    4. 分别喂到模型中，看看会怎么样
    :return:
    """
    # x, y, feature_columns, behavior_feature_list = get_xy_fd() #说一下哪几列是当前的item需要和历史的行为进行attention的。所以之后就可以尝试，还是像之前一样读数据，然后只是把需要attention的列名单拿出来，放到list中就可以了
    x, y, feature_columns, behavior_feature_list = get_xy_from_txt() #说一下哪几列是当前的item需要和历史的行为进行attention的。所以之后就可以尝试，还是像之前一样读数据，然后只是把需要attention的列名单拿出来，放到list中就可以了
    # dataset = tf.data.Dataset.from_tensor_slices((x.values, y.values))

    model = DIN(feature_columns, behavior_feature_list)
    model.compile('adam', keras.losses.binary_crossentropy, metrics=[keras.metrics.AUC(), keras.metrics.categorical_accuracy])
    history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)
    # history = model.fit(dataset, verbose=1, epochs=10, validation_data=(x,y))
    # history = model.fit(dataset, verbose=1, epochs=10, validation_split=0.5)
    print("history: ", history)

def get_xy_from_txt(file_path="data/movielens_sample_din.txt"):
    feature_columns = [SparseFeat('user', 3, embedding_dim=10),
                       SparseFeat('gender', 2, embedding_dim=4),
                       SparseFeat('item_id', 3 + 1, embedding_dim=8),
                       SparseFeat('cate_id', 2 + 1, embedding_dim=4),
                       DenseFeat('pay_score', 1)]
    feature_columns += [
                    VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'), maxlen=4),
                    VarLenSparseFeat(SparseFeat('hist_cate_id', 2 + 1, embedding_dim=4, embedding_name='cate_id'), maxlen=4)]

    behavior_feature_list = ["item_id", "cate_id"]
    # head = ['label', 'user', 'gender', 'item_id', 'cate_id', 'hist_item_id', 'hist_cate_id', 'pay_score']

    data = pd.read_csv(file_path, delimiter=',')
    def to_int_array(x):
        ret = []
        a = x.split('|')
        for str in a:
            ret.append(int(str))
        # return np.array(ret)
        return np.array(ret)
    data['hist_item_id'] = data['hist_item_id'].apply(to_int_array)
    data['hist_cate_id'] = data['hist_cate_id'].apply(to_int_array)

    uid = np.array(data['user'])

    ugender = np.array(data['gender'])
    iid = np.array(data['item_id'])  # 0 is mask value
    cate_id = np.array(data['cate_id'])  # 0 is mask value
    pay_score = np.array(data['pay_score'])
    hist_iid = np.array(data['hist_item_id'])
    hist_cate_id = np.array(data['hist_cate_id'])
    print("uid: ", type(uid), uid)
    print("hist_cate_id: ", type(hist_cate_id), hist_cate_id)
    feature_dict = {'user': uid, 'gender': ugender, 'item_id': iid, 'cate_id': cate_id,
                    'hist_item_id': hist_iid, 'hist_cate_id': hist_cate_id, 'pay_score': pay_score}
    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array(data.pop('label'))

    return x, y, feature_columns,behavior_feature_list



if __name__ == "__main__":
    example_din()
