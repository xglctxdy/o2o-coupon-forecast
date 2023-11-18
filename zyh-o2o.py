import pandas as pd
import xgboost as xgb
import warnings
import numpy as np

warnings.filterwarnings('ignore')  # 不显示警告


def pretreatment(dataset):
    data = dataset.copy()
    data['is_manjian'] = data['Discount_rate'].apply(lambda x: 1 if ':' in str(x) else 0)
    data['discount_rate'] = data['Discount_rate'].apply(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
    data['min_cost_of_manjian'] = data['Discount_rate'].apply(
        lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[0]))
    data['Distance'].fillna(-1, inplace=True)
    data['null_distance'] = data['Distance'].apply(lambda x: 1 if x == -1 else 0)
    data['date_received'] = pd.to_datetime(data['Date_received'], format='%Y%m%d')
    if 'Date' in data.columns.tolist():
        data['date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    return data


def split(tr, te):
    train_history_field = tr[
        tr['date_received'].isin(pd.date_range('2016/1/1', periods=30))]
    train_label_field = tr[
        tr['date_received'].isin(pd.date_range('2016/4/1', periods=30))]
    test_history_field = tr[
        tr['date_received'].isin(pd.date_range('2016/5/1', periods=30))]
    test_label_field = te.copy()
    return train_history_field, train_label_field, test_history_field, test_label_field


def get_labels(dataset):
    data = dataset.copy()
    data['label'] = ((data['date'] - data['date_received']).dt.total_seconds() / (60 * 60 * 24) <= 15).astype(int)
    return data


def get_feature(label_field, history_field):
    data = label_field.copy()
    history = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(lambda x: int(x) if x == x else 0)
    data['Date_received'] = data['Date_received'].map(lambda x: int(x) if x == x else 0)
    data['cnt'] = 1
    history['cnt'] = 1
    feature = data.copy()
    t = feature[['User_id', 'cnt']]  # dataframe类型，不加[]为series
    center = pd.pivot_table(t, index='User_id', values='cnt', aggfunc=len)
    center = pd.DataFrame(center).rename(columns={'cnt': 'consume'}).reset_index()
    feature = pd.merge(feature, center, on='User_id', how='left')
    t = history[history.Date.notnull() & history.Date_received.notnull()][['User_id', 'Distance']]
    center = pd.pivot_table(t, index='User_id', values='Distance', aggfunc=np.min)
    center = pd.DataFrame(center).rename(columns={'Distance': 'min_distance'}).reset_index()
    feature = pd.merge(feature, center, on='User_id', how='left')
    t = history[history.Date.notnull() & history.Date_received.notnull()][['User_id', 'Distance']]
    center = pd.pivot_table(t, index='User_id', values='Distance', aggfunc=np.max)
    center = pd.DataFrame(center).rename(columns={'Distance': 'max_distance'}).reset_index()
    feature = pd.merge(feature, center, on='User_id', how='left')
    t = history[history.Date.notnull() & history.Date_received.notnull()][['User_id', 'Distance']]
    center = pd.pivot_table(t, index='User_id', values='Distance', aggfunc=np.mean)
    center = pd.DataFrame(center).rename(columns={'Distance': 'mean_distance'}).reset_index()
    feature = pd.merge(feature, center, on='User_id', how='left')
    t = history[history['Date'].notnull() & history['Coupon_id'].notnull()][['User_id', 'discount_rate']]
    t = t.groupby(['User_id'])['discount_rate'].mean().reset_index(name='receive_buy_mean_rate')
    feature = pd.merge(feature, t, on='User_id', how='left')
    t = feature[feature['Coupon_id'].notnull()][['User_id']]
    t['cnt'] = 1
    t = t.groupby(['User_id'])['cnt'].sum().reset_index(name='Coupon_amount')
    feature = pd.merge(feature, t, on='User_id', how='left')
    t = feature[feature['Date_received'].notnull()][['User_id']]
    t['cnt'] = 1
    t = t.groupby(['User_id'])['cnt'].sum().reset_index(name='User_received_total_coupon')
    feature = pd.merge(feature, t, on='User_id', how='left')
    t = feature[feature['Date_received'].notnull() & feature['is_manjian'] == 1][['User_id']]
    t['cnt'] = 1
    t = t.groupby(['User_id'])['cnt'].sum().reset_index(name='manjian_coupon')
    feature = pd.merge(feature, t, on='User_id', how='left')
    t = feature[feature['Date_received'].notnull()][['User_id', 'Date_received']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Date_received'])['cnt'].sum().reset_index(name='User_Date_received')
    feature = pd.merge(feature, t, on=['User_id', 'Date_received'], how='left')
    feature['User_manjian_rate'] = list(
        map(lambda x, y: x / y if y != 0 else 0, feature['manjian_coupon'], feature['User_received_total_coupon']))
    t = feature[['User_id', 'Coupon_id', 'Date_received']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Coupon_id', 'Date_received'])['cnt'].sum().reset_index(
        name='User_Date_received_special_Coupon')
    feature = pd.merge(feature, t, on=['User_id', 'Coupon_id', 'Date_received'], how='left')
    t = feature[['User_id', 'Coupon_id', 'Date_received']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Coupon_id', 'Date_received'])['cnt'].agg(lambda x: 1 if sum(x) > 1 else 0).reset_index(
        name='User_if_same_Date_received_special_Coupon')
    feature = pd.merge(feature, t, on=['User_id', 'Coupon_id', 'Date_received'], how='left')
    a = history[(history['Date'].notnull()) & (history['Coupon_id'].notnull())]
    t = a.groupby(['User_id'])['cnt'].sum().reset_index(name='get_consume')
    feature = pd.merge(feature, t, on=['User_id'], how='left')
    a = history[history['Date'].notnull() & history['Coupon_id'].notnull()][['User_id', 'date_received', 'date']]
    a['time'] = list(map(lambda x, y: (x - y).total_seconds() / (3600 * 24), a['date_received'], a['date']))
    t = a.groupby('User_id')['time'].mean().reset_index(name='receive_use_mean')
    feature = pd.merge(feature, t, on='User_id', how='left')
    t = a.groupby('User_id')['time'].max().reset_index(name='receive_use_max')
    feature = pd.merge(feature, t, on='User_id', how='left')
    t = a.groupby('User_id')['time'].min().reset_index(name='receive_use_min')
    feature = pd.merge(feature, t, on='User_id', how='left')
    t = history[history.Date.notnull()][['Merchant_id', 'cnt']]
    center = pd.pivot_table(t, index='Merchant_id', values='cnt', aggfunc=len)
    center = pd.DataFrame(center).rename(columns={'cnt': 'cunsume_cnt'}).reset_index()
    feature = pd.merge(feature, center, on='Merchant_id', how='left')
    t = history[history['Coupon_id'].notnull() & history['Date'] != np.nan][['Merchant_id']]
    t['cnt'] = 1
    t = t.groupby(['Merchant_id'])['cnt'].sum().reset_index(name='Merchant_use_coupon_amount')
    feature = pd.merge(feature, t, on='Merchant_id', how='left')
    t = history[history.Date.notnull() & history.Date_received.notnull()][['Merchant_id', 'cnt']]
    center = pd.pivot_table(t, index='Merchant_id', values='cnt', aggfunc=len)
    center = pd.DataFrame(center).rename(columns={'cnt': 'receive_consume_cnt'}).reset_index()
    feature = pd.merge(feature, center, on='Merchant_id', how='left')
    t = history[history.Date_received.notnull()][['Merchant_id', 'cnt']]
    center = pd.pivot_table(t, index='Merchant_id', values='cnt', aggfunc=len)
    center = pd.DataFrame(center).rename(columns={'cnt': 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, center, on='Merchant_id', how='left')
    feature['cut_price'] = list(
        map(lambda x: float(str(x).split(':')[1]) if ':' in str(x) else 0, feature['Discount_rate']))
    t = history[history.Date.notnull()][['Coupon_id', 'cnt']]
    center = pd.pivot_table(t, index='Coupon_id', values='cnt', aggfunc=len)
    center = pd.DataFrame(center).rename(columns={'cnt': 'cunsume_cnt'}).reset_index()
    feature = pd.merge(feature, center, on='Coupon_id', how='left')
    feature['receive_rate'] = feature['get_consume'] / feature['Coupon_amount']
    t = feature[feature['Coupon_id'].notnull()][['User_id', 'Merchant_id']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Merchant_id'])['cnt'].sum().reset_index(name='receive_coupon_in_different_merchant')
    t.drop_duplicates(inplace=True)
    feature = pd.merge(feature, t, on=['User_id', 'Merchant_id'], how='left')
    t = history[history['Coupon_id'].notnull() & history['Date'].notnull()][['User_id', 'Merchant_id']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Merchant_id'])['cnt'].sum().reset_index(name='receive_use__coupon_in_different_merchant')
    t.drop_duplicates(inplace=True)
    feature = pd.merge(feature, t, on=['User_id', 'Merchant_id'], how='left')
    t = history[history.Date_received.notnull()]
    center = pd.pivot_table(t, index=['User_id', 'Merchant_id'], values='cnt', aggfunc=len)
    center = pd.DataFrame(center).rename(columns={'cnt': 'receive1_cnt'}).reset_index()
    feature = pd.merge(feature, center, on=['User_id', 'Merchant_id'], how='left')
    t = feature[feature['Date_received'].notnull() & feature['is_manjian'] == 1][['User_id', 'Merchant_id']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Merchant_id'])['cnt'].sum().reset_index(name='User_Merchant_manjian_coupon')
    feature = pd.merge(feature, t, on=['User_id', 'Merchant_id'], how='left')
    feature['User_Merchant_manjian_rate'] = list(
        map(lambda x, y: x / y if y != 0 else 0, feature['User_Merchant_manjian_coupon'],
            feature['receive1_cnt']))
    t = feature[['User_id', 'Distance']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Distance'])['cnt'].sum().reset_index(name='User_received_Distance')
    feature = pd.merge(feature, t, on=['User_id', 'Distance'], how='left')
    t = history[history.Date_received.notnull()][['User_id', 'cnt']]  # dataframe类型，不加[]为series
    center = pd.pivot_table(t, index='User_id', values='cnt', aggfunc=len)
    center = pd.DataFrame(center).rename(columns={'cnt': 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, center, on='User_id', how='left')
    t = feature[['User_id', 'Date_received']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Date_received'])['cnt'].sum().reset_index(name='User_Date_received')
    feature = pd.merge(feature, t, on=['User_id', 'Date_received'], how='left')
    t = history[history.Date.notnull() & history.Date_received.notnull()][['User_id', 'discount_rate']]
    center = pd.pivot_table(t, index='User_id', values='discount_rate', aggfunc=np.min)
    center = pd.DataFrame(center).rename(columns={'discount_rate': 'min_discount_rate'}).reset_index()
    feature = pd.merge(feature, center, on='User_id', how='left')
    t = history[history.Date.notnull() & history.Date_received.notnull()][['User_id', 'discount_rate']]
    center = pd.pivot_table(t, index='User_id', values='discount_rate', aggfunc=np.max)
    center = pd.DataFrame(center).rename(columns={'discount_rate': 'max_discount_rate'}).reset_index()
    feature = pd.merge(feature, center, on='User_id', how='left')
    a = feature[feature['Distance'] != -1]
    b = a.groupby(['User_id'])['Distance'].min().reset_index(name='min_distance')
    feature = pd.merge(feature, b, on=['User_id'], how='left')

    # 顾客消费最远距离
    t = history[history.Date.notnull() & history.Date_received.notnull()][['User_id', 'Distance']]
    t['Distance'] = t['Distance'].map(lambda x: 0 if x == -1 else int(x))
    center = pd.pivot_table(t, index='User_id', values='Distance', aggfunc=np.max)
    center = pd.DataFrame(center).rename(columns={'Distance': 'max_distance'}).reset_index()
    feature = pd.merge(feature, center, on='User_id', how='left')
    # 商家被领取优惠券数量
    t = history[history.Date_received.notnull()][['Merchant_id', 'cnt']]
    center = pd.pivot_table(t, index='Merchant_id', values='cnt', aggfunc=len)
    center = pd.DataFrame(center).rename(columns={'cnt': 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, center, on='Merchant_id', how='left')
    t = feature[feature['Date_received'].notnull()][['Merchant_id', 'Coupon_id']]
    t['cnt'] = 1
    t = t.groupby(['Merchant_id', 'Coupon_id'])['cnt'].sum().reset_index(name='Merchant_received_special_coupon')
    feature = pd.merge(feature, t, on=['Merchant_id', 'Coupon_id'], how='left')
    t = feature[feature['Discount_rate'].notnull()][['discount_rate']]
    t['cnt'] = 1
    t = t.groupby(['discount_rate'])['cnt'].sum().reset_index(name='different_rate_coupon_amount')
    feature = pd.merge(feature, t, on='discount_rate', how='left')
    feature.drop(['cnt'], axis=1, inplace=True)
    feature['week'] = feature['date_received'].map(lambda x: x.isoweekday())
    feature['is_weekend'] = feature['week'].map(lambda x: 1 if x == 5 or x == 6 else 0)
    feature = pd.concat([feature, pd.get_dummies(feature['week'], prefix='week')], axis=1)
    feature.index = range(len(feature))
    # 将Distance中的nan填充为-1
    feature['Distance'].fillna(-1, inplace=True)
    # 判断距离是否为空
    feature['isnull_Distance'] = list(map(lambda x: 1 if x == -1 else 0, feature['Distance']))
    return feature


def create_dataset(history_field, label_field):
    def get_week_feature(label_field):
        # 源数据
        data = label_field.copy()
        data['Coupon_id'] = data['Coupon_id'].map(int)
        data['Date_received'] = data['Date_received'].map(int)
        # 返回的特征数据集
        feature = data.copy()
        feature.index = range(len(feature))
        # 返回
        return feature

    week_feat = get_week_feature(label_field)
    simple_feat = get_feature(label_field, history_field)
    share_characters = list(set(simple_feat.columns.tolist()) & set(week_feat.columns.tolist()))
    dataset = pd.concat([week_feat, simple_feat.drop(share_characters, axis=1)], axis=1)
    if 'Date' in dataset.columns.tolist():
        dataset.drop(['Merchant_id', 'Discount_rate', 'Date', 'date_received', 'date'], axis=1, inplace=True)
        label = dataset['label'].tolist()
        dataset.drop(['label'], axis=1, inplace=True)
        dataset['label'] = label
    else:
        dataset.drop(['Merchant_id', 'Discount_rate', 'date_received'], axis=1, inplace=True)
    dataset['User_id'] = dataset['User_id'].map(int)
    dataset['Coupon_id'] = dataset['Coupon_id'].map(int)
    dataset['Date_received'] = dataset['Date_received'].map(int)
    dataset['Distance'] = dataset['Distance'].map(int)
    if 'label' in dataset.columns.tolist():
        dataset['label'] = dataset['label'].map(int)
    dataset.drop_duplicates(keep='first', inplace=True)
    dataset.index = range(len(dataset))
    return dataset


def xgb_mo(train, test):
    # xgb参数
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'silent': 1,
              'eta': 0.03,
              'max_depth': 6,
              'min_child_weight': 1,
              'gamma': 0,
              'lambda': 1,
              'colsample_bytree': 1,
              'subsample': 1,
              'scale_pos_weight': 1}
    dtrain = xgb.DMatrix(train.drop(['label'], axis=1), label=train['label'])
    dtest = xgb.DMatrix(test)
    watchlist = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, num_boost_round=5, evals=watchlist)
    # 预测
    predict = model.predict(dtest)
    # 处理结果
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)
    # 特征重要性
    feat_importance = pd.DataFrame(columns=['feature_name', 'importance'])
    feat_importance['feature_name'] = model.get_score().keys()
    feat_importance['importance'] = model.get_score().values()
    feat_importance.sort_values(['importance'], ascending=False, inplace=True)
    # 返回
    return result, feat_importance


def merge(history_field, label_field):
    temp = create_dataset(history_field, label_field)
    return temp


if __name__ == '__main__':
    tr = pd.read_csv(r'ccf_offline_stage1_train.csv')
    te = pd.read_csv(r'ccf_offline_stage1_test_revised.csv')
    tr = pretreatment(tr)
    te = pretreatment(te)
    tr = get_labels(tr)
    train1, train2, test1, test2 = split(tr, te)
    train = merge(train1, train2)
    test = merge(test1, test2)
    result, feat_importance = xgb_mo(train, test)
    result.to_csv(r'result.csv', index=False, header=None)
