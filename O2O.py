# -*- coding: utf-8 -*-
import os
import pandas as pd
import xgboost as xgb
import warnings
import numpy as np

warnings.filterwarnings('ignore')  # 不显示警告


def pre_do_data(dataset):

    # 源数据
    data = dataset.copy()
    # 折扣率处理
    data['is_manjian'] = data['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)  # Discount_rate是否为满减
    data['discount_rate'] = data['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))  # 满减全部转换为折扣率
    data['min_cost_of_manjian'] = data['Discount_rate'].map(
        lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[0]))  # 满减的最低消费
    # 距离处理
    data['Distance'].fillna(-1, inplace=True)  # 空距离填充为-1
    data['null_distance'] = data['Distance'].map(lambda x: 1 if x == -1 else 0)
    # 时间处理
    data['date_received'] = pd.to_datetime(data['Date_received'], format='%Y%m%d')
    if 'Date' in data.columns.tolist():  # off_train
        data['date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    # 返回
    return data

def deal(dataset):
    # 源数据
    data = dataset.copy()
    # 打标:领券后15天内消费为1,否则为0
    data['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0,
                             data['date'],
                             data['date_received']))
    # 返回
    return data

def get_label_field_feature(label_field, history_field):
    data = label_field.copy()
    history = history_field.copy()  # 有'Date'
    data['Coupon_id'] = data['Coupon_id'].map(lambda x: int(x) if x == x else 0)
    data['Date_received'] = data['Date_received'].map(lambda x: int(x) if x == x else 0)
    data['cnt'] = 1  # 辅助列
    history['cnt'] = 1  # 辅助列
    feature = data.copy()

    # 顾客特征
    # 顾客购买总次数
    t = feature[['User_id', 'cnt']]  # dataframe类型，不加[]为series
    center = pd.pivot_table(t, index='User_id', values='cnt', aggfunc=len)
    center = pd.DataFrame(center).rename(columns={'cnt': 'consume'}).reset_index()
    feature = pd.merge(feature, center, on='User_id', how='left')

    # 顾客是否第一次领券(热启动)
    t = feature.groupby(['User_id'])['date_received'].min().reset_index(name='isFirst_received')
    feature = pd.merge(feature, t, on='User_id', how='left')
    feature['if_first_time'] = list(
        map(lambda x, y: 1 if x == y else 0, feature['date_received'], feature['isFirst_received']))
    feature.drop('isFirst_received', axis=1, inplace=True)

    # 顾客是否最后一次领券
    t = feature.groupby(['User_id'])['date_received'].max().reset_index(name='islast_received')
    feature = pd.merge(feature, t, on='User_id', how='left')
    feature['if_last_time'] = list(
        map(lambda x, y: 1 if x == y else 0, feature['date_received'], feature['islast_received']))
    feature.drop('islast_received', axis=1, inplace=True)

    #

    # 顾客领取并消费优惠券的最小距离
    t = history[history.Date.notnull() & history.Date_received.notnull()][['User_id', 'Distance']]
    center = pd.pivot_table(t, index='User_id', values='Distance', aggfunc=np.min)
    center = pd.DataFrame(center).rename(columns={'Distance': 'min_distance'}).reset_index()
    feature = pd.merge(feature, center, on='User_id', how='left')

    # 顾客领取并消费优惠券的最大距离
    t = history[history.Date.notnull() & history.Date_received.notnull()][['User_id', 'Distance']]
    center = pd.pivot_table(t, index='User_id', values='Distance', aggfunc=np.max)
    center = pd.DataFrame(center).rename(columns={'Distance': 'max_distance'}).reset_index()
    feature = pd.merge(feature, center, on='User_id', how='left')

    # 顾客领取并消费优惠券的平均距离
    t = history[history.Date.notnull() & history.Date_received.notnull()][['User_id', 'Distance']]
    center = pd.pivot_table(t, index='User_id', values='Distance', aggfunc=np.mean)
    center = pd.DataFrame(center).rename(columns={'Distance': 'mean_distance'}).reset_index()
    feature = pd.merge(feature, center, on='User_id', how='left')

    # 顾客领取优惠券进行购买的中位距离
    t = history[(history['Date'].notnull()) & (history['Coupon_id'].notnull())][['User_id', 'Distance']]
    t = t.groupby(['User_id'])['Distance'].median().reset_index(name='receive_buy_median_distance')
    feature = pd.merge(feature, t, on='User_id', how='left')

    # 顾客领取并消费优惠券的平均折扣率
    t = history[history['Date'].notnull() & history['Coupon_id'].notnull()][['User_id', 'discount_rate']]
    t = t.groupby(['User_id'])['discount_rate'].mean().reset_index(name='receive_buy_mean_rate')
    feature = pd.merge(feature, t, on='User_id', how='left')
    # 顾客领取特定发优惠券的数量
    t = feature[feature['Coupon_id'].notnull()][['User_id']]
    t['cnt'] = 1
    t = t.groupby(['User_id'])['cnt'].sum().reset_index(name='Coupon_amount')
    feature = pd.merge(feature, t, on='User_id', how='left')

    # 用户领取优惠券的数量
    t = feature[feature['Date_received'].notnull()][['User_id']]
    t['cnt'] = 1
    t = t.groupby(['User_id'])['cnt'].sum().reset_index(name='User_received_total_coupon')
    feature = pd.merge(feature, t, on='User_id', how='left')

    # 顾客领取满减优惠券数量
    t = feature[feature['Date_received'].notnull() & feature['is_manjian'] == 1][['User_id']]
    t['cnt'] = 1
    t = t.groupby(['User_id'])['cnt'].sum().reset_index(name='manjian_coupon')
    feature = pd.merge(feature, t, on='User_id', how='left')

    # 顾客当天领券数
    t = feature[feature['Date_received'].notnull()][['User_id', 'Date_received']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Date_received'])['cnt'].sum().reset_index(name='User_Date_received')
    feature = pd.merge(feature, t, on=['User_id', 'Date_received'], how='left')

    # 满减优惠券的占比
    feature['User_manjian_rate'] = list(
        map(lambda x, y: x / y if y != 0 else 0, feature['manjian_coupon'], feature['User_received_total_coupon']))

    # 顾客当天领取特定优惠券数
    t = feature[['User_id', 'Coupon_id', 'Date_received']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Coupon_id', 'Date_received'])['cnt'].sum().reset_index(
        name='User_Date_received_special_Coupon')
    feature = pd.merge(feature, t, on=['User_id', 'Coupon_id', 'Date_received'], how='left')

    # 顾客是否在同一天重复领取了特定优惠券
    t = feature[['User_id', 'Coupon_id', 'Date_received']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Coupon_id', 'Date_received'])['cnt'].agg(lambda x: 1 if sum(x) > 1 else 0).reset_index(
        name='User_if_same_Date_received_special_Coupon')
    feature = pd.merge(feature, t, on=['User_id', 'Coupon_id', 'Date_received'], how='left')

    a = history[(history['Date'].notnull()) & (history['Coupon_id'].notnull())]  # 领券且消费

    # 顾客领券并消费
    t = a.groupby(['User_id'])['cnt'].sum().reset_index(name='get_consume')
    feature = pd.merge(feature, t, on=['User_id'], how='left')

    # 辅助特征
    a = history[history['Date'].notnull() & history['Coupon_id'].notnull()][['User_id', 'date_received', 'date']]
    a['time'] = list(map(lambda x, y: (x - y).total_seconds() / (3600 * 24), a['date_received'], a['date']))
    # 顾客领取优惠券到使用优惠券的平均时间间隔
    t = a.groupby('User_id')['time'].mean().reset_index(name='receive_use_mean')
    feature = pd.merge(feature, t, on='User_id', how='left')

    # 顾客领取优惠券到使用优惠券的中位时间间隔
    t = a.groupby('User_id')['time'].median().reset_index(name='receive_use_median')
    feature = pd.merge(feature, t, on='User_id', how='left')

    # 顾客领取优惠券到使用优惠券的最大时间间隔
    t = a.groupby('User_id')['time'].max().reset_index(name='receive_use_max')
    feature = pd.merge(feature, t, on='User_id', how='left')

    # 顾客领取优惠券到使用优惠券的最小时间间隔
    t = a.groupby('User_id')['time'].min().reset_index(name='receive_use_min')
    feature = pd.merge(feature, t, on='User_id', how='left')

    # 商家特征
    # 商家销售商品数量
    t = history[history.Date.notnull()][['Merchant_id', 'cnt']]
    center = pd.pivot_table(t, index='Merchant_id', values='cnt', aggfunc=len)
    center = pd.DataFrame(center).rename(columns={'cnt': 'cunsume_cnt'}).reset_index()
    feature = pd.merge(feature, center, on='Merchant_id', how='left')

    # 商家销售使用了优惠券的商品数量
    t = history[history['Coupon_id'].notnull() & history['Date'] != np.nan][['Merchant_id']]
    t['cnt'] = 1
    t = t.groupby(['Merchant_id'])['cnt'].sum().reset_index(name='Merchant_use_coupon_amount')
    feature = pd.merge(feature, t, on='Merchant_id', how='left')

    # 商家被领券并消费数
    t = history[history.Date.notnull() & history.Date_received.notnull()][['Merchant_id', 'cnt']]
    center = pd.pivot_table(t, index='Merchant_id', values='cnt', aggfunc=len)
    center = pd.DataFrame(center).rename(columns={'cnt': 'receive_consume_cnt'}).reset_index()
    feature = pd.merge(feature, center, on='Merchant_id', how='left')

    # 商家被领券数
    t = history[history.Date_received.notnull()][['Merchant_id', 'cnt']]
    center = pd.pivot_table(t, index='Merchant_id', values='cnt', aggfunc=len)
    center = pd.DataFrame(center).rename(columns={'cnt': 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, center, on='Merchant_id', how='left')

    # 使用了优惠券进行购买的用户距离商家的最小距离
    t = history[history['Date'].notnull() & history['Coupon_id'] != np.nan][['Distance', 'Merchant_id']]
    t = t.groupby(['Merchant_id'])['Distance'].min().reset_index(name='Coupon_Merchant_min_distance')
    feature = pd.merge(feature, t, on='Merchant_id', how='left')

    # 使用了优惠券进行购买的用户距离商家的最大距离
    t = history[history['label'] == 1][['Merchant_id', 'Distance']]
    t['Distance'] = t['Distance'].map(lambda x: 0 if x == -1 else int(x))
    center = pd.pivot_table(t, index='Merchant_id', values='Distance', aggfunc=np.max)
    center = pd.DataFrame(center).rename(columns={'Distance': '15_max_consume_distance'}).reset_index()
    feature = pd.merge(feature, center, on='Merchant_id', how='left')

    # 使用了优惠券进行购买的用户距离商家的平均距离
    center = pd.pivot_table(t, index='Merchant_id', values='Distance', aggfunc=np.mean)
    center = pd.DataFrame(center).rename(columns={'Distance': '15_mean_consume_distance'}).reset_index()
    feature = pd.merge(feature, center, on='Merchant_id', how='left')
    # 优惠券特征
    # 满减的优惠价格
    feature['cut_price'] = list(
        map(lambda x: float(str(x).split(':')[1]) if ':' in str(x) else 0, feature['Discount_rate']))

    # 每种优惠券的数量
    t = history[history.Date.notnull()][['Coupon_id', 'cnt']]
    center = pd.pivot_table(t, index='Coupon_id', values='cnt', aggfunc=len)
    center = pd.DataFrame(center).rename(columns={'cnt': 'cunsume_cnt'}).reset_index()
    feature = pd.merge(feature, center, on='Coupon_id', how='left')
    # 领券并消费数/领券数
    feature['receive_rate'] = feature['get_consume'] / feature['Coupon_amount']
    # 顾客-商家特征
    # 顾客在不同商家领取优惠券的数量
    t = feature[feature['Coupon_id'].notnull()][['User_id', 'Merchant_id']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Merchant_id'])['cnt'].sum().reset_index(name='receive_coupon_in_different_merchant')
    t.drop_duplicates(inplace=True)
    feature = pd.merge(feature, t, on=['User_id', 'Merchant_id'], how='left')

    # 顾客在不同商家领取优惠券并消费的数量
    t = history[history['Coupon_id'].notnull() & history['Date'].notnull()][['User_id', 'Merchant_id']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Merchant_id'])['cnt'].sum().reset_index(name='receive_use__coupon_in_different_merchant')
    t.drop_duplicates(inplace=True)
    feature = pd.merge(feature, t, on=['User_id', 'Merchant_id'], how='left')

    # 顾客在商家领取优惠券数
    t = history[history.Date_received.notnull()]
    center = pd.pivot_table(t, index=['User_id', 'Merchant_id'], values='cnt', aggfunc=len)
    center = pd.DataFrame(center).rename(columns={'cnt': 'receive1_cnt'}).reset_index()
    feature = pd.merge(feature, center, on=['User_id', 'Merchant_id'], how='left')

    # 用户在该商家领取满减优惠券的数量
    t = feature[feature['Date_received'].notnull() & feature['is_manjian'] == 1][['User_id', 'Merchant_id']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Merchant_id'])['cnt'].sum().reset_index(name='User_Merchant_manjian_coupon')
    feature = pd.merge(feature, t, on=['User_id', 'Merchant_id'], how='left')

    # 满减优惠券占比
    feature['User_Merchant_manjian_rate'] = list(
        map(lambda x, y: x / y if y != 0 else 0, feature['User_Merchant_manjian_coupon'],
            feature['receive1_cnt']))

    # 一个顾客在一个商家领取并消费的数量
    t = history[history['label'] == 1]
    t['consume_gap'] = (t['date'] - t['date_received']).map(lambda x: x.total_seconds() / (60 * 60 * 24))
    center = pd.pivot_table(t, index=['User_id', 'Merchant_id'], values='consume_gap', aggfunc=np.max)
    center = pd.DataFrame(center).rename(columns={'consume_gap': '15_max_consume_gap'}).reset_index()
    feature = pd.merge(feature, center, on=['User_id', 'Merchant_id'], how='left')

    # 顾客在不同商家领取优惠券消费/领取优惠券
    feature['consume_rate'] = feature['15_max_consume_gap'] / feature[
        'receive_coupon_in_different_merchant']

    # 顾客在不同商家消费的数量
    t = history[history['Date'] != np.nan][['User_id', 'Merchant_id']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Merchant_id'])['cnt'].sum().reset_index(name='User_different_Merchant')
    t.drop_duplicates(inplace=True)
    feature = pd.merge(feature, t, on=['User_id', 'Merchant_id'], how='left')


    # 一个顾客在一个商家消费数量
    t = feature[['User_id', 'Merchant_id']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Merchant_id'])['cnt'].sum().reset_index(name='User_Merchant_per_consume')
    feature = pd.merge(feature, t, on=['User_id', 'Merchant_id'], how='left')

    # 一个顾客在一个商家领取特定优惠券数量
    t = feature[['User_id', 'Merchant_id', 'Coupon_id']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Merchant_id', 'Coupon_id'])['cnt'].sum().reset_index(
        name='User_Merchant_Coupon_per_consume')
    feature = pd.merge(feature, t, on=['User_id', 'Merchant_id', 'Coupon_id'], how='left')

    # 用户领取优惠券距离
    t = feature[['User_id', 'Distance']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Distance'])['cnt'].sum().reset_index(name='User_received_Distance')
    feature = pd.merge(feature, t, on=['User_id', 'Distance'], how='left')

    # 其他特征

    # 顾客收到优惠券的数量
    t = history[history.Date_received.notnull()][['User_id', 'cnt']]  # dataframe类型，不加[]为series
    center = pd.pivot_table(t, index='User_id', values='cnt', aggfunc=len)
    center = pd.DataFrame(center).rename(columns={'cnt': 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, center, on='User_id', how='left')

    # 顾客收到的不同的优惠券数量
    t = history[history.Date_received.notnull()][['User_id', 'Coupon_id']]
    center = pd.pivot_table(t, index='User_id', values='Coupon_id', aggfunc=lambda x: len(set(x)))
    center = pd.DataFrame(center).rename(columns={'Coupon_id': 'differ_coupon_cnt'}).reset_index()
    feature = pd.merge(feature, center, on='User_id', how='left')

    # 顾客在不同日期收到的优惠券的数量
    t = feature[['User_id', 'Date_received']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Date_received'])['cnt'].sum().reset_index(name='User_Date_received')
    feature = pd.merge(feature, t, on=['User_id', 'Date_received'], how='left')

    # 顾客在不同日期收到的不同的优惠券的数量
    t = feature[['User_id', 'Coupon_id', 'Date_received']]
    t['cnt'] = 1
    t = t.groupby(['User_id', 'Coupon_id', 'Date_received'])['cnt'].sum().reset_index(name='User_Date_received_coupon')
    feature = pd.merge(feature, t, on=['User_id', 'Coupon_id', 'Date_received'], how='left')

    # 每个用户的最小折扣率
    t = history[history.Date.notnull() & history.Date_received.notnull()][['User_id', 'discount_rate']]
    center = pd.pivot_table(t, index='User_id', values='discount_rate', aggfunc=np.min)
    center = pd.DataFrame(center).rename(columns={'discount_rate': 'min_discount_rate'}).reset_index()
    feature = pd.merge(feature, center, on='User_id', how='left')

    # 每个用户的最大折扣率
    t = history[history.Date.notnull() & history.Date_received.notnull()][['User_id', 'discount_rate']]
    center = pd.pivot_table(t, index='User_id', values='discount_rate', aggfunc=np.max)
    center = pd.DataFrame(center).rename(columns={'discount_rate': 'max_discount_rate'}).reset_index()
    feature = pd.merge(feature, center, on='User_id', how='left')

    # 顾客消费最近距离
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

    # 商家被领取特定优惠券种类
    t = feature[feature['Date_received'].notnull()][['Merchant_id', 'Coupon_id']]
    t['cnt'] = 1
    t = t.groupby(['Merchant_id', 'Coupon_id'])['cnt'].sum().reset_index(name='Merchant_received_special_coupon')
    feature = pd.merge(feature, t, on=['Merchant_id', 'Coupon_id'], how='left')

    # 不同优惠率折扣券数量
    t = feature[feature['Discount_rate'].notnull()][['discount_rate']]
    t['cnt'] = 1
    t = t.groupby(['discount_rate'])['cnt'].sum().reset_index(name='different_rate_coupon_amount')
    feature = pd.merge(feature, t, on='discount_rate', how='left')

    # 在多少不同商家领取并消费
    t = a.groupby(['User_id'])['Merchant_id'].agg(lambda x: len(set(x))).reset_index(name='different_merchant_receive_consume')
    feature = pd.merge(feature, t, on=['User_id'], how='left')

    # 在多少不同商家领取优惠券
    t = feature.groupby(['User_id'])['Merchant_id'].agg(lambda x: len(set(x))).reset_index(
        name='different_merchant_receive')
    feature = pd.merge(feature, t, on=['User_id'], how='left')

    # 在多少不同商家领取并消费/在多少不同商家领取优惠券4
    feature['merchant_consume_rate'] = feature['different_merchant_receive_consume'] / feature['different_merchant_receive']

    # 不同商家发放优惠券的数量
    t = feature.groupby('Merchant_id').cnt.sum().reset_index(name='Merchant_Coupon_amount')
    feature = pd.merge(feature, t, on='Merchant_id', how='left')

    # 删除辅助提特征的'cnt'
    feature.drop(['cnt'], axis=1, inplace=True)
    feature['week'] = feature['date_received'].map(lambda x: x.isoweekday())
    feature['is_weekend'] = feature['week'].map(lambda x: 1 if x == 5 or x == 6 else 0)
    feature = pd.concat([feature, pd.get_dummies(feature['week'], prefix='week')], axis=1)
    feature.index = range(len(feature))

    # 提取距离特征
    # 将Distance中的nan填充为-1
    feature['Distance'].fillna(-1, inplace=True)
    # 判断距离是否为空
    feature['isnull_Distance'] = list(map(lambda x: 1 if x == -1 else 0, feature['Distance']))

    return feature


def get_week_feature(label_field):
    # 源数据
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)  # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Date_received'] = data['Date_received'].map(
        int)  # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    # 返回的特征数据集
    feature = data.copy()
    feature['week'] = feature['date_received'].map(lambda x: x.weekday())  # 星期几
    feature['is_weekend'] = feature['week'].map(lambda x: 1 if x == 5 or x == 6 else 0)  # 判断领券日是否为休息日
    feature = pd.concat([feature, pd.get_dummies(feature['week'], prefix='week')], axis=1)  # one-hot离散星期几
    feature.index = range(len(feature))  # 重置index
    # 返回
    return feature


def get_dataset(history_field, label_field):
    # 特征工程
    week_feat = get_week_feature(label_field)  # 日期特征
    simple_feat = get_label_field_feature(label_field, history_field)  # 示例简单特征

    # 构造数据集
    share_characters = list(
        set(simple_feat.columns.tolist()) & set(week_feat.columns.tolist()))  # 共有属性,包括id和一些基础特征,为每个特征块的交集
    dataset = pd.concat([week_feat, simple_feat.drop(share_characters, axis=1)], axis=1)
    # 删除无用属性并将label置于最后一列
    if 'Date' in dataset.columns.tolist():  # 表示训练集和验证集
        dataset.drop(['Merchant_id', 'Discount_rate', 'Date', 'date_received', 'date'], axis=1, inplace=True)
        label = dataset['label'].tolist()
        dataset.drop(['label'], axis=1, inplace=True)
        dataset['label'] = label
    else:  # 表示测试集
        dataset.drop(['Merchant_id', 'Discount_rate', 'date_received'], axis=1, inplace=True)
    # 修正数据类型
    dataset['User_id'] = dataset['User_id'].map(int)
    dataset['Coupon_id'] = dataset['Coupon_id'].map(int)
    dataset['Date_received'] = dataset['Date_received'].map(int)
    dataset['Distance'] = dataset['Distance'].map(int)
    if 'label' in dataset.columns.tolist():
        dataset['label'] = dataset['label'].map(int)
    # 去重
    dataset.drop_duplicates(keep='first', inplace=True)
    dataset.index = range(len(dataset))
    # 返回
    return dataset

def do_xgb1(dtrain,dtest,model):

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

def model_xgb(train, test):
    # xgb参数
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'silent': 1,
              'eta': 0.01,
              'max_depth': 5,
              'min_child_weight': 1,
              'gamma': 0,
              'lambda': 1,
              'colsample_bylevel': 0.7,
              'colsample_bytree': 0.7,
              'subsample': 0.9,
              'scale_pos_weight': 1}
    # 数据集
    dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
    dtest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))
    # 训练
    watchlist = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, num_boost_round=3000, evals=watchlist)
    result, feat_importance = do_xgb1(dtrain,dtest,model)
    return result, feat_importance

def start(tr11,te11):
    # 预处理
    tr11 = pre_do_data(tr11)
    te11 = pre_do_data(te11)
    # 打标
    tr11 = deal(tr11)
    return tr11,te11

def classify(tr11, te11):
    # 划分区间
    # 训练集历史区间、标签区间
    train_history_field = tr11[
        tr11['date_received'].isin(pd.date_range('2016/3/2', periods=60))]  # [20160302,20160501)
    train_label_field = tr11[
        tr11['date_received'].isin(pd.date_range('2016/5/16', periods=31))]  # [20160516,20160616)
    # 验证集历史区间、标签区间
    validate_history_field = tr11[
        tr11['date_received'].isin(pd.date_range('2016/1/16', periods=60))]  # [20160116,20160316)

    validate_label_field = tr11[
        tr11['date_received'].isin(pd.date_range('2016/3/31', periods=31))]  # [20160331,20160501)
    # 测试集历史区间、标签区间
    test_history_field = tr11[
        tr11['date_received'].isin(pd.date_range('2016/4/17', periods=60))]  # [20160417,20160616)
    test_label_field = te11.copy()  # [20160701,20160801)

    # 构造训练集、验证集、测试集
    print('构造训练集')
    train = get_dataset(train_history_field, train_label_field)
    print('构造验证集')
    validate = get_dataset(validate_history_field, validate_label_field)
    print('构造测试集')
    test = get_dataset(test_history_field, test_label_field)

    return train,validate,test

if __name__ == '__main__':
    # 源数据
    tr11 = pd.read_csv(r'ccf_offline_stage1_train.csv')
    te11 = pd.read_csv(r'ccf_offline_stage1_test_revised.csv')

    tr11, te11 = start(tr11, te11)
    train, validate, test = classify(tr11,te11)
    # 线上训练
    big_train = pd.concat([train, validate], axis=0)
    result, feat_importance = model_xgb(big_train, test)
    # 保存
    result.to_csv(r'result.csv', index=False, header=None)