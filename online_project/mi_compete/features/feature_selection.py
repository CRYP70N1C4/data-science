from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
import numpy as np

pd.set_option('display.expand_frame_repr', False)

# clf = RandomForestClassifier(n_estimators=100, max_depth=10)
#
# X = None
# y = None
# clf.fit(X, y)

# print(clf.feature_importances_)

TEST = 'mifi_compete_test_labels'
TRAIN = 'mifi_compete_train_labels'

features = ["mibasic_compete_data",
            "mispend_compete_data",
            "keywords_compete_data",
            "keyword_classapp_micloud_compete_samples",
            "interest_compete_data",
            "miothers_compete_data",
            "app_details_1_compete_data",
            "app_details_2_1_compete_data",
            "app_details_2_2_compete_data",
            "app_details_3_1_compete_data",
            "app_details_3_2_compete_data",
            "app_usage_samples",
            "app_usage_sequence_samples",
            "mifi_page_visit_samples",
            "user_info",
            "user_interest_data"]


def read(path):
    base_dir = 'D:\\code\\xiaomi\\dmcontest\\2018'
    return pd.read_parquet(os.path.join(base_dir, path)).set_index('user_id')

def categories(df):
    df.f3 = pd.Categorical(df.f3)
    df.f3 = df.f3.cat.codes
    return df

def text_vector(df):
    vec = CountVectorizer()
    X = vec.fit_transform(df)
    return pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

def select_by_variance(df,rate=0.1):
    sel = VarianceThreshold(threshold=(rate * (1 - rate)))
    sel.fit_transform(df)
    columns =  sel.get_support(indices=True)
    return df[df.columns[columns]]

def gen_test():
    data = {'id': np.arange(10), 'f0': np.ones(10),'f1': np.ones(10),'f2':np.random.uniform(size=10),'f3':["boy","girl"]*5}
    df = pd.DataFrame(data).set_index('id')
    new_df = pd.DataFrame({'f0': [1 if x > 5 else 0for x in range(10) ]})
    df.update(new_df)
    df['f4'] = [str(i // 2)+"abc" for i in range(10)]
    return df

# df = gen_test()
# df = categories(df)
# print(df)
# df = df.join(text_vector(df['f4'])).drop('f4',axis=1)
# print(select_by_variance(df))

df = read('mibasic_compete_data')
print(df.head(10))