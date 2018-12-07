import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer,Binarizer,OneHotEncoder
#https://www.cnblogs.com/jasonfreak/p/5448385.html
#https://www.cnblogs.com/jasonfreak/p/5448462.html
data = np.array([[1, 2, 3], [2, 2, 2], [3, 3, 3]], dtype=np.float32)

# （x-min)/(max - min)标准化 基于列
print(MinMaxScaler().fit_transform(data))
# （x-avg)/S 标准化 基于列
print(StandardScaler().fit_transform(data))
# 归一化 基于行
print(Normalizer().fit_transform(data))
#二值化
print(Binarizer(threshold=2).fit_transform(data))
#二值化
print(OneHotEncoder().fit_transform(data).reshape(-1,1))
