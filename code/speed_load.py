import pandas as pd
import numpy as np
def transform_csv2pickle(path,usecols,dtypes):
    train = pd.read_csv(path,usecols=usecols,dtype=dtypes)
    train.to_pickle('/share/zongyu/task_embedding/ours/torch_ubi/data/ubi/train.pkl')

path = '/share/zongyu/task_embedding/ours/torch_ubi/data/ubi/train.csv'
basecols = ['row_id', 'time_id', 'investment_id', 'target']
features = [f'f_{i}' for i in range(300)]
dtypes = {
    'row_id': 'str',
    'time_id': 'uint16',
    'investment_id': 'uint16',
    'target': 'float32',
}
for col in features:
    dtypes[col] = 'float32'
    
transform_csv2pickle(path, basecols+features, dtypes)
print(0)