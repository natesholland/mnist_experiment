import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

# import code; code.interact(local=dict(globals(), **locals()))

TREE_NUM = 10
TREE_DEPTH=10

data = pd.read_csv('train.csv')
X_tr = data.values[:, 1:].astype(float)
y_tr = data.values[:, 0]


print("Beginning training")
recognizer = RandomForestClassifier(TREE_NUM, max_depth=TREE_DEPTH)
print("Training Complete. Beginning Classification")
print(cross_val_score(recognizer, X_tr, y_tr))
