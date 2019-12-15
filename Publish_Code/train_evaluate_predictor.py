# Train and Evaluate the Predictor
from preprocessing import *
from sklearn import linear_model
import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import shap

dataset = data_preprocessing(threshold=2)
dataset["avg_agi"].replace('', np.nan, inplace=True)
dataset.dropna(subset=["avg_agi"], inplace=True)
train, test, valid = train_test_validation(dataset)

# Baseline Model
model = linear_model.LogisticRegression(C = 0.005, class_weight='balanced')
train_data, train_label = data_label(train)
valid_data, valid_label = data_label(valid)
test_data, test_label = data_label(test)
model.fit(train_data, train_label)
# Baseline accuracy
print("Baseline accuracy: ", model.score(test_data, test_label))

# XGBoost Model
D_train = xgb.DMatrix(train_data, label=train_label)
D_valid = xgb.DMatrix(valid_data, label=valid_label)
param = {
    # 'eta': 0.3, 
    # 'max_depth': 3,  
    # 'objective': 'multi:softprob',  
    # 'num_class': 2,
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'eval_metrix': 'auc',
    'eta': 0.3,
    'gamma': 0,
    'min_child_weight': 0.01,
    'max_depth': 6,
    'max_delta_step': 1,
    'subsample': 0.85,
    'colsample_bytree': 0.45,
    'colsample_bylevel': 0.7,
    'colsample_bynode': 1.0,
    'lambda': 5,
    'alpha': 0.2
    } 
steps = 100  # The number of training iterations
model = xgb.train(param, D_train, steps)
preds = model.predict(D_valid)
best_preds = np.asarray([1 if p >= 0.5 else 0 for p in preds])
# XGBoost Metrics
print("Precision = {}".format(precision_score(valid_label, best_preds, average='macro')))
print("Recall = {}".format(recall_score(valid_label, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(valid_label, best_preds)))
# ROC Curve
# Roc Curve
fpr, tpr, _ = roc_curve(valid_label, best_preds)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange',\
lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

# Shap Feature Importance analysis
shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(train_data)
shap.summary_plot(shap_values, train_data)