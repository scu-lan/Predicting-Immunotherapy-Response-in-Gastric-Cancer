import numpy as np
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, entropy_sampling
import torch

early_stopping_rounds = 30 
min_improvement = 1e-4     
no_improvement_count = 0   
best_accuracy = 0           
fake_data  = x_6.iloc[:, :-1].values # All columns except the last one
fake_labels_class = x_6.iloc[:, -1].values     # Only the last column

initial_idx = np.array([], dtype=int)
for class_label in [0, 1]:  
    idx = np.random.choice(
        np.where(fake_labels_class == class_label)[0],  
        size=5, 
        replace=False
    )
    initial_idx = np.concatenate((initial_idx, idx))

# 选择初始的 X 和 y
X_initial = fake_data[initial_idx]
y_initial = fake_labels_class[initial_idx]


X_pool = np.delete(fake_data, initial_idx, axis=0)
y_pool = np.delete(fake_labels_class, initial_idx, axis=0)

learner = ActiveLearner(estimator=LogisticRegression(random_state=42, max_iter=1000),
                        query_strategy=entropy_sampling,
                        X_training=X_initial, y_training=y_initial
                        )
predict_hist = [learner.score(data_test, labels_test)]
n_queries = 100

for index in range(n_queries):
    query_idx, query_instance = learner.query(X_pool, n_instances=1)
    learner.teach(X_pool[query_idx], y_pool[query_idx])
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    model_accuracy = learner.score(data_test, labels_test)
    model_auc = learner.score(data_test, labels_test)
    print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
    predict_hist.append(model_accuracy)

 
    if model_accuracy - best_accuracy > min_improvement:
        best_accuracy = model_accuracy
        no_improvement_count = 0  # 重置计数器
    else:
        no_improvement_count += 1  # 增加计数器

    if no_improvement_count >= early_stopping_rounds:
        print(f"Early stopping at query {index + 1}. Best accuracy: {best_accuracy:0.4f}")
        break
