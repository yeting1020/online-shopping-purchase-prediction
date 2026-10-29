from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import BernoulliNB
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.naive_bayes import GaussianNB
from scipy.special import logsumexp

# Load the dataset
df=pd.read_csv("online_shoppers_intention.csv")
X=df.drop(columns=["Revenue"])
y=df["Revenue"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

numeric_cols = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]

categorical_cols = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

preprocessor=ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), categorical_cols)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)

X_val_processed = preprocessor.transform(X_val)

def select_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
        pos_label=True
    )

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall

    f1_values = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0
    )

    best_index = np.argmax(f1_values)
    best_threshold = thresholds[best_index]

    predictions = probabilities >= best_threshold

    return best_threshold, predictions

# logistic regression model
C_values = [0.001, 0.01, 0.1, 1, 10, 100]

best_lr = None
best_C = None
best_val_f1 = -1
lr_results = []

for C in C_values:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train_processed, y_train)

    predictions = model.predict(X_val_processed)
    val_f1 = f1_score(
        y_val, predictions,
        pos_label=True,
        zero_division=0
    )

    lr_results.append({"C": C, "Validation_F1": val_f1})

    if val_f1 > best_val_f1:
        best_lr = model
        best_C = C
        best_val_f1 = val_f1

print(pd.DataFrame(lr_results).to_string(index=False))
print("best C:", best_C)
print("best validation F1:", best_val_f1)

lr_purchase_index = list(best_lr.classes_).index(True)

lr_val_prob = best_lr.predict_proba(
    X_val_processed
)[:, lr_purchase_index]

lr_threshold, lr_tuned_pred=select_threshold(
    y_val,
    lr_val_prob
)

print("\nbest threshold for logistic regression:", lr_threshold)

print("Validation F1 before threshold tuning:", best_val_f1)
print("Validation F1 after threshold tuning:", f1_score(
    y_val,
    lr_tuned_pred,
    pos_label=True,
    zero_division=0
))

print(classification_report(
    y_val,
    lr_tuned_pred,
    labels=[False, True],
    target_names=["No purchase", "Purchase"],
    digits=4,
    zero_division=0
))


#neural network model
alpha_values = [0.0001, 0.01, 0.1, 1.0]
mlp_results = []
best_mlp = None
best_alpha = None
best_mlp_f1 = -1

for alpha in alpha_values:
    for hidden_layer_size in [(16,), (32,), (64,)]:

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_size,
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            alpha=alpha,
            max_iter=1000,
            random_state=42
        )

        model.fit(X_train_processed, y_train)

        train_pred = model.predict(X_train_processed)
        val_pred = model.predict(X_val_processed)

        train_f1 = f1_score(
            y_train, train_pred,
            pos_label=True, zero_division=0
        )
        val_f1 = f1_score(
            y_val, val_pred,
            pos_label=True, zero_division=0
        )

        mlp_results.append({
            "alpha": alpha,
            "Hidden_Layer_Size": hidden_layer_size,
            "Train_F1": train_f1,
            "Validation_F1": val_f1,
            "Epochs": model.n_iter_,
            "Validation_Accuracy": accuracy_score(y_val, val_pred)
        })

        if val_f1 > best_mlp_f1:
            best_mlp = model
            best_alpha = alpha
            best_hidden_layer_size = hidden_layer_size
            best_mlp_f1 = val_f1

print("\nMLP Results:")
print(pd.DataFrame(mlp_results).to_string(index=False))

print("best alpha:", best_alpha)
print("best hidden layer size:", best_hidden_layer_size)
print("best validation F1:", best_mlp_f1)
best_mlp_val_pred = best_mlp.predict(X_val_processed)

mlp_purchase_index = list(best_mlp.classes_).index(True)

mlp_val_prob = best_mlp.predict_proba(
    X_val_processed
)[:, mlp_purchase_index]

mlp_threshold, mlp_tuned_pred = select_threshold(
    y_val,
    mlp_val_prob
)

print("\nBest threshold for MLP:", mlp_threshold)

print("Validation F1 before threshold tuning:", best_mlp_f1)
print("Validation F1 after threshold tuning:", f1_score(
    y_val,
    mlp_tuned_pred,
    pos_label=True,
    zero_division=0
))

print(classification_report(
    y_val,
    mlp_tuned_pred,
    labels=[False, True],
    target_names=["No purchase", "Purchase"],
    digits=4,
    zero_division=0
))

#Bayes
n_numeric = len(numeric_cols)

X_train_num = X_train_processed[:, :n_numeric]
X_train_cat = X_train_processed[:, n_numeric:]

X_val_num = X_val_processed[:, :n_numeric]
X_val_cat = X_val_processed[:, n_numeric:]

nb_results = []
best_nb_f1 = -1
best_nb_params = None
best_gaussian = None
best_bernoulli = None
best_nb_val_pred = None

for var_smoothing in [1e-9, 1e-7, 1e-5]:
    for alpha in [0.01, 0.1, 1.0, 10.0]:

        gaussian = GaussianNB(
            var_smoothing=var_smoothing
        )
        gaussian.fit(X_train_num, y_train)

        bernoulli = BernoulliNB(
            alpha=alpha,
            binarize=None
        )
        bernoulli.fit(X_train_cat, y_train)

        assert np.array_equal(
            gaussian.classes_,
            bernoulli.classes_
        )
        assert np.allclose(
            np.log(gaussian.class_prior_),
            bernoulli.class_log_prior_
        )

        gaussian_score = gaussian.predict_joint_log_proba(
            X_val_num
        )
        bernoulli_score = bernoulli.predict_joint_log_proba(
            X_val_cat
        )

        combined_score = (
            gaussian_score
            + bernoulli_score
            - np.log(gaussian.class_prior_)
        )

        pred = gaussian.classes_[
            np.argmax(combined_score, axis=1)
        ]

        val_f1 = f1_score(
            y_val,
            pred,
            pos_label=True,
            zero_division=0
        )

        nb_results.append({
            "alpha": alpha,
            "var_smoothing": var_smoothing,
            "Validation_Accuracy": accuracy_score(y_val, pred),
            "Validation_F1": val_f1
        })

        if val_f1 > best_nb_f1:
            best_nb_f1 = val_f1
            best_nb_params = {
                "alpha": alpha,
                "var_smoothing": var_smoothing
            }
            best_gaussian = gaussian
            best_bernoulli = bernoulli
            best_nb_val_pred = pred.copy()

nb_scores = (
    best_gaussian.predict_joint_log_proba(X_val_num)
    + best_bernoulli.predict_joint_log_proba(X_val_cat)
    - np.log(best_gaussian.class_prior_)
)

nb_log_prob = nb_scores - logsumexp(
    nb_scores, axis=1, keepdims=True
)

purchase_index = list(best_gaussian.classes_).index(True)
nb_purchase_prob = np.exp(nb_log_prob[:, purchase_index])

nb_threshold, nb_tuned_pred = select_threshold(
    y_val,
    nb_purchase_prob
)

print("\nbest threshold:", nb_threshold)

print("Validation F1 before threshold tuning:", best_nb_f1)
print("Validation F1 after threshold tuning:", f1_score(
    y_val,
    nb_tuned_pred,
    pos_label=True,
    zero_division=0
))

print(classification_report(
    y_val,
    nb_tuned_pred,
    labels=[False, True],
    target_names=["No purchase", "Purchase"],
    digits=4,
    zero_division=0
))



X_test_processed = preprocessor.transform(X_test)

# 逻辑回归
lr_index = list(best_lr.classes_).index(True)
lr_test_prob = best_lr.predict_proba(
    X_test_processed
)[:, lr_index]

# MLP
mlp_index = list(best_mlp.classes_).index(True)
mlp_test_prob = best_mlp.predict_proba(
    X_test_processed
)[:, mlp_index]

# 混合朴素贝叶斯
X_test_num = X_test_processed[:, :n_numeric]
X_test_cat = X_test_processed[:, n_numeric:]

nb_test_scores = (
    best_gaussian.predict_joint_log_proba(X_test_num)
    + best_bernoulli.predict_joint_log_proba(X_test_cat)
    - np.log(best_gaussian.class_prior_)
)

nb_test_log_prob = nb_test_scores - logsumexp(
    nb_test_scores, axis=1, keepdims=True
)

nb_index = list(best_gaussian.classes_).index(True)
nb_test_prob = np.exp(nb_test_log_prob[:, nb_index])

test_results = []

for name, probabilities, threshold in [
    ("Logistic Regression", lr_test_prob, lr_threshold),
    ("MLP", mlp_test_prob, mlp_threshold),
    ("Mixed NB", nb_test_prob, nb_threshold)
]:
    # 固定使用验证集选出的阈值
    pred = probabilities >= threshold

    test_results.append({
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(
            y_test, pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, pred, zero_division=0
        ),
        "F1": f1_score(
            y_test, pred, zero_division=0
        )
    })

print("\n最终测试集结果：")
print(pd.DataFrame(test_results).to_string(index=False))

from pathlib import Path

# 在当前文件夹下建立results文件夹
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# 保存三个模型的调参记录
pd.DataFrame(lr_results).to_csv(
    results_dir / "lr_tuning.csv", index=False
)

pd.DataFrame(mlp_results).to_csv(
    results_dir / "mlp_tuning.csv", index=False
)

pd.DataFrame(nb_results).to_csv(
    results_dir / "nb_tuning.csv", index=False
)

# 保存最终测试集比较
pd.DataFrame(test_results).to_csv(
    results_dir / "test_results.csv", index=False
)

print("\n实验结果已保存到results文件夹。")

final_settings = [
    {
        "Model": "Logistic Regression",
        "Parameters": f"C={best_C}",
        "Threshold": lr_threshold
    },
    {
        "Model": "MLP",
        "Parameters": (
            f"hidden_layer_sizes={best_hidden_layer_size}, "
            f"alpha={best_alpha}"
        ),
        "Threshold": mlp_threshold
    },
    {
        "Model": "Mixed NB",
        "Parameters": (
            f"alpha={best_nb_params['alpha']}, "
            f"var_smoothing={best_nb_params['var_smoothing']}"
        ),
        "Threshold": nb_threshold
    }
]

settings_df = pd.DataFrame(final_settings)

settings_df.to_csv(
    results_dir / "final_settings.csv",
    index=False
)

print("\n最终模型设置：")
print(settings_df.to_string(index=False))

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("results")
df = pd.read_csv(results_dir / "test_results.csv")

# 画两个子图
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, metric, title in [
    (axes[0], "Accuracy", "Test Accuracy"),
    (axes[1], "F1", "Purchase-Class Test F1")
]:
    bars = ax.bar(
        df["Model"],
        df[metric],
        color=["#4477AA", "#228833", "#CCBB44"]
    )

    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")

    # 在柱子上标出数值
    ax.bar_label(bars, fmt="%.4f", padding=3)

fig.tight_layout()
fig.savefig(
    results_dir / "test_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()