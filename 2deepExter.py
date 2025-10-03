import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
import torch
import torch.nn as nn
from pycox.models import CoxPH as PyCoxPH
from sklearn.model_selection import KFold

# ========== 1. 读取数据 ==========
df_train = pd.read_csv("D:\\ChongQwork\\F63GBMlipid\\S2cliMo\\GBM\\trainSet.csv", index_col=0)

cols = [c for c in df_train.columns if c not in ["OS", "OS_Time"]]
df_train["duration"] = df_train["OS_Time"]
df_train["event"] = df_train["OS"]

df_test = pd.read_csv("D:\\ChongQwork\\F63GBMlipid\\S2cliMo\\GBM\\extestSet.csv", index_col=0)
df_test["duration"] = df_test["OS_Time"]
df_test["event"] = df_test["OS"]

X_train = df_train[cols].values
X_test = df_test[cols].values

y_train = Surv.from_arrays(event=df_train["event"].astype(bool), time=df_train["duration"])
y_test = Surv.from_arrays(event=df_test["event"].astype(bool), time=df_test["duration"])

print("最大随访时间 (年):", df_test["duration"].max())

# ========== 2. 单点 AUC 计算 ==========
def calc_auc(y_train, y_test, risk, t):
    """risk: 一维风险分数 (n_test,)"""
    if t >= df_test["duration"].max():
        return np.nan
    riskmat = risk.reshape(-1, 1)
    _, auc = cumulative_dynamic_auc(y_train, y_test, riskmat, [t])
    return float(np.atleast_1d(auc)[0])


# ========== 3. 逐个模型训练 + 评估 ==========

results = []
risk_dict = {}
kf = KFold(n_splits=2, shuffle=True, random_state=0)

# ---- CoxPH ----
print("\n[CoxPH] 5-fold CV on penalizer")
best_penalizer = None
best_score = -np.inf
for penalizer in [0.0, 0.01, 0.1, 1.0]:
    fold_scores = []
    for tr_idx, va_idx in kf.split(df_train):
        # 按索引切分 DataFrame
        df_tr = df_train.iloc[tr_idx]
        df_va = df_train.iloc[va_idx]
        cph_cv = CoxPHFitter(penalizer=penalizer)
        cph_cv.fit(df_tr[cols + ["duration", "event"]],
                   duration_col="duration", event_col="event")
        risk_va = cph_cv.predict_partial_hazard(df_va[cols]).values.reshape(-1)
        c = concordance_index_censored(df_va["event"].astype(bool),
                                       df_va["duration"].astype(float),
                                       risk_va)[0]
        fold_scores.append(c)
    mean_c = float(np.mean(fold_scores))
    print(f"  penalizer={penalizer:<6} mean C-index={mean_c:.4f}")
    if mean_c > best_score:
        best_score = mean_c
        best_penalizer = penalizer

print(f"==> Best penalizer: {best_penalizer} (mean C-index={best_score:.4f})")
# 用最佳参数在全训练集重训，并在测试集评估
cph = CoxPHFitter(penalizer=best_penalizer)
cph.fit(df_train[cols+["duration","event"]], duration_col="duration", event_col="event")
risk_CoxPH = cph.predict_partial_hazard(df_test[cols]).values.reshape(-1)
print(risk_CoxPH)
print('risk_______cph')
cindex = concordance_index_censored(df_test["event"].astype(bool),
                                    df_test["duration"].astype(float),
                                    risk_CoxPH)[0]
auc1 = calc_auc(y_train, y_test, risk_CoxPH, 1.0)
auc2 = calc_auc(y_train, y_test, risk_CoxPH, 2.0)
auc3 = calc_auc(y_train, y_test, risk_CoxPH, 3.0)
results.append(("CoxPH", cindex, auc1, auc2, auc3))
risk_dict['CoxPH'] = risk_CoxPH.copy()

times_grid = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
              2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]

auc_table_CoxPH = {}
for t in times_grid:
    auc_table_CoxPH[t] = calc_auc(y_train, y_test, risk_CoxPH, t)

print("CoxPH 多时间点 AUC：", auc_table_CoxPH)

# ---- CoxNet ----

print("\n[CoxNet] 5-fold CV on l1_ratio")
best_l1_ratio = None
best_score = -np.inf
for l1r in [0.1, 0.5, 0.9]:
    fold_scores = []
    for tr_idx, va_idx in kf.split(X_train):
        pipe_cv = make_pipeline(StandardScaler(),
                                CoxnetSurvivalAnalysis(l1_ratio=l1r, n_alphas=50))
        pipe_cv.fit(X_train[tr_idx], y_train[tr_idx])
        risk_va = pipe_cv.predict(X_train[va_idx])
        c = concordance_index_censored(y_train[va_idx]["event"],
                                       y_train[va_idx]["time"],
                                       risk_va)[0]
        fold_scores.append(c)
    mean_c = float(np.mean(fold_scores))
    print(f"  l1_ratio={l1r:<3} mean C-index={mean_c:.4f}")
    if mean_c > best_score:
        best_score = mean_c
        best_l1_ratio = l1r

print(f"==> Best l1_ratio: {best_l1_ratio} (mean C-index={best_score:.4f})")
pipe = make_pipeline(StandardScaler(),
                     CoxnetSurvivalAnalysis(l1_ratio=best_l1_ratio, n_alphas=50))
pipe.fit(X_train, y_train)
risk_CoxNet = pipe.predict(X_test)
print(risk_CoxNet)
print('risk_______CoxNet')
cindex = concordance_index_censored(df_test["event"].astype(bool), df_test["duration"], risk_CoxNet)[0]
auc1 = calc_auc(y_train, y_test, risk_CoxNet, 1.0)
auc2 = calc_auc(y_train, y_test, risk_CoxNet, 2.0)
auc3 = calc_auc(y_train, y_test, risk_CoxNet, 3.0)
results.append(("CoxNet", cindex, auc1, auc2, auc3))
risk_dict['CoxNet'] = risk_CoxNet.copy()

auc_table_CoxNet = {}
for t in times_grid:
    auc_table_CoxNet[t] = calc_auc(y_train, y_test, risk_CoxNet, t)

print("CoxNet 多时间点 AUC：", auc_table_CoxNet)


# ---- RSF ----
print("\n[RSF] 5-fold CV on n_estimators")
best_n_estimators_rsf = None
best_score = -np.inf
for n_est in [100, 200, 400]:
    fold_scores = []
    for tr_idx, va_idx in kf.split(X_train):
        rsf_cv = RandomSurvivalForest(n_estimators=n_est, random_state=0)
        rsf_cv.fit(X_train[tr_idx], y_train[tr_idx])
        surv_funcs_va = rsf_cv.predict_survival_function(X_train[va_idx])
        # 用 1 年时刻的 1 - S(t) 作为风险分数
        risk_va = np.array([1.0 - fn(1.0) for fn in surv_funcs_va])
        c = concordance_index_censored(y_train[va_idx]["event"],
                                       y_train[va_idx]["time"],
                                       risk_va)[0]
        fold_scores.append(c)
    mean_c = float(np.mean(fold_scores))
    print(f"  n_estimators={n_est:<3} mean C-index={mean_c:.4f}")
    if mean_c > best_score:
        best_score = mean_c
        best_n_estimators_rsf = n_est

print(f"==> Best n_estimators (RSF): {best_n_estimators_rsf} (mean C-index={best_score:.4f})")
rsf = RandomSurvivalForest(n_estimators=best_n_estimators_rsf, random_state=0)
rsf.fit(X_train, y_train)
surv_funcs = rsf.predict_survival_function(X_test)

risk_RSF = np.array([1.0 - fn(1.0) for fn in surv_funcs])
print(risk_RSF)
print('risk_______RSF')
cindex = concordance_index_censored(df_test["event"].astype(bool), df_test["duration"], risk_RSF)[0]
auc1 = calc_auc(y_train, y_test, risk_RSF, 1.0)
auc2 = calc_auc(y_train, y_test, risk_RSF, 2.0)
auc3 = calc_auc(y_train, y_test, risk_RSF, 3.0)
results.append(("RSF", cindex, auc1, auc2, auc3))
risk_dict['RSF'] = risk_RSF.copy()

auc_table_RSF = {}
for t in times_grid:
    auc_table_RSF[t] = calc_auc(y_train, y_test, risk_RSF, t)

print("RSF 多时间点 AUC：", auc_table_RSF)



# ---- SurSVM ----
print("\n[SurSVM] 5-fold CV on alpha")
best_alpha = None
best_score = -np.inf
for alpha in [1e-6, 1e-5, 1e-4]:
    fold_scores = []
    for tr_idx, va_idx in kf.split(X_train):
        svm_cv = make_pipeline(StandardScaler(),
                               FastSurvivalSVM(alpha=alpha, rank_ratio=0.9, random_state=0))
        svm_cv.fit(X_train[tr_idx], y_train[tr_idx])
        # 约定风险为 -预测值（预测值越大代表生存越好）
        risk_va = -svm_cv.predict(X_train[va_idx])
        c = concordance_index_censored(y_train[va_idx]["event"],
                                       y_train[va_idx]["time"],
                                       risk_va)[0]
        fold_scores.append(c)
    mean_c = float(np.mean(fold_scores))
    print(f"  alpha={alpha:<.0e} mean C-index={mean_c:.4f}")
    if mean_c > best_score:
        best_score = mean_c
        best_alpha = alpha

print(f"==> Best alpha (SurSVM): {best_alpha} (mean C-index={best_score:.4f})")
svm = make_pipeline(StandardScaler(),
                    FastSurvivalSVM(alpha=best_alpha, rank_ratio=0.9, random_state=0))
svm.fit(X_train, y_train)
risk_SVM = -svm.predict(X_test)
print(risk_SVM)
print('risk_______SVM')
cindex = concordance_index_censored(df_test["event"].astype(bool), df_test["duration"], risk_SVM)[0]
auc1 = calc_auc(y_train, y_test, risk_SVM, 1.0)
auc2 = calc_auc(y_train, y_test, risk_SVM, 2.0)
auc3 = calc_auc(y_train, y_test, risk_SVM, 3.0)
results.append(("SurSVM", cindex, auc1, auc2, auc3))
risk_dict['SurSVM'] = risk_SVM.copy()

auc_table_SVM = {}
for t in times_grid:
    auc_table_SVM[t] = calc_auc(y_train, y_test, risk_SVM, t)

print("SurSVM 多时间点 AUC：", auc_table_SVM)


# ---- GBST (GradientBoostingSurvivalAnalysis) ----

print("\n[GBST] 5-fold CV on n_estimators")
best_n_estimators_gbst = None
best_score = -np.inf
for n_est in [100, 200, 300]:
    fold_scores = []
    for tr_idx, va_idx in kf.split(X_train):
        gbst_cv = GradientBoostingSurvivalAnalysis(
            loss="coxph", learning_rate=0.1, n_estimators=n_est, max_depth=3, random_state=0
        )
        gbst_cv.fit(X_train[tr_idx], y_train[tr_idx])
        risk_va = gbst_cv.predict(X_train[va_idx])
        c = concordance_index_censored(y_train[va_idx]["event"],
                                       y_train[va_idx]["time"],
                                       risk_va)[0]
        fold_scores.append(c)
    mean_c = float(np.mean(fold_scores))
    print(f"  n_estimators={n_est:<3} mean C-index={mean_c:.4f}")
    if mean_c > best_score:
        best_score = mean_c
        best_n_estimators_gbst = n_est

print(f"==> Best n_estimators (GBST): {best_n_estimators_gbst} (mean C-index={best_score:.4f})")
gbst = GradientBoostingSurvivalAnalysis(
    loss="coxph", learning_rate=0.1, n_estimators=best_n_estimators_gbst, max_depth=3, random_state=0
)

gbst.fit(X_train, y_train)

risk_GBST = gbst.predict(X_test)
print(risk_GBST)
print('risk_______GBST')

cindex = concordance_index_censored(
    df_test["event"].astype(bool),
    df_test["duration"].astype(float),
    risk_GBST
)[0]
auc1 = calc_auc(y_train, y_test, risk_GBST, 1.0)
auc2 = calc_auc(y_train, y_test, risk_GBST, 2.0)
auc3 = calc_auc(y_train, y_test, risk_GBST, 3.0)
results.append(("GBST", cindex, auc1, auc2, auc3))
risk_dict['GBST'] = risk_GBST.copy()


auc_table_GBST = {}
for t in times_grid:
    auc_table_GBST[t] = calc_auc(y_train, y_test, risk_GBST, t)

print("GBST 多时间点 AUC：", auc_table_GBST)


# ---- DeepSurv ----
print("\n[DeepSurv] 5-fold CV on hidden_units")

torch.manual_seed(0)
np.random.seed(0)
# 超参搜索空间（可按需增减）
hidden_grid = [16, 32, 64]

best_hidden = None
best_cv_cindex = -np.inf

# 定义网络
class Net(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Linear(h, 1)
        )
    def forward(self, x):
        return self.seq(x)

for h in hidden_grid:
    fold_scores = []
    for tr_idx, va_idx in kf.split(X_train):
        # 1) 每折标准化：训练折 fit_transform，验证折 transform
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[tr_idx]).astype("float32")
        X_va = scaler.transform(X_train[va_idx]).astype("float32")

        dur_tr = df_train["duration"].values[tr_idx].astype("float32")
        evt_tr = df_train["event"].values[tr_idx].astype("float32")

        # 2) 每折固定随机种子，提升复现性
        torch.manual_seed(0)
        np.random.seed(0)

        # 3) 建模与优化设置（更稳：较小 lr + weight_decay）
        net_cv = Net(X_tr.shape[1], h)
        optimizer = torch.optim.Adam(net_cv.parameters(), lr=5e-4, weight_decay=1e-4)
        model_cv = PyCoxPH(net_cv, optimizer)

        # 4) 训练（适度 epoch）
        model_cv.fit(X_tr, (dur_tr, evt_tr), batch_size=128, epochs=20, verbose=False)

        # 5) 验证集预测 + 兜底，计算 C-index
        risk_va = model_cv.predict(X_va).reshape(-1).astype(np.float64)
        if np.any(~np.isfinite(risk_va)):
            risk_va = np.nan_to_num(risk_va, nan=0.0, posinf=1e6, neginf=-1e6)

        c = concordance_index_censored(
            y_train[va_idx]["event"],
            y_train[va_idx]["time"],
            risk_va
        )[0]
        fold_scores.append(c)

    mean_c = float(np.mean(fold_scores))
    print(f"  hidden_units={h:<2} mean C-index={mean_c:.4f}")
    if mean_c > best_cv_cindex:
        best_cv_cindex = mean_c
        best_hidden = h

print(f"==> Best hidden_units: {best_hidden} (CV mean C-index={best_cv_cindex:.4f})")

# ===== 用最佳 hidden 在全训练集重训，并在独立测试集评估 =====
# 全训练集标准化（fit on train, transform train/test）
scaler_final = StandardScaler()
X_train_ds = scaler_final.fit_transform(X_train).astype("float32")
X_test_ds  = scaler_final.transform(X_test).astype("float32")

dur_train_np = df_train["duration"].values.astype("float32")
evt_train_np = df_train["event"].values.astype("float32")

torch.manual_seed(0)
np.random.seed(0)

net = Net(X_train_ds.shape[1], best_hidden)
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-4)
model = PyCoxPH(net, optimizer)
model.fit(X_train_ds, (dur_train_np, evt_train_np), batch_size=128, epochs=20, verbose=False)

risk_DeepSurv = model.predict(X_test_ds).reshape(-1).astype(np.float64)
if np.any(~np.isfinite(risk_DeepSurv)):
    risk_DeepSurv = np.nan_to_num(risk_DeepSurv, nan=0.0, posinf=1e6, neginf=-1e6)

print(risk_DeepSurv)
print('risk_______DeepSurv')

cindex = concordance_index_censored(df_test["event"].astype(bool), df_test["duration"], risk_DeepSurv)[0]
auc1 = calc_auc(y_train, y_test, risk_DeepSurv, 1.0)
auc2 = calc_auc(y_train, y_test, risk_DeepSurv, 2.0)
auc3 = calc_auc(y_train, y_test, risk_DeepSurv, 3.0)
results.append(("DeepSurv", cindex, auc1, auc2, auc3))
risk_dict['DeepSurv'] = risk_DeepSurv.copy()

auc_table_DeepSurv = {}
for t in times_grid:
    auc_table_DeepSurv[t] = calc_auc(y_train, y_test, risk_DeepSurv, t)

print("DeepSurv 多时间点 AUC：", auc_table_DeepSurv)

# ========== 4. 结果表 ==========
res_df = pd.DataFrame(results, columns=["Model", "C-index", "AUC1y", "AUC2y", "AUC3y"])
print(res_df)
res_df.to_csv('exterCindex.csv')

auc_df = pd.DataFrame({
    "CoxPH": auc_table_CoxPH,
    "CoxNet": auc_table_CoxNet,
    "RSF": auc_table_RSF,
    "SurSVM": auc_table_SVM,
    "GBST": auc_table_GBST,
    "DeepSurv": auc_table_DeepSurv,
}).T   # 转置一下，模型作为行

auc_df.index.name = "Model"
auc_df.columns = [f"AUC@{t}y" for t in auc_df.columns]

print(auc_df)
auc_df.to_csv("exterAUC.csv")

# ========= 额外：按多个时间点输出每位患者的生存概率 =========
# 想要的时间点（年）
times_multi = np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
                        2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0])

# 1) CoxPH（lifelines）：直接取每位病人的 S(t)
# 会返回一个 list（长度=样本数），每个元素是一个以 time 为索引的 Series
# 1) CoxPH（lifelines）
coxph_surv_df = cph.predict_survival_function(df_test[cols], times=times_multi)
S_coxph = coxph_surv_df.T.values   # 形状: (n_test, len(times_multi))
print(S_coxph)
print('S_coxph')
# 2) RSF（sksurv）：predict_survival_function 返回 StepFunction，可在任意 t 调用
rsf_surv_funcs = rsf.predict_survival_function(X_test)    # 列表，每个元素 fn: t -> S(t)
S_rsf = np.vstack([fn(times_multi) for fn in rsf_surv_funcs])  # (n_test, len(times_multi))
print(S_rsf)
print('S_rsf')

# ========== 汇总各模型 risk 为一个表：行=样本，列=模型 ==========
cols_for_df = ['CoxPH', 'CoxNet', 'RSF',  'SurSVM', 'GBST', 'DeepSurv']
available = [k for k in cols_for_df if k in risk_dict]

risk_df = pd.DataFrame(
    {k: risk_dict[k] for k in available},
    index=df_test.index
)

print("\n各模型风险分数（前5行）：")
print(risk_df.head())
risk_df.to_csv("exterRisk.csv", index=True)
# ==================================================
import joblib
#joblib.dump(cph, "CoxPH_model.pkl")
#joblib.dump(pipe, "CoxNet_model.pkl")
#joblib.dump(rsf, "RSF_model.pkl")
#joblib.dump(svm, "SVM_model.pkl")
#joblib.dump(gbst, "GBST_model.pkl")
#joblib.dump(model, "DeepSurv_model.pkl")
#cph2 = joblib.load("CoxPH_model.pkl")
