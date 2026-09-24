"""
eda_report.py
--------------
Generates EDA charts saved as PNG files for use in the Streamlit app.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SAVE_DIR = "eda_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

PALETTE = "Set2"
sns.set_theme(style="whitegrid", palette=PALETTE)


def _save(fig, name):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# GERMAN CREDIT RISK
# ──────────────────────────────────────────────────────────────────────────────

def german_eda(df):
    plots = {}

    # 1. Target distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    df["Risk"].value_counts().plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
    ax.set_title("German Credit: Risk Distribution")
    ax.set_xlabel("Risk"); ax.set_ylabel("Count")
    ax.bar_label(ax.containers[0])
    plots["german_target"] = _save(fig, "german_target.png")

    # 2. Age distribution by Risk
    fig, ax = plt.subplots(figsize=(7, 4))
    for risk, grp in df.groupby("Risk"):
        ax.hist(grp["Age"], bins=20, alpha=0.6, label=risk)
    ax.set_title("Age Distribution by Risk")
    ax.set_xlabel("Age"); ax.legend()
    plots["german_age"] = _save(fig, "german_age.png")

    # 3. Approval rate by Sex
    fig, ax = plt.subplots(figsize=(5, 4))
    approval = df.groupby("Sex")["Risk"].apply(lambda x: (x == "good").mean())
    approval.plot(kind="bar", ax=ax, color=["#3498db", "#e67e22"])
    ax.set_title("Approval Rate by Sex")
    ax.set_ylabel("Approval Rate"); ax.set_ylim(0, 1)
    ax.axhline(0.8, color="red", linestyle="--", label="80% DI threshold")
    ax.legend()
    plots["german_sex_approval"] = _save(fig, "german_sex_approval.png")

    # 4. Credit amount by Purpose (box)
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.groupby("Purpose")["Credit_amount"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="Purpose", y="Credit_amount", order=order, ax=ax, palette=PALETTE)
    ax.set_title("Credit Amount by Purpose")
    ax.set_xlabel("Purpose"); ax.set_ylabel("Credit Amount")
    plt.xticks(rotation=30, ha="right")
    plots["german_credit_purpose"] = _save(fig, "german_credit_purpose.png")

    # 5. Correlation heatmap
    df_num = df.select_dtypes(include=np.number)
    df_num["Risk_bin"] = (df["Risk"] == "good").astype(int)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_num.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("German Credit: Feature Correlation")
    plots["german_corr"] = _save(fig, "german_corr.png")

    # 6. Job vs Risk
    fig, ax = plt.subplots(figsize=(6, 4))
    crosstab = pd.crosstab(df["Job"], df["Risk"], normalize="index")
    crosstab.plot(kind="bar", stacked=True, ax=ax, color=["#e74c3c", "#2ecc71"])
    ax.set_title("Job Level vs Risk (Normalised)")
    ax.set_ylabel("Proportion"); ax.legend(title="Risk")
    plots["german_job_risk"] = _save(fig, "german_job_risk.png")

    return plots


# ──────────────────────────────────────────────────────────────────────────────
# LOAN DATASET
# ──────────────────────────────────────────────────────────────────────────────

def loan_eda(df):
    plots = {}

    # 1. Loan Status distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    df["Loan_Status"].value_counts().plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
    ax.set_title("Loan Status Distribution")
    ax.set_xlabel("Status"); ax.set_ylabel("Count")
    ax.bar_label(ax.containers[0])
    plots["loan_target"] = _save(fig, "loan_target.png")

    # 2. Approval rate by Gender
    fig, ax = plt.subplots(figsize=(5, 4))
    approval = df.groupby("Gender")["Loan_Status"].apply(lambda x: (x == "Y").mean())
    approval.plot(kind="bar", ax=ax, color=["#3498db", "#e67e22", "#9b59b6"])
    ax.set_title("Loan Approval Rate by Gender")
    ax.set_ylabel("Approval Rate"); ax.set_ylim(0, 1)
    ax.axhline(0.8, color="red", linestyle="--", label="80% DI threshold")
    ax.legend()
    plots["loan_gender_approval"] = _save(fig, "loan_gender_approval.png")

    # 3. Applicant Income Distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    df["ApplicantIncome"].plot(kind="hist", bins=40, ax=ax, color="#3498db", edgecolor="white")
    ax.set_title("Applicant Income Distribution")
    ax.set_xlabel("Income")
    plots["loan_income"] = _save(fig, "loan_income.png")

    # 4. LoanAmount vs Loan_Status
    fig, ax = plt.subplots(figsize=(7, 4))
    for status, grp in df.groupby("Loan_Status"):
        ax.hist(grp["LoanAmount"].dropna(), bins=30, alpha=0.6, label=status)
    ax.set_title("Loan Amount by Approval Status")
    ax.set_xlabel("Loan Amount"); ax.legend()
    plots["loan_amount_status"] = _save(fig, "loan_amount_status.png")

    # 5. Credit History vs Approval
    fig, ax = plt.subplots(figsize=(5, 4))
    crosstab = pd.crosstab(df["Credit_History"], df["Loan_Status"], normalize="index")
    crosstab.plot(kind="bar", stacked=True, ax=ax, color=["#e74c3c", "#2ecc71"])
    ax.set_title("Credit History vs Loan Approval")
    ax.set_xlabel("Credit History (1=Good)"); ax.set_ylabel("Proportion")
    ax.legend(title="Status")
    plots["loan_credit_hist"] = _save(fig, "loan_credit_hist.png")

    # 6. Education vs Approval
    fig, ax = plt.subplots(figsize=(5, 4))
    edu_approval = df.groupby("Education")["Loan_Status"].apply(lambda x: (x == "Y").mean())
    edu_approval.plot(kind="bar", ax=ax, color=["#8e44ad", "#2980b9"])
    ax.set_title("Approval Rate by Education")
    ax.set_ylabel("Approval Rate"); ax.set_ylim(0, 1)
    plots["loan_education"] = _save(fig, "loan_education.png")

    # 7. Property Area vs Approval
    fig, ax = plt.subplots(figsize=(6, 4))
    area_approval = df.groupby("Property_Area")["Loan_Status"].apply(lambda x: (x == "Y").mean())
    area_approval.plot(kind="bar", ax=ax, color=["#27ae60", "#e74c3c", "#2980b9"])
    ax.set_title("Approval Rate by Property Area")
    ax.set_ylabel("Approval Rate"); ax.set_ylim(0, 1)
    plots["loan_area"] = _save(fig, "loan_area.png")

    # 8. Correlation heatmap
    df_num = df.select_dtypes(include=np.number)
    df_num = df_num.copy()
    df_num["Loan_bin"] = (df["Loan_Status"] == "Y").astype(int)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_num.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Loan Dataset: Feature Correlation")
    plots["loan_corr"] = _save(fig, "loan_corr.png")

    return plots


if __name__ == "__main__":
    df_g = pd.read_csv("german_credit_risk.csv")
    df_l = pd.read_csv("loan_train_cleaned.csv")
    g_plots = german_eda(df_g)
    l_plots = loan_eda(df_l)
    print("German plots saved:", list(g_plots.keys()))
    print("Loan plots saved:  ", list(l_plots.keys()))
