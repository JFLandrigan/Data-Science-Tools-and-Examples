import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_importance(ft_df=None, mod_type=None, num_top_fts=None):
    if num_top_fts:
        ft_df = ft_df.head(num_top_fts).copy(deep=True)

    if (
        ("RandomForest" in mod_type)
        or ("GradientBoosting" in mod_type)
        or ("DecisionTree" in mod_type)
    ):
        plt.figure(figsize=(10, 10), dpi=80, facecolor="w", edgecolor="k")
        sns.barplot(x="Importance", y="Feature", data=ft_df)

    elif (
        ("Regression" in mod_type)
        or (mod_type == "Lasso")
        or (mod_type == "ElasticNet")
    ):
        plt.figure(figsize=(10, 10), dpi=80, facecolor="w", edgecolor="k")
        sns.barplot(x="Coef", y="Feature", data=ft_df)

    return


def plot_feature_correlations(df=None):
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )


def plot_confusion_matrix(cf=None, labels=None):

    cf_df = pd.DataFrame(cf, index=labels, columns=labels)
    cf_df = cf_df.div(cf_df.sum(axis=1), axis=0)

    plt.figure(figsize=(12, 12))
    sns.heatmap(cf_df, annot=True, cmap="coolwarm")

    return


def plot_roc_curve(y_true, pred_probs):
    ns_probs = [0 for _ in range(len(y_true))]
    ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probs)
    clf_fpr, clf_tpr, _ = roc_curve(y_true, pred_probs)

    plt.figure(figsize=(8, 8))
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
    plt.plot(clf_fpr, clf_tpr, marker=".", label="Classifier")
    # axis labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    return


def plot_precision_recall_curve(y_true, pred_probs):
    clf_precision, clf_recall, _ = precision_recall_curve(y_true, pred_probs)

    # plot the precision-recall curves
    no_skill = len(y_true[y_true == 1]) / len(y_true)

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
    plt.plot(clf_recall, clf_precision, marker=".", label="Classifier")
    # axis labels
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    return


def plot_true_pred_scatter(y_true, y_pred):

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

    fig, axs = plt.subplots(nrows=2, figsize=(12, 12))
    sns.regplot(x="y_true", y="y_pred", data=df, ax=axs[0])
    sns.kdeplot(df["y_true"], bw=0.2, label="true", color="r", ax=axs[1])
    sns.kdeplot(df["y_pred"], bw=2, label="pred", color="b", ax=axs[1])

    return
