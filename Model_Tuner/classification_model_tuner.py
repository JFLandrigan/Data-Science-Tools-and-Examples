import Model_Tuner.tuner_utils as tu
import Model_Tuner.plot_utils as pu

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import numpy as np

import time


def tune_test_model(
    X=None,
    y=None,
    model=None,
    params={},
    tune_metric="f1",
    eval_metrics=["f1"],
    num_cv=5,
    pipe=None,
    scale=None,
    select_features=None,
    bins=None,
    num_top_fts=None,
    tuner="random_cv",
    n_iterations=15,
    get_ft_imp=True,
    n_jobs=6,
    random_seed=None,
    binary=True,
    log=True,
    log_name=None,
    log_path=None,
    log_note=None,
):

    if random_seed is None:
        random_seed = np.random.randint(1000, size=1)[0]
        print("Random Seed Value: " + str(random_seed))

    if select_features:
        print("Selecting features")

        sub_fts, drop_fts = tu.select_features(
            X=X,
            y=y,
            methods=select_features["methods"],
            problem_type="clf",
            model_pipe=select_features["model_pipe"],
            imp_thresh=select_features["imp_thresh"],
            corr_thresh=select_features["corr_thresh"],
            bin_fts=select_features["bin_fts"],
            dont_drop=select_features["dont_drop"],
            random_seed=random_seed,
            n_jobs=n_jobs,
            plot_ft_importance=select_features["plot_ft_importance"],
            plot_ft_corr=select_features["plot_ft_corr"],
        )

        X = X[sub_fts].copy(deep=True)

        features = sub_fts.copy()
    else:
        features = list(X.columns)

    if model == "rf":
        mod_scv = RandomForestClassifier(random_state=random_seed)
    elif model == "gbc":
        mod_scv = GradientBoostingClassifier(random_state=random_seed)
    elif model == "dt":
        mod_scv = DecisionTreeClassifier(random_state=random_seed)
    elif model == "lr":
        mod_scv = LogisticRegression(random_state=random_seed)
    elif model == "svm":
        mod_scv = SVC(random_state=random_seed)
    elif model == "knn":
        mod_scv = KNeighborsClassifier(random_state=random_seed)
    elif model == "nn":
        mod_scv = MLPClassifier(random_state=random_seed)
    elif model == "ada":
        mod_scv = AdaBoostClassifier(random_state=random_seed)
    else:
        if model is None:
            print("NO MODEL PASSED IN")
        else:
            mod_scv = model

    if pipe and scale:
        print("ERROR CAN'T PASS IN PIPE OBJECT AND ALSO SCALE ARG")
        return

    if pipe:
        tmp_mod_scv = pipe
        tmp_mod_scv.steps.append(["clf", mod_scv])
        mod_scv = tmp_mod_scv
        params = {k if "clf__" in k else "clf__" + k: v for k, v in params.items()}

    elif scale:
        if scale == "standard":
            mod_scv = Pipeline([("scale", StandardScaler()), ("clf", mod_scv)])
            params = {k if "clf__" in k else "clf__" + k: v for k, v in params.items()}
        elif scale == "minmax":
            mod_scv = Pipeline([("scale", MinMaxScaler()), ("clf", mod_scv)])
            params = {k if "clf__" in k else "clf__" + k: v for k, v in params.items()}

    if tuner == "random_cv":
        scv = RandomizedSearchCV(
            mod_scv,
            param_distributions=params,
            n_iter=n_iterations,
            scoring=tune_metric,
            cv=num_cv,
            n_jobs=n_jobs,
            verbose=2,
            random_state=random_seed,
        )

    elif tuner == "bayes_cv":
        scv = BayesSearchCV(
            estimator=mod_scv,
            search_spaces=params,
            n_iter=n_iterations,
            cv=num_cv,
            verbose=2,
            refit=True,
            n_jobs=n_jobs,
        )

    elif tuner == "grid_cv":
        scv = GridSearchCV(
            mod_scv,
            param_grid=params,
            scoring=tune_metric,
            cv=num_cv,
            n_jobs=n_jobs,
            verbose=2,
        )

    scv.fit(X, y)

    mod = scv.best_estimator_
    params = mod.get_params()
    print("Parameters of the best model: \n")
    print(mod.get_params())
    print("\n")

    print("Performing model eval on best estimator")

    log_data = model_eval(
        X=X,
        y=y,
        model=mod,
        params={},
        metrics=eval_metrics,
        bins=bins,
        pipe=pipe,
        scale=scale,
        num_top_fts=num_top_fts,
        num_cv=num_cv,
        get_ft_imp=get_ft_imp,
        random_seed=random_seed,
        binary=binary,
        log=log,
        log_name=log_name,
        log_path=log_path,
        tune_test=True,
    )

    if log:
        log_data["test_params"] = params
        log_data["tune_metric"] = tune_metric
        if log_note:
            log_data["note"] = log_note

        tu.log_results(
            fl_name=log_name, fl_path=log_path, log_data=log_data, tune_test=True
        )

    return [mod, params, features]


def model_eval(
    X=None,
    y=None,
    model=None,
    params={},
    metrics=["f1"],
    bins=None,
    pipe=None,
    scale=None,
    num_top_fts=None,
    num_cv=5,
    get_ft_imp=False,
    random_seed=None,
    binary=True,
    log=True,
    log_name=None,
    log_path=None,
    log_note=None,
    tune_test=False,
):
    """
    Model Eval function. Used to perform cross validation on model and is automatically called post tune_test_model
    :param X: pandas dataframe containing features for model training
    :param y: series or np array containing prediction values
    :param model: Model object containing fit, predict, predict_proba attributes, sklearn pipeline object or string indicator of model to eval
    :param params: dictionary containing parameters of model to fit on
    :param metrics: list of metrics to eval model on default is ['f1]
    :param bins: list of bin ranges to output the score to percent actual distribution
    :param pipe: Sklearn pipeline object without classifier 
    :param scale: string Standard or MinMax indicating to scale the features during cross validation
    :param num_top_fts: int number of top features to be plotted
    :param num_cv: int number of cross validations to do
    :param get_ft_imp: boolean indicating to get and plot the feature importances
    :param random_seed: int for random seed setting
    :param binary: boolean indicating if model predictions are binary or multi-class
    :param log: boolean indicator to log out results
    :param log_name: string name of the logger doc
    :param log_path: string path to store logger doc if none data dir in model tuner dir is used
    :param log_note: string containing note to add at top of logger doc
    :param tune_test:
    :return:
    """

    if random_seed is None:
        random_seed = np.random.randint(1000, size=1)[0]
    print("Random Seed Value: " + str(random_seed))

    if model == "rf":
        mod = RandomForestClassifier(**params)
    elif model == "gbc":
        mod = GradientBoostingClassifier(**params)
    elif model == "dt":
        mod = DecisionTreeClassifier(**params)
    elif model == "lr":
        mod = LogisticRegression(**params)
    elif model == "svm":
        mod = SVC(**params)
    elif model == "knn":
        mod = KNeighborsClassifier(**params)
    elif model == "nn":
        mod = MLPClassifier(**params)
    elif model == "ada":
        mod = AdaBoostClassifier(**params)
    else:
        if model is None:
            print("NO MODEL PASSED IN")
            return
        else:
            mod = model

    print("Performing CV Runs: " + str(num_cv))
    kf = KFold(n_splits=num_cv, shuffle=True, random_state=random_seed)

    if binary:
        avg = "binary"
    else:
        avg = "macro"

    metric_dictionary = tu.init_model_metrics(metrics=metrics)

    cnt = 1
    for train_index, test_index in kf.split(X):
        cv_st = time.time()

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if pipe and tune_test == False:
            tmp_mod = pipe
            tmp_mod.steps.append(["clf", mod])
            mod = tmp_mod
            params = {"clf__" + k: v for k, v in params.items()}

        elif scale and tune_test == False:
            if scale == "standard":
                scaler = StandardScaler()
            elif scale == "minmax":
                scaler = MinMaxScaler()

            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        mod.fit(X_train, y_train)
        preds = mod.predict(X_test)
        pred_probs = mod.predict_proba(X_test)[:, 1]

        for metric in metrics:
            if metric == "accuracy":
                metric_dictionary[metric + "_scores"] = np.append(
                    metric_dictionary[metric + "_scores"],
                    metric_dictionary[metric + "_func"](y_test, preds),
                )
            elif metric in ["f1", "precision", "recall"]:
                metric_dictionary[metric + "_scores"] = np.append(
                    metric_dictionary[metric + "_scores"],
                    metric_dictionary[metric + "_func"](y_test, preds, average=avg),
                )
            elif metric in ["roc_auc", "precision_recall_auc"]:
                metric_dictionary[metric + "_scores"] = np.append(
                    metric_dictionary[metric + "_scores"],
                    metric_dictionary[metric + "_func"](y_test, pred_probs),
                )

        print(
            "Finished cv run: "
            + str(cnt)
            + " time: "
            + str(time.time() - cv_st)
            + " \n \n"
        )
        cnt += 1

    print("CV Run Scores")
    for metric in metrics:
        print(metric + " scores: " + str(metric_dictionary[metric + "_scores"]))
        print(metric + " mean: " + str(metric_dictionary[metric + "_scores"].mean()))
        print(
            metric
            + " standard deviation: "
            + str(metric_dictionary[metric + "_scores"].std())
            + " \n"
        )

    print(" \n")

    print("Final cv train test split")
    for metric in metrics:
        print(metric + " score: " + str(metric_dictionary[metric + "_scores"][-1]))

    print(" \n")
    cf = confusion_matrix(y_test, preds)
    cr = classification_report(
        y_test, preds, target_names=[str(x) for x in mod.classes_]
    )

    pu.plot_confusion_matrix(cf=cf, labels=mod.classes_)
    print(cr)

    if binary:
        prob_df = pd.DataFrame({"probab": pred_probs, "actual": y_test})
        bt = tu.create_bin_table(
            df=prob_df, bins=bins, bin_col="probab", actual_col="actual"
        )
        print(bt)

    if "roc_auc" in metrics:
        pu.plot_roc_curve(y_true=y_test, pred_probs=pred_probs)
    if "precision_recall_auc" in metrics:
        pu.plot_precision_recall_curve(y_true=y_test, pred_probs=pred_probs)

    if get_ft_imp:
        ft_imp_df = tu.feature_importances(mod=mod, X=X, num_top_fts=num_top_fts)

    if log:
        log_data = {
            "features": list(X.columns),
            "random_seed": random_seed,
            "params": mod.get_params(),
            "metrics": metric_dictionary,
            "cf": cf,
            "cr": cr,
        }

        if type(mod).__name__ == "Pipeline":
            log_data["model"] = type(mod).__name__
            pipe_steps = "Pipe steps: "
            for k in mod.named_steps.keys():
                pipe_steps = pipe_steps + type(mod.named_steps[k]).__name__ + " "
            log_data["pipe_steps"] = pipe_steps
        else:
            log_data["model"] = type(mod).__name__

        if log_note:
            log_data["note"] = log_note

        if binary:
            log_data["bin_table"] = bt

        if get_ft_imp:
            log_data["ft_imp_df"] = ft_imp_df

        if tune_test:
            return log_data
        else:
            tu.log_results(
                fl_name=log_name, fl_path=log_path, log_data=log_data, tune_test=False
            )

    return
