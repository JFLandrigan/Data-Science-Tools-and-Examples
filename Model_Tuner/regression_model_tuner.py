import Model_Tuner.tuner_utils as tu
import Model_Tuner.plot_utils as pu

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

from sklearn.model_selection import KFold

import numpy as np

import time


def tune_test_model(
    X=None,
    y=None,
    model=None,
    params={},
    tune_metric="neg_mean_squared_error",
    eval_metrics=["mse"],
    num_cv=5,
    pipe=None,
    scale=None,
    select_features=None,
    num_top_fts=None,
    tuner="random_cv",
    n_iterations=15,
    get_ft_imp=True,
    n_jobs=6,
    random_seed=None,
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
            problem_type="rgr",
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
        mod_scv = RandomForestRegressor(random_state=random_seed)
    elif model == "gbc":
        mod_scv = GradientBoostingRegressor(random_state=random_seed)
    elif model == "dt":
        mod_scv = DecisionTreeRegressor(random_state=random_seed)
    elif model == "lr":
        mod_scv = LinearRegression()
    elif model == "lasso":
        mod_scv = Lasso(random_state=random_seed)
    elif model == "elastic":
        mod_scv = ElasticNet(random_state=random_seed)
    elif model == "svm":
        mod_scv = SVR()
    elif model == "knn":
        mod_scv = KNeighborsRegressor(random_state=random_seed)
    else:
        print("Model not supported")
        return

    if pipe and scale:
        print("ERROR CAN'T PASS IN PIPE OBJECT AND ALSO SCALE ARG")
        return

    if pipe:
        tmp_mod_scv = pipe
        tmp_mod_scv.steps.append(["rgr", mod_scv])
        mod_scv = tmp_mod_scv
        params = {"rgr__" + k: v for k, v in params.items()}

    elif scale:
        if scale == "standard":
            mod_scv = Pipeline([("scale", StandardScaler()), ("rgr", mod_scv)])
            params = {"rgr__" + k: v for k, v in params.items()}
        elif scale == "minmax":
            mod_scv = Pipeline([("scale", MinMaxScaler()), ("rgr", mod_scv)])
            params = {"rgr__" + k: v for k, v in params.items()}

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
        pipe=pipe,
        scale=scale,
        num_top_fts=num_top_fts,
        num_cv=num_cv,
        get_ft_imp=get_ft_imp,
        random_seed=random_seed,
        log=log,
        log_name=log_name,
        log_path=log_path,
        tune_test=True,
    )

    if log:
        if type(mod).__name__ == "Pipeline":
            log_data["model"] = type(mod).__name__
            pipe_steps = "Pipe steps: "
            for k in mod.named_steps.keys():
                pipe_steps = pipe_steps + type(mod.named_steps[k]).__name__ + " "
            log_data["pipe_steps"] = pipe_steps
        else:
            log_data["model"] = type(mod).__name__

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
    metrics=["mse"],
    pipe=None,
    scale=None,
    num_top_fts=None,
    num_cv=5,
    get_ft_imp=False,
    random_seed=None,
    log=True,
    log_name=None,
    log_path=None,
    log_note=None,
    tune_test=False,
):

    if random_seed is None:
        random_seed = np.random.randint(1000, size=1)[0]
    print("Random Seed Value: " + str(random_seed))

    if model == "rf":
        mod = RandomForestRegressor(**params)
    elif model == "gbc":
        mod = GradientBoostingRegressor(**params)
    elif model == "dt":
        mod = DecisionTreeRegressor(**params)
    elif model == "lr":
        mod = LinearRegression(**params)
    elif model == "lasso":
        mod = Lasso(**params)
    elif model == "elastic":
        mod = ElasticNet(**params)
    elif model == "svm":
        mod = SVR(**params)
    elif model == "knn":
        mod = KNeighborsRegressor(**params)
    else:
        mod = model

    metric_dictionary = tu.init_model_metrics(metrics=metrics)

    print("Performing CV Runs: " + str(num_cv))
    kf = KFold(n_splits=num_cv, shuffle=True, random_state=random_seed)

    cnt = 1
    for train_index, test_index in kf.split(X):
        cv_st = time.time()

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if pipe and tune_test == False:
            tmp_mod = pipe
            tmp_mod.steps.append(["rgr", mod])
            mod = tmp_mod
            params = {"rgr__" + k: v for k, v in params.items()}

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

        for metric in metrics:
            metric_dictionary[metric + "_scores"] = np.append(
                metric_dictionary[metric + "_scores"],
                metric_dictionary[metric + "_func"](y_test, preds),
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

    pu.plot_true_pred_scatter(y_test, preds)

    if get_ft_imp:
        ft_imp_df = tu.feature_importances(mod=mod, X=X, num_top_fts=num_top_fts)

    if log:
        log_data = {
            "features": list(X.columns),
            "random_seed": random_seed,
            "params": mod.get_params(),
            "metrics": metric_dictionary,
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

        if get_ft_imp:
            log_data["ft_imp_df"] = ft_imp_df

        if tune_test:
            return log_data
        else:
            tu.log_results(
                fl_name=log_name, fl_path=log_path, log_data=log_data, tune_test=False
            )

    return
