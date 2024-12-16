# import license_execute as le
# license_status = le.check()
# if license_status == 'Valid@tAi':

from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer
from fairlearn.reductions import GridSearch, EqualizedOdds, ExponentiatedGradient

# Metrics
from fairlearn.metrics import (
    MetricFrame,
    selection_rate, demographic_parity_difference, demographic_parity_ratio,
    false_positive_rate, false_negative_rate,
    false_positive_rate_difference, false_negative_rate_difference,
    equalized_odds_difference, equalized_odds_ratio, count )
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, precision_score, accuracy_score, recall_score

import os
from configparser import ConfigParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from helper_functions.load import load_training_data, load_testing_data, load_test_preds, load_model_data



def get_bias_feature(y_true, y_pred, x_test):
    """
    Parameters : takes 3 parameter actual, prediction and test data
    Return : Equalized odds difference for each feature
    """

    bias_dic = {}
    min_bias_feature, min_bias_value = None, float('inf')
    max_bias_feature, max_bias_value = None, float('-inf')
    eodl, dpdl, minl, maxl, meanl = [], [], [], [], []
    for col in x_test.columns:
        dpd = demographic_parity_difference(y_true, y_pred, sensitive_features = x_test[col])
        eod = equalized_odds_difference(y_true, y_pred, sensitive_features = x_test[col])
        eodl.append(eod)
        dpdl.append(dpd)
        minl.append(min(eod, dpd))
        maxl.append(max(eod, dpd))
        avg = (dpd + eod) * 0.5
        meanl.append(avg)

        bias_measure = eod

        # Update min and max bias values and corresponding features
        if bias_measure < min_bias_value:
            min_bias_feature, min_bias_value = col, bias_measure
        if bias_measure > max_bias_value:
            max_bias_feature, max_bias_value = col, bias_measure


    # all_dict = {'Feature':x_test.columns, 'EOD':eodl, 'DPD':dpdl, 'Min':minl, 'Max':maxl, 'Average':meanl}
    score_df = pd.DataFrame({'Feature':x_test.columns, 'EOD':eodl, 'DPD':dpdl, 'Min':minl, 'Max':maxl, 'Average':meanl})
    score_df.to_json('results/Bias/bias_dict.json', orient='records')
    dic_avg = dict(zip(x_test.columns, meanl))
    min_key, min_value = min((k, v) for k, v in dic_avg.items())
    max_key, max_value = max((k, v) for k, v in dic_avg.items())
    #print(min_bias_feature,' ', min_bias_value,' ', max_bias_feature, ' ',max_bias_value)
    return score_df, dic_avg, min_bias_feature, min_bias_value, max_bias_feature, max_bias_value


def plot_bias_score(score_df, THRESHOLD):
    col_list = score_df.columns
    for ind in range(1, len(score_df.columns)):
        k = score_df['Feature']
        v = score_df[col_list[ind]]
        dic1 = dict(zip(k, v))
        dic1 = dict(sorted(dic1.items(), key=lambda item: item[1]))
        # print('dic1 = \n',dic1)

        selected_bar = None
        dropped_bar = None
        # colors = ["red" if score < THRESHOLD else "green" for score in dic1.values()]


        fig, ax = plt.subplots(figsize=(8, 4))

        for i, (feature, score) in enumerate(dic1.items()):
            params = dict(x=i, height=score, edgecolor="black", alpha=0.5)

            if score < THRESHOLD:
                bar = ax.bar(color="green", **params)
                if not dropped_bar:
                    dropped_bar = bar[0]
            else:
                bar = ax.bar(color="red", **params)
                if not selected_bar:
                    selected_bar = bar[0]

        thresh_line = ax.axhline(y=THRESHOLD, color="black", linestyle="--")
        ax.set_xticks(range(len(dic1)))
        ax.set_xticklabels(list(dic1.keys()), rotation=30, ha="right")
        ax.set(xlabel="Feature", ylabel="Score", title=f"Bias Score of Each Attribute {col_list[ind]}")
        ax.legend(handles=[selected_bar, dropped_bar, thresh_line], labels=["High Biased", "Biased", "Threshold"], loc="upper left")
        plt.tight_layout()

        plt.savefig(f'results/Bias/plots/bias_measure_plot({col_list[ind]}).jpg', dpi=300)
        plt.clf()

def plot_bias_measure(bias_score, max_bias_feature, THRESHOLD, plot_name):

    k = bias_score.keys()
    v = bias_score.values()
    dic1 = dict(zip(k, v))
    dic1 = dict(sorted(dic1.items(), key=lambda item: item[1]))
    # print('dic1 = \n',dic1)

    selected_bar = None
    dropped_bar = None
    # colors = ["red" if score < THRESHOLD else "green" for score in dic1.values()]


    fig, ax = plt.subplots(figsize=(8, 4))

    for i, (feature, score) in enumerate(dic1.items()):
        params = dict(x=i, height=score, edgecolor="black", alpha=0.5)

        if score < THRESHOLD:
            bar = ax.bar(color="green", **params)
            if not dropped_bar:
                dropped_bar = bar[0]
        else:
            bar = ax.bar(color="red", **params)
            if not selected_bar:
                selected_bar = bar[0]

    thresh_line = ax.axhline(y=THRESHOLD, color="black", linestyle="--")
    ax.set_xticks(range(len(dic1)))
    ax.set_xticklabels(list(dic1.keys()), rotation=30, ha="right")
    ax.set(xlabel="Feature", ylabel="Score", title=f"Bias Score of Each Attribute")
    ax.legend(handles=[selected_bar, dropped_bar, thresh_line], labels=["High Biased", "Biased", "Threshold"], loc="upper left")
    plt.tight_layout()

    # plt.savefig(f'results/Bias/plots/bias_measure_plot({col_list[ind]}).jpg', dpi=300)
    save_path = 'results/Bias/plots' if plot_name == 'bias_measure_plot' else 'results/debias/plots'
    plt.savefig(f'{save_path}/{plot_name}.jpg', dpi=100, bbox_inches='tight') 
    
    
    plt.clf()


def selection_bar_plot(x_test,y_test, test_pred, sensitive_feature, max_bias_feature, title):
    """
        plot selection rate of biased feature
    """
    metrics = {
        "selection rate": selection_rate
    }

    mf = MetricFrame(
        metrics=metrics, y_true=y_test, y_pred=test_pred, sensitive_features = sensitive_feature)

    mfdf = mf.by_group
    #labels = list(sensitive_feature.value_counts().keys())
    labels = list(x_test[max_bias_feature].value_counts().keys())
    labels = [str(x) for x in labels]

    color = None
    if title[0:6].lower() == 'before':
        # Update colors with red for the highest value
        color = 'red'
    if title[0:5].lower() == 'after':
        # Update colors with green for the highest value
        color = 'green'
    # Plot bar chart
    plt.bar(labels, mfdf['selection rate'], color=color)
    plt.xticks(rotation=90)
    plt.ylabel('Values')
    plt.title(title+' in '+max_bias_feature)
    save_path = 'results/Bias/plots' if title == 'before_debias' else 'results/debias/plots'
    plt.savefig(f'{save_path}/{title}.jpg', dpi=100, bbox_inches='tight')  # Corrected save path
    plt.clf()
    return mf



# Helper functions
def get_metrics_df(models_dict, y_true, group):
    metrics_dict = {
        "Overall selection rate": (
            lambda x: selection_rate(y_true, x), True),
        "Demographic parity difference": (
            lambda x: demographic_parity_difference(y_true, x, sensitive_features=group), True),
        "Demographic parity ratio": (
            lambda x: demographic_parity_ratio(y_true, x, sensitive_features=group), True),
        "Overall balanced error rate": (
            lambda x: 1-balanced_accuracy_score(y_true, x), True),
        "Balanced error rate difference": (
            lambda x: MetricFrame(metrics=balanced_accuracy_score, y_true=y_true, y_pred=x, sensitive_features=group).difference(method='between_groups'), True),
        "False positive rate difference": (
            lambda x: false_positive_rate_difference(y_true, x, sensitive_features=group), True),
        "False negative rate difference": (
            lambda x: false_negative_rate_difference(y_true, x, sensitive_features=group), True),
        "Equalized odds difference": (
            lambda x: equalized_odds_difference(y_true, x, sensitive_features=group), True),
        "Equalized odds ratio": (
            lambda x: equalized_odds_ratio(y_true, x, sensitive_features=group), True),
        "AUC difference": (
            lambda x: roc_auc_score(y_true, x), False),
        # "AUC difference": (
        #     lambda x: MetricFrame(metrics=roc_auc_score, y_true=y_true, y_pred=x, sensitive_features=group).difference(method='between_groups'), False),
    }
    df_dict = {}
    for metric_name, (metric_func, use_preds) in metrics_dict.items():
        df_dict[metric_name] = [metric_func(preds) if use_preds else metric_func(scores)
                                for model_name, (preds, scores) in models_dict.items()]
    # print(metrics_dict)
    return pd.DataFrame.from_dict(df_dict, orient="index", columns=models_dict.keys())


def debias_thresholdoptimizer(model, x_train, x_test, y_train, y_test,sensitive_feature_train, sensitive_feature_test, max_bias_feature, models_dict, categorical_columns, THRESHOLD):    

    postprocess_est = ThresholdOptimizer(
        estimator=model,
        constraints="equalized_odds",
        prefit=False)
    
    postprocess_est.fit(x_train, y_train, sensitive_features = sensitive_feature_train)
    
    postprocess_preds = postprocess_est.predict(x_test, sensitive_features = sensitive_feature_test)
    print(postprocess_est,'postprocess_est')
    print(type(postprocess_est))
    # efficacy
    # model_df_th = metrics_dataframe(postprocess_preds, y_test)
    save_path = 'results/debias/'
    filename = 'postprocess_est.pkl'
    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(postprocess_est, f)
    


    mft  = selection_bar_plot(x_test,y_test, postprocess_preds, sensitive_feature_test, max_bias_feature, title='After_ThresholdOptimizer')
    _, bias_dic , _, _, max_bias_feature, _ = get_bias_feature(y_test, postprocess_preds, x_test[categorical_columns])
    bias_dic = dict(sorted(bias_dic.items(), key=lambda item: item[1]))
    plot_name = 'debias_threshold_plot'
    plot_bias_measure(bias_dic, max_bias_feature, THRESHOLD, plot_name)

    threshold_model_dict = {"After ThresholdOptimizer": (postprocess_preds, postprocess_preds)}
    models_dict.update(threshold_model_dict)
    get_metrics = get_metrics_df(models_dict, y_test, sensitive_feature_test)

    return mft, get_metrics


def debias_exponentiatedgradient(model, x_train, x_test, y_train, y_test,sensitive_feature_train, sensitive_feature_test, max_bias_feature, models_dict, categorical_columns, THRESHOLD):    

    expgrad_est = ExponentiatedGradient(
            estimator=model,
            constraints= EqualizedOdds()
            )
    
    expgrad_est.fit(x_train, y_train, sensitive_features = sensitive_feature_train)
    
    expgrad_preds = expgrad_est.predict(x_test)
    print(expgrad_est,'Expgrad')
    print(type(expgrad_est))
    # efficacy
    # model_df_th = metrics_dataframe(postprocess_preds, y_test)
    save_path = 'results/debias/'
    filename = 'expgrad_est.pkl'
    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(expgrad_est, f)
    



    mfe  = selection_bar_plot(x_test,y_test, expgrad_preds, sensitive_feature_test, max_bias_feature, title='After_ExponentiatedGradient')
    _, bias_dic , _, _, max_bias_feature, _ = get_bias_feature(y_test, expgrad_preds, x_test[categorical_columns])
    bias_dic = dict(sorted(bias_dic.items(), key=lambda item: item[1]))
    plot_name = 'debias_exponent_plot'
    plot_bias_measure(bias_dic, max_bias_feature, THRESHOLD, plot_name)

    exponent_model_dict = {"After ExponentiatedGradient": (expgrad_preds, expgrad_preds)}
    models_dict.update(exponent_model_dict)
    get_metrics = get_metrics_df(models_dict, y_test, sensitive_feature_test)

    return mfe, get_metrics


def debias_gridsearch(model, x_train, x_test, y_train, y_test,sensitive_feature_train, sensitive_feature_test, max_bias_feature, models_dict, categorical_columns, THRESHOLD):

    # Train GridSearch
    grid = GridSearch(model,
                    constraints=EqualizedOdds(),
                    grid_size=10,
                    grid_limit=3)

    grid.fit(x_train, y_train, sensitive_features= sensitive_feature_train)


    # Get the best model
    best_model = grid.best_estimator_
    print(best_model)
    # Evaluate the best model
    y_pred = best_model.predict(x_test)

    

    grid_preds = [predictor.predict(x_test) for predictor in grid.predictors_]
    grid_scores = [predictor.predict_proba(x_test)[:, 1] for predictor in grid.predictors_]

    equalized_odds_grid = [
        equalized_odds_difference(y_test, preds, sensitive_features= sensitive_feature_test)
        for preds in grid_preds
    ]

    balanced_accuracy_grid = [balanced_accuracy_score(y_test, preds) for preds in grid_preds]
    auc_grid = [roc_auc_score(y_test, scores) for scores in grid_scores]

    # Select only non-dominated models (with respect to balanced accuracy and equalized odds difference)
    all_results = pd.DataFrame(
        {"predictor": grid.predictors_, "accuracy": balanced_accuracy_grid, "disparity": equalized_odds_grid}
    )
    non_dominated = []
    for row in all_results.itertuples():
        accuracy_for_lower_or_eq_disparity = all_results["accuracy"][all_results["disparity"] <= row.disparity]
        if row.accuracy >= accuracy_for_lower_or_eq_disparity.max():
            non_dominated.append(True)
        else:
            non_dominated.append(False)

    equalized_odds_sweep_non_dominated = np.asarray(equalized_odds_grid)[non_dominated]
    balanced_accuracy_non_dominated = np.asarray(balanced_accuracy_grid)[non_dominated]
    auc_non_dominated = np.asarray(auc_grid)[non_dominated]

    # Compare GridSearch models with low values of equalized odds difference with the previously constructed models
    grid_search_dict = {"After GridSearch_{}".format(i): (grid_preds[i], grid_scores[i])
                        for i in range(len(grid_preds))
                        if non_dominated[i] and equalized_odds_grid[i]<0.5}

    models_dict.update(grid_search_dict)

    result = get_metrics_df(models_dict, y_test, sensitive_feature_test)
    mfg  = selection_bar_plot(x_test,y_test, grid_preds[int(result.columns[3].split('_')[1])], sensitive_feature_test,max_bias_feature, title='After_Grid')
    return result


def main():
    parser = ConfigParser()
    parser.read('config/debias.properties')
    threshold = float(parser['parameters']['threshold'])

    os.makedirs('results/Bias/plots', exist_ok=True)
    os.makedirs('results/Bias/plots', exist_ok=True)
    os.makedirs('results/debias/plots', exist_ok=True)


    x_train, _ ,y_train = load_training_data()
    if not isinstance(y_train, pd.Series):
        y_train = y_train.iloc[:, 0]

    x_test, _ ,y_test = load_testing_data() 
    if not isinstance(y_test, pd.Series):
        y_test = y_test.iloc[:, 0]
    test_preds, test_scores = load_test_preds()
    model = load_model_data()

    categorical_columns = [col for col in x_test.columns
                            if x_test[col].dtype == 'object' or len(x_test[col].unique()) <= 5]

    if len(categorical_columns) > 0:
        # calling bias function
        bias_df, bias_dic , min_bias_feature, min_bias_value, max_bias_feature, max_bias_value = get_bias_feature(y_test, test_preds, x_test[categorical_columns])
        #print(min_bias_feature, ' ',min_bias_value,' ' ,max_bias_feature,' ' ,max_bias_value)
        # Sort the dictionary by values
        bias_dic = dict(sorted(bias_dic.items(), key=lambda item: item[1]))
        #print('bias_dic', bias_dic)

        sensitive_feature_test = x_test[max_bias_feature]
        sensitive_feature_train = x_train[max_bias_feature]
        #print('======================\n')
        #print()

        if parser['bias_flag']['bias_measure'].lower() == 'true': 
            # calling selection rate bar plot function
            selection_bar_plot(x_test,y_test, test_preds, sensitive_feature_test, max_bias_feature, title='before_debias')
            plot_name = 'bias_measure_plot'
            plot_bias_score(bias_df, threshold)


        models_dict = {"Before Debias": (test_preds, test_scores)}
        get_metrics = get_metrics_df(models_dict, y_test, sensitive_feature_test)
        
        if parser['debias_flag']['thresholdoptimizer'].lower() == 'true': 
            # calling debias_threshold function
            print('\nthreshold start\n')
            mft, get_metrics = debias_thresholdoptimizer(model, x_train, x_test, y_train, y_test,sensitive_feature_train, sensitive_feature_test, max_bias_feature, models_dict, categorical_columns, threshold)

        
        if parser['debias_flag']['exponentiatedgradient'].lower() == 'true': 
            # calling debias_threshold function
            print('\nexponentiated start\n')
            mft, get_metrics = debias_exponentiatedgradient(model, x_train, x_test, y_train, y_test,sensitive_feature_train, sensitive_feature_test, max_bias_feature, models_dict, categorical_columns, threshold)
            

        if parser['debias_flag']['gridsearch'].lower() == 'true': 
            # calling debias_gridsearch function
            print('\ngridsearch start\n')
            get_metrics = debias_gridsearch(model, x_train, x_test, y_train, y_test,sensitive_feature_train, sensitive_feature_test, max_bias_feature, models_dict, categorical_columns, threshold)
        
        print(get_metrics)
        get_metrics.index.name = "Metrics"
        get_metrics.to_csv('results/debias/bias_scores.csv')
        bias_rs = pd.read_csv('results/debias/bias_scores.csv')
        # Convert DataFrame to JSON
        bias_json_data = bias_rs.to_json(orient='records')

        # Save JSON data to a file
        with open('results/debias/bias_scores.json', 'w') as file:
            file.write(bias_json_data)
    else :
        print("No categorical Column has found in Dataset")

# else:
#     print("Invalid License")

