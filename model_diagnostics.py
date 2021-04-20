# Ensemble prediction ----
def ensemble_prediction(clfs, model_stats, X):
    import numpy as np
    import pandas as pd
    
    clf_votes = pd.DataFrame({'Model':model_stats.Model,
                              'NumVotes':np.round(model_stats.TestF1*100)})
    
    pred_votes = pd.DataFrame({
        'Row':np.arange(X.shape[0]),
        'Votes_0':np.repeat(0, X.shape[0]),
        'Votes_1':np.repeat(0, X.shape[0])
    })
    for clf_num in range(len(clfs)):
        clf = clfs[clf_num]
        preds = clf.predict(X)
        for row in range(len(preds)):
            if preds[row] == 0:
                pred_votes.iloc[row,1] = pred_votes.iloc[row,1] + clf_votes.iloc[clf_num,1]
            if preds[row] == 1:
                pred_votes.iloc[row,2] = pred_votes.iloc[row,2] + clf_votes.iloc[clf_num,1]
    
    pred_votes['FinalPred'] = pred_votes['Votes_1'] > pred_votes['Votes_0']
    pred_votes['FinalPred'] = pred_votes['FinalPred'].map({False:0, True:1})
    
    return pred_votes, pred_votes.FinalPred.values

# Partial dependence ----
def partial_dependence(clf, df_X, column, scalers, num_test = 30, show_plot = True):
    """
    Purpose:
        Partial dependence analysis is a way to evaluate how the level of variables impact predictions from a predictive model. To do this, it takes a data frame column and substitutes the column's values with one value. It then makes predictions on this data frame with this substituted column and averages the predictions. Then, a new value is tested in column and the process is repeated to see how changing the value in the column impact predictions.
        
        This function takes the range of values in the data frame and creates a sequence across this range for the length specified in num_test. Then, all these values are tested for the input model and saved into a data frame.
        
    Inputs:
    * clf = predictive model object
    * df_X = data frame with columns for model (don't include y column)
    * column = column to calculate partial dependence values for
    * scalers = scalers dictionary needed to transform original form of data frame
    * num_test = how many values to test for this column (from min to max of actual values)
    * show_plot = whether to show a partial dependence plot
    
    Output:
    * avg_preds = data frame with the average prediction for a given value in the column
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    unique_val = df[column].unique()
    
    if len(unique_val) > num_test:
        min_val = unique_val.min()
        max_val = unique_val.max()
        step = (max_val - min_val)/num_test
        val_range = np.arange(min_val, max_val, step)
    else:
        val_range = np.sort(unique_val)
        
    df_copy = df_X.copy()
    avg_preds = {'Value':[], 'AvgPred':[]}
    for val in val_range:
        df_copy[column] = val
        df_copy_scl = scalers['X'].transform(df_copy)
        preds = clf.predict(df_copy_scl)
        avg_pred = preds.mean()
        avg_preds['Value'].append(val)
        avg_preds['AvgPred'].append(avg_pred)
        
    avg_preds = pd.DataFrame(avg_preds)
    avg_preds['column'] = column
    
    if show_plot:
        plt.plot(avg_preds.Value, avg_preds.AvgPred)
        plt.xlabel(column)
        plt.title(f'Partial Dependence Plot for {column}')
        plt.show()
    
    return avg_preds

def partial_dependence_loop(clf, df_X, scalers, num_test, show_plot):
    """
    Purpose:
        This function loops through all the columns in df_X, performing partial dependence analysis on each.
        
    Inputs:
    * clf = predictive model object
    * df_X = data frame with columns for model (don't include y column)
    * scalers = scalers dictionary needed to transform original form of data frame
    * num_test = how many values to test for this column (from min to max of actual values)
    * show_plot = whether to show a partial dependence plot
    
    Output:
    * partial_dependences_df = data frame with test values and average prediction for each column
    * partial_dependences_stats = min, max, and range of average predictions for each column
    """
    import pandas as pd
    
    partial_dependences = []
    counter = 1
    for col in df_X.columns:
        part_dep = partial_dependence(clf = nn_clf, df_X = df_X, column = col,
                                      scalers = scalers, num_test = num_test, show_plot = show_plot)
        partial_dependences.append(part_dep)
        print(f'Column {counter}/{len(df_X.columns)} complete')
        counter += 1

    partial_dependences_df = pd.concat(partial_dependences)


    partial_dependences_stats = partial_dependences_df\
        .groupby('column')['AvgPred']\
        .agg(['min', 'max'])\
        .reset_index()
    partial_dependences_stats['range'] = partial_dependences_stats['max'] - partial_dependences_stats['min']

    return partial_dependences_df, partial_dependences_stats

def plot_top_partial_dependences(partial_dependences_df, partial_dependences_stats, top_n):
    """
    Purpose:
        Plots the top n partial dependences in terms of the largest ranges of average prediction (biggest differences across values of a column)
        
    Inputs:
    * partial_dependences_df = data frame with test values and average prediction for each column
    * partial_dependences_stats = min, max, and range of average predictions for each column
    * top_n = how many top columns to plot
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    biggest_ranges = partial_dependences_stats\
        .sort_values('range', ascending = False)\
        .head(top_n)
    print(biggest_ranges)

    for col in biggest_ranges.column:
        col_part_dep = partial_dependences_df.loc[partial_dependences_df.column == col]
        plt.plot(col_part_dep.Value, col_part_dep.AvgPred)
        plt.xlabel(f'{col} Test Value')
        plt.title(f'Partial Dependence Plot for {col}')
        plt.show()