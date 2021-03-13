def weight(row):
    if row['bmi'] >= 30:
        val = 'obese'
    elif ((row['bmi'] >= 25) & (row['bmi'] < 30)):
        val = 'over weight'
    elif ((row['bmi'] >= 18.5) & (row['bmi'] < 25)):
        val = 'normal weight'
    else:
        val = 'under weight'
    return val

def stroke_pivot_graph(index, xlabel):
    fig, ax = plt.subplots(figsize=(7, 5))
    data.pivot_table(index=index, columns='stroke', aggfunc='size').plot(kind='bar', ax=ax, colormap=c_map);
    plt.ylabel('Count')
    plt.xlabel(xlabel)
    plt.title(f'Stroke Distribution by {xlabel}')
    plt.show()
    return None

def plot_partioned_graph(index):
    plt.figure(figsize=(16,4))
    ax1 = plt.subplot(1,3,1)
    sns.histplot(x=index, data=data, bins=30)
    plt.title(f'{index.capitalize()} Distribution (Total)')

    ax2 = plt.subplot(1,3,2)
    sns.histplot(x=index, data=data[data.stroke==1], bins=30)
    plt.title(f'{index.capitalize()} Distribution (Stroke)')

    ax2 = plt.subplot(1,3,3)
    sns.histplot(x=index, data=data[data.stroke==0], bins=30)
    plt.title(f'{index.capitalize()} Distribution (No Stroke)')

    plt.show();
    return None

def feature_ratio(feature):
    '''
    Create a pivot table for the number of strokes relative to the observations
    for a given feature. Ratio is returned as a percentage of the total subgroup
    that had a stroke.
    '''
    data_pivot = data.pivot_table(index=feature, columns='stroke', aggfunc='size')
    data_pivot['Ratio'] = round(data_pivot[1]/data_pivot[0], 4)*100
    return data_pivot.iloc[:,-1:]

def age(row):
    if row['child'] == 1:
        val = 'child'
    elif row['youth'] == 1:
        val = 'youth'
    elif row['adult'] == 1:
        val = 'adult'
    else:
        val = 'senior'
    return val

def plot_confusion_matrix(true_labels, y_preds, title, map_color):
    '''
    Plot an annotated confusion matrix to visualize the evaluation for
    our model predictions. Includes error labels and counts.
    '''
    cf_matrix = confusion_matrix(true_labels, y_preds)
    counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    labels = ['True Neg','False Pos','False Neg','True Pos']
    labels = [f'{label}\n{count}' for label,count in zip(labels, counts)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize=(5,4))
    sns.heatmap(cf_matrix, fmt='', annot=labels, cmap=map_color, linewidths=1, linecolor='black')
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show();
    return None

def fitModel(feature_subset):
    response_var = 'stroke ~ '
    explanatory_vars = ' + '.join(X_train[list(feature_subset)].columns.values)
    formula = response_var + explanatory_vars
    model = sm.GLM.from_formula(formula, family=sm.families.Binomial(), data=data_glm)
    result = model.fit()
    y_preds = round(result.predict(X_test[list(feature_subset)]))
    model_recall = recall_score(y_test, y_preds)
    model_fbeta = fbeta_score(y_test, y_preds, beta=0.85)
    return {'model': result, 'bic': result.bic, 'recall': model_recall, 
            'fbeta': model_fbeta, 'features' :X_train[list(feature_subset)].columns.values}

def forwardSelection(predictors):
    start = time.time()
    remaining_predictors = [p for p in X.columns if p not in predictors]
    results = []
    for p in remaining_predictors:
        results.append(fitModel(predictors+[p]))
    
    models = pd.DataFrame(results)
    best_model = models.sort_values(by='fbeta', ascending=False).iloc[0,:]
    print("Processed", models.shape[0], "models on", len(predictors)+1, "predictors in", round(time.time()-start,3), "seconds.")
    
    return best_model.values

def exhaustiveSearch(k):
    start = time.time()
    results = []
    for combo in itertools.combinations(X_train, k):
        results.append(fitModel(combo))
    
    models = pd.DataFrame(results)
    best_model = models.sort_values(by='fbeta', ascending=False).iloc[0,:]
    print("Processed", models.shape[0], "models on", k, "predictors in", round(time.time()-start,3), "seconds.")
    
    return best_model.values

def backward(predictors):
    start = time.time()
    results = []
    for combo in itertools.combinations(predictors, len(predictors)-1):
        results.append(fitModel(combo))
    
    models = pd.DataFrame(results)
    best_model = models.sort_values(by='fbeta', ascending=False).iloc[0,:]
    print('Processed ', models.shape[0], 'models on', len(predictors)-1, "predictors in", round(time.time()-start,3), 'seconds.')
    
    return best_model.values

def plot_search_results(results):
    plt.figure(figsize=(16,4))
    xticks = np.arange(1,len(results)+1)
    colors = ['blue', 'red', 'green']
    for idx in range(1,4):
        ax = plt.subplot(1,3,idx)
        sns.scatterplot(x=xticks, y=results.iloc[:, idx], color=colors[idx-1])
        plt.xlabel('Number of Features')
        plt.title(f'{results.columns[idx]}')
        plt.ylabel(' ')
        plt.xticks(xticks)
    plt.show();
    return None















