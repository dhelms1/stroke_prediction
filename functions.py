import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer
from sklearn.metrics import fbeta_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_validate
sns.set_style("darkgrid", {'axes.edgecolor': 'black'})
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams["legend.edgecolor"] = 'black'
plt.rcParams["legend.fontsize"] = 13

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

def cross_val_model(model, X, y, beta):
    '''
    Cross-validate multiple models, returning a graph of multiple evaluation metrics
    to help choose an initial model. 
    '''
    fbeta = make_scorer(fbeta_score, beta=beta)
    score_methods = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'fbeta': fbeta}
    scores = cross_validate(model, X, y, scoring=score_methods)
    model_name = str(model).split('(')[0]
    print(model_name)
    plt.figure(figsize=(16,4))
    xticks = np.arange(1,6)
    colors = ['blue', 'red', 'green', 'orange']
    index = ['test_accuracy', 'test_precision', 'test_recall', 'test_fbeta']
    titles = list(score_methods.keys())
    for i, idx in enumerate(index):
        ax = plt.subplot(1,4,i+1)
        sns.scatterplot(x=xticks, y=scores[idx], color=colors[i])
        plt.xlabel('Data Split')
        plt.title(f'{titles[i].capitalize()} ({round((scores[idx].mean()*100),2)}%)')
        plt.ylabel(' ')
        plt.xticks(xticks)
    plt.show();
    return None

def plot_feature_importance(X, model):
    '''
    Graph the feature importance for a given model. Returns data frame used for graphing
    '''
    feats_importance = pd.DataFrame()
    feats_importance['Feature'] = X.columns
    feats_importance['Importance'] = model.feature_importances_
    feats_importance.sort_values(by='Importance', ascending=False, inplace=True)
    plt.figure(figsize=(7,5))
    sns.barplot(x='Feature', y='Importance', data=feats_importance, palette=sns.color_palette("viridis",len(feats_importance)))
    plt.xticks(rotation=90)
    plt.xlabel(' ')
    plt.title('Feature Importance')
    plt.show();
    return feats_importance

def sampled_data_split(data):
    '''
    Split data into training and testing sets, randomly under-sample the
    test data to balance classes, randomly over-sample the training data
    to balance classes, and return new training and testing data.
    '''
    stroke_obs = data[data.stroke == 1]
    no_stroke_obs = data[data.stroke == 0]

    sample_size = np.ceil(0.2 * len(stroke_obs))
    stroke_sample = stroke_obs.sample(n=int(sample_size), random_state=42, axis=0)
    stroke_extra = pd.concat([stroke_obs, stroke_sample]).loc[stroke_obs.index.symmetric_difference(stroke_sample.index)]

    sample_size = np.ceil(len(stroke_sample)*1.5)
    no_stroke_sample = no_stroke_obs.sample(n=int(sample_size), random_state=42, axis=0)
    no_stroke_extra = pd.concat([no_stroke_obs, 
                                 no_stroke_sample]).loc[no_stroke_obs.index.symmetric_difference(no_stroke_sample.index)]

    train_set = shuffle(pd.concat([stroke_extra, no_stroke_extra], axis=0))
    test_set = shuffle(pd.concat([stroke_sample, no_stroke_sample], axis=0))
    X_train, y_train = train_set.drop('stroke', axis=1), train_set.stroke
    X_test, y_test = test_set.drop('stroke', axis=1), test_set.stroke

    plt.figure(figsize=(10,4))
    ax1 = plt.subplot(1,2,1)
    sns.countplot(y_train, palette=sns.color_palette("viridis",2))
    plt.xlabel('Classes')
    plt.ylabel('Occurance Count')
    plt.title('Training Split')
    ax2 = plt.subplot(1,2,2)
    sns.countplot(y_test, palette=sns.color_palette("viridis",5))
    plt.xlabel('Classes')
    plt.ylabel('Occurance Count')
    plt.title('Testing Split')
    plt.tight_layout()
    plt.show();
    
    return X_train, X_test, y_train, y_test













