from os.path import (
    abspath,
    dirname,
    exists,
    join,
)
from pandas import (
    read_csv,
    merge,
    concat,
    DataFrame,
    Series,
)
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_percentage_error,
    average_precision_score,
    mean_absolute_error,
    coverage_error,
    ndcg_score,
    precision_score,
    f1_score,
    recall_score,
)

def get_accuracy(y_true, y_pred, k=3):
    y_true = [1 if y >= k else 0 for y in y_true]
    y_pred = [1 if y >= k else 0 for y in y_pred]
    
    return f1_score(y_true, y_pred)

def get_harmonic(W, X):
    den = 0
    for (w, x) in zip(W, X):
        den += (w/x)
    num = sum(W)
    res = num/den
    
    return res

PWD = dirname(abspath(__file__))
datadir = join(PWD, '../Datasets/ml-1m/')
files = {}
files['train'] = join(datadir, 'train.csv')
files['test'] = join(datadir, 'test.csv')
SIMILARITIES = [
    .7,
    .8,
    .9,
]
THRESHOLDS = [
    10,
    20,
    30,
    50,
    100,
]
TAKE_FIRST = [
    10,
    20,
    30,
    50,
    100,
    120,
    200,
]

dfs = {}
for file in files:
    dfs[file] = read_csv(
        files[file],
        engine='python',
        encoding='latin1',
    )
    dfs[file]['rating'] = dfs[file]['rating'].astype(int)

ratings = {}
users = dfs['train']['userId'].to_numpy()
users = set(users)
users = list(users)
for user in users:
    ratings[user] = {}

rows = dfs['train'].to_numpy()
for row in rows:
    ratings[row[2]][row[0]] = row[1]

dfs['users'] = dfs['train'].pivot_table(
    index=['title'],
    columns=['userId'],
    values='rating',
)

def predict(similarity, threshold, take_first):
    corr_user = dfs['users'].corr(min_periods=threshold)

    corr_user += 1.0
    corr_user /= 2.0

    users = corr_user.columns.tolist()

    similarities = {}
    for u in users:
        users_sim = corr_user[u].dropna()
        similarities[u] = {}
        for (v, s) in zip(users_sim.index, users_sim.values):
            similarities[u][v] = s

    dfs['pivot'] = dfs['train'].pivot_table(
        index=['title'],
        columns=['userId'],
        values='rating',
    )
    rows = dfs['test'].to_numpy()
    predictions = []
    for row in rows[:]:
        rating = 0
        u = row[2]
        m = row[0]
        r = row[1]
        tot = 0
        raters = dfs['pivot'].loc[m].dropna().index
        sim_tot = 0
        #print(raters)
        W = []
        X = []
        for v in raters:
            if v not in similarities[u]:
                continue
            if similarities[u][v] >= .7:
                rating += ratings[v][m]*similarities[u][v]
                W.append(similarities[u][v])
                X.append(ratings[v][m])
                sim_tot += similarities[u][v]
                tot += 1
                if tot > take_first:
                    break
        if tot > 0:
            rating /= sim_tot
            rating = get_harmonic(W, X)
            #rating *= 5
            #rating /= 5
        else:
            rating = dfs['pivot'].loc[m].dropna().values
            if len(rating) < 1:
                rating = dfs['user'].loc[u].dropna().values
                rating = sum(rating)/len(rating)
            else:
                rating = sum(rating)/len(rating)
        rating += .5
        rating = int(rating)
        predictions.append(rating)

    dfs['test']['prediction'] = predictions
    error_row = {}
    error_row['threshold'] = threshold
    error_row['similarity'] = similarity
    error_row['take_first'] = take_first
    mae = mean_absolute_percentage_error(dfs['test']['rating'], dfs['test']['prediction'])
    error_row['mae'] = mae
    error_row['f1_3'] = get_accuracy(dfs['test']['rating'], dfs['test']['prediction'])
    error_row['f1_4'] = get_accuracy(dfs['test']['rating'], dfs['test']['prediction'], 4)
    
    return error_row

if __name__ == '__main__':
    errors = []
    for similarity in SIMILARITIES:
        for threshold in THRESHOLDS:
            for take_first in TAKE_FIRST:
                errors.append(predict(similarity, threshold, take_first))
    df = DataFrame(errors)
    df.to_csv('errors.csv', index=False)