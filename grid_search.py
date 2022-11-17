import argparse
import ast
import numpy as np
import pandas as pd
import sklearn
import xgboost
import lightgbm

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', '-c', required=True)
    parser.add_argument('--parameters', '-p', required=True)
    parser.add_argument('--label_encoding', '-l', default=False,
                        help='Whether to use default one-hot encoding or label encoding',
                        action='store_true')

    args = parser.parse_args()

    parameters = ast.literal_eval(args.parameters)

    # Select model
    model = None
    if args.classifier == 'xgboost':
        model = xgboost.XGBClassifier()
    elif args.classifier == 'lightgbm':
        model = lightgbm.sklearn.LGBMClassifier()

    # Get data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test-full.csv')

    train_df_input = train_df.drop(['Id', 'Cover_Type'], axis=1)
    train_df_input = train_df_input.drop(train_df_input.columns[28], axis=1)
    test_df_input = test_df.drop(['Id'], axis=1)
    test_df_input = test_df_input.drop(train_df_input.columns[28], axis=1)

    # Prepare inputs
    full_train_X = (train_df_input.values - train_df_input.values.mean(0)) / train_df_input.values.std(0)
    full_train_y = train_df['Cover_Type'].values - 1

    full_test_X = (test_df_input.values - train_df_input.values.mean(0)) / train_df_input.values.std(0)

    if args.label_encoding:
        new_train_X = np.zeros((full_train_X.shape[0], 12))
        new_test_X = np.zeros((full_test_X.shape[0], 12))

        new_train_X[:, :-2] = full_train_X[:, :10]
        new_train_X[:, -2] = np.argmax(full_train_X[:, -43:-39], axis=1)
        new_train_X[:, -1] = np.argmax(full_train_X[:, -39:], axis=1)

        new_test_X[:, :-2] = full_test_X[:, :10]
        new_test_X[:, -2] = np.argmax(full_test_X[:, -43:-39], axis=1)
        new_test_X[:, -1] = np.argmax(full_test_X[:, -39:], axis=1)

        full_test_X = new_test_X
        full_train_X = new_train_X

    # Prepare grid search
    grid_search = sklearn.model_selection.GridSearchCV(model,
                                                       parameters,
                                                       scoring='accuracy',
                                                       cv=5,
                                                       verbose=2)

    grid_search.fit(full_train_X, full_train_y)

    print(grid_search.best_estimator_)
    print(grid_search.best_score_)
