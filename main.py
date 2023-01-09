import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def main():
    labels, features, features_list = create_features()
    train_features, test_features, train_labels, test_labels \
        = train_test_split(features, labels, test_size=0.25, random_state=23)

    rf = train_model(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)

    print_model_stats(features_list, predictions, test_features, test_labels)

    print_feature_importances(features_list, rf)


def print_feature_importances(features_list, rf):
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


def print_model_stats(features_list, predictions, test_features, test_labels):
    baseline_errors = calc_aseline_errors(features_list, test_features, test_labels)
    print('Average baseline error: ', round(np.mean(baseline_errors), 2))
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')


def train_model(train_features, train_labels):
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=23)
    # Train the model on training data
    rf.fit(train_features, train_labels)
    return rf


def calc_aseline_errors(features_list, test_features, test_labels):
    # The baseline predictions are the historical averages
    baseline_preds = test_features[:, features_list.index('average')]
    # Baseline errors, and display average baseline error
    return abs(baseline_preds - test_labels)


def create_features():
    features = read_data()

    # Labels are the values we want to predict
    labels = np.array(features['actual'])

    # Remove the labels from the features
    features = features.drop('actual', axis=1)

    # Saving features names for later use
    features_list = list(features.columns)

    # Convert to numpy array
    features = np.array(features)

    return labels, features, features_list


def read_data():
    # pd.set_option('display.max_columns', None)

    features = pd.read_csv('temps.csv')
    features = pd.get_dummies(features)

    return features


if __name__ == '__main__':
    main()
