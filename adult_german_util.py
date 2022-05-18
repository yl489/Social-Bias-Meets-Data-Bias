import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from fairlearn.reductions import DemographicParity, TruePositiveRateParity, FalsePositiveRateParity, EqualizedOdds
from sklearn.metrics import accuracy_score
from fairlearn.metrics import false_positive_rate, true_positive_rate, demographic_parity_difference, equalized_odds_difference


def flip_labels(Y, prob=0, seed=None):
    Y = np.copy(Y)
    np.random.seed(seed)
    p = np.random.rand(len(Y))
    Y[p < prob] = 1 - Y[p < prob] # flip 0,1
    return Y


def init_constraint(s, diff_bound=None):
    if s == 'DP':
        return DemographicParity(difference_bound=diff_bound)
    elif s == 'TPR':
        return TruePositiveRateParity(difference_bound=diff_bound)
    elif s == 'FPR':
        return FalsePositiveRateParity(difference_bound=diff_bound)
    elif s == 'EO':
        return EqualizedOdds(difference_bound=diff_bound)


def init_metric(s):
    if s == 'acc':
        return accuracy_score
    elif s == 'fpr':
        return false_positive_rate
    elif s == 'tpr':
        return true_positive_rate
    elif s == 'dp':
        return demographic_parity_difference
    elif s == 'eo':
        return equalized_odds_difference


class Adult:
    
    ################################################################################################
    # Code for data_transform and load_data extracted from: https://fairmlbook.org/code/adult.html #
    ################################################################################################
    @staticmethod
    def data_transform(df):
        binary_data = pd.get_dummies(df)
        feature_cols = binary_data[binary_data.columns[:-2]]
        scaler = preprocessing.StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(feature_cols), columns=feature_cols.columns)
        return data

    @staticmethod
    def load_data():
        features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
                "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per week", "Country", "Target"] 
        original_train  = pd.read_csv('./data/adult/adult.data', names=features, sep=r'\s*,\s*', 
                                    engine='python', na_values="?")
        original_test = pd.read_csv('./data/adult/adult.test', names=features, sep=r'\s*,\s*', 
                                    engine='python', na_values="?", skiprows=1)

        num_train = len(original_train)
        original = pd.concat([original_train, original_test])
        labels = original['Target']
        labels = labels.replace('<=50K', 0).replace('>50K', 1)
        labels = labels.replace('<=50K.', 0).replace('>50K.', 1)
        # Redundant column
        del original["Education"]
        # Remove target variable
        del original["Target"]

        original_train = original[:num_train]
        original_test = original[num_train:]
        data = Adult.data_transform(original)
        train_data = data[:num_train]
        train_labels = labels[:num_train]
        test_data = data[num_train:]
        test_labels = labels[num_train:]

        return train_data, test_data, train_labels, test_labels, original_train['Sex'], original_test['Sex']
        
    @staticmethod
    def flip_female_1(Y, group_array, prob, seed=None):
        flipped = np.copy(Y)
        female_arr = group_array == 'Female'
        ones_arr = Y == 1
        flipped[female_arr & ones_arr] = flip_labels(Y[female_arr & ones_arr], prob, seed)
        return flipped
        
    @staticmethod
    def flip_male_0(Y, group_array, prob, seed=None):
        flipped = np.copy(Y)
        male_arr = group_array == 'Male'
        zeros_arr = Y == 0
        flipped[male_arr & zeros_arr] = flip_labels(Y[male_arr & zeros_arr], prob, seed)
        return flipped
        
    
class German:
    
    @staticmethod
    def load_data():
        data = np.loadtxt('./data/german/german.data-numeric')
        X = data[:, :-1]
        y = data[:, -1]
        y[y == 2] = 0

        ############################################################################################################
        # Following code extracted from: https://github.com/google-research/google-research/tree/master/label_bias #
        ############################################################################################################
        X_train_german, X_test_german, y_train_german, y_test_german = train_test_split(X, y, test_size=0.33, random_state=42)
        protected_train_german = np.where(X_train_german[:, 9] <= 30, 1, 0)
        protected_test_german = np.where(X_test_german[:, 9] <= 30, 1, 0)
        return X_train_german, X_test_german, y_train_german, y_test_german, protected_train_german, protected_test_german

    @staticmethod
    def flip_above_30_bad(Y, group_array, prob, seed=None):
        flipped = np.copy(Y)
        above_30_arr = group_array == 0
        ones_arr = Y == 0
        flipped[above_30_arr & ones_arr] = flip_labels(Y[above_30_arr & ones_arr], prob, seed)
        return flipped

    @staticmethod
    def flip_below_30_good(Y, group_array, prob, seed=None):
        flipped = np.copy(Y)
        below_30_arr = group_array == 1
        ones_arr = Y == 1
        flipped[below_30_arr & ones_arr] = flip_labels(Y[below_30_arr & ones_arr], prob, seed)
        return flipped
        