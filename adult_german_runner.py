from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.metrics import MetricFrame
import sys, os, argparse
from time import time
from adult_german_util import *


def main(args):
    # load data
    if args.d == 'adult':
        X_train, X_test, y_train_original, y_test, protected_train, protected_test = Adult.load_data()
    elif args.d == 'german':
        X_train, X_test, y_train_original, y_test, protected_train, protected_test = German.load_data()
    # to store stats
    prob = np.linspace(0, 0.5, 6)
    diff = np.zeros((len(prob), args.n))
    diff_mitigated = np.zeros_like(diff)
    acc_overall = np.zeros_like(diff)
    acc_overall_mitigated = np.zeros_like(diff)
    acc_diff = np.zeros_like(diff)
    acc_diff_mitigated = np.zeros_like(diff)
    diff_train = np.zeros_like(diff)
    diff_train_mitigated = np.zeros_like(diff)
    acc_overall_train = np.zeros_like(diff)
    acc_overall_train_mitigated = np.zeros_like(diff)
    acc_diff_train = np.zeros_like(diff)
    acc_diff_train_mitigated = np.zeros_like(diff)
    for i in range(len(prob)):
        p = prob[i]
        print(p)
        for j in range(args.n):
            print(j)
            # flip
            if args.d == 'adult' and args.f == 1: # female 1
                y_train = Adult.flip_female_1(y_train_original, protected_train, p)
            elif args.d == 'adult' and args.f == 2: # male 0
                y_train = Adult.flip_male_0(y_train_original, protected_train, p)
            elif args.d == 'adult' and args.f == 3: # both
                y_train = Adult.flip_female_1(y_train_original, protected_train, p)
                y_train = Adult.flip_male_0(y_train, protected_train, p)
            elif args.d == 'german' and args.f == 1: # <30 1
                y_train = German.flip_below_30_good(y_train_original, protected_train, p)
            elif args.d == 'german' and args.f == 2: # >30 0
                y_train = German.flip_above_30_bad(y_train_original, protected_train, p)
            elif args.d == 'german' and args.f == 3: # both
                y_train = German.flip_below_30_good(y_train_original, protected_train, p)
                y_train = German.flip_above_30_bad(y_train, protected_train, p)
            # w/o constraint
            LR = LogisticRegression(max_iter=10000, n_jobs=-1)
            LR.fit(X_train, y_train)
            pred = LR.predict(X_test)
            pred_train = LR.predict(X_train)
            # w/ constraint
            base = LogisticRegression(max_iter=10000, n_jobs=-1)
            c = init_constraint(args.c, args.db)
            mitigator = ExponentiatedGradient(base, c)
            mitigator.fit(X_train, y_train, sensitive_features=protected_train)
            pred_mitigated = mitigator.predict(X_test)
            pred_train_mitigated = mitigator.predict(X_train)
            # evaluate
            m = init_metric(args.m)
            if args.m == 'dp' or args.m == 'eo':
                # before mitigation
                diff[i,j] = m(y_test, pred, sensitive_features=protected_test)
                diff_train[i,j] = m(y_train, pred_train, sensitive_features=protected_train)
                # after mitigation
                diff_mitigated[i,j] = m(y_test, pred_mitigated, sensitive_features=protected_test)
                diff_train_mitigated[i,j] = m(y_train, pred_train_mitigated, sensitive_features=protected_train)
            else:
                # before mitigation
                mf = MetricFrame(m, y_test, pred, sensitive_features=protected_test)
                diff[i,j] = mf.difference()
                mf_train = MetricFrame(m, y_train, pred_train, sensitive_features=protected_train)
                diff_train[i,j] = mf.difference()
                # after mitigation
                mf_mitigated = MetricFrame(m, y_test, pred_mitigated, sensitive_features=protected_test)
                diff_mitigated[i,j] = mf_mitigated.difference()
                mf_train_mitigated = MetricFrame(m, y_train, pred_train_mitigated, sensitive_features=protected_train)
                diff_train_mitigated[i,j] = mf_train_mitigated.difference()
            # acc before mitigation
            mf_acc = MetricFrame(accuracy_score, y_test, pred, sensitive_features=protected_test)
            acc_overall[i,j] = mf_acc.overall
            acc_diff[i,j] = mf_acc.difference()
            mf_acc_train = MetricFrame(accuracy_score, y_train, pred_train, sensitive_features=protected_train)
            acc_overall_train[i,j] = mf_acc_train.overall
            acc_diff_train[i,j] = mf_acc_train.difference()
            # acc after mitigation
            mf_acc_mitigated = MetricFrame(accuracy_score, y_test, pred_mitigated, sensitive_features=protected_test)
            acc_overall_mitigated[i,j] = mf_acc_mitigated.overall
            acc_diff_mitigated[i,j] = mf_acc_mitigated.difference()
            mf_acc_train_mitigated = MetricFrame(accuracy_score, y_train, pred_train_mitigated, sensitive_features=protected_train)
            acc_overall_train_mitigated[i,j] = mf_acc_train_mitigated.overall
            acc_diff_train_mitigated[i,j] = mf_acc_train_mitigated.difference()
        print()
    # save res
    np.savez(
        '{}_{}_{}_{}_{}_{}.npz'.format(args.d, args.c, str(args.db), args.m, str(args.f), str(args.n)),
        diff=diff,
        diff_mitigated=diff_mitigated,
        acc_overall=acc_overall,
        acc_overall_mitigated=acc_overall_mitigated,
        acc_diff=acc_diff,
        acc_diff_mitigated=acc_diff_mitigated,
        diff_train=diff_train,
        diff_train_mitigated=diff_train_mitigated,
        acc_overall_train=acc_overall_train,
        acc_overall_train_mitigated=acc_overall_train_mitigated,
        acc_diff_train=acc_diff_train,
        acc_diff_train_mitigated=acc_diff_train_mitigated
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fair Classifier')
    parser.add_argument('--d', default='adult', type=str) # data
    parser.add_argument('--c', default='DP', type=str) # constraint
    parser.add_argument('--db', default=0.01, type=float) # difference bound
    parser.add_argument('--m', default='dp', type=str) # metrics
    parser.add_argument('--f', default=1, type=int) # flip
    parser.add_argument('--n', default=1, type=int) # number of iterations
    args = parser.parse_args()
    print(args)
    start = time()
    main(args)
    print('Runtime:', time() - start)
