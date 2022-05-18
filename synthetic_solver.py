import numpy as np


def store_res(
        ind, theta_a_list, theta_b_list, 
        util_a_list, util_b_list, util_total_list, 
        select_a_list, select_b_list, select_total_list, 
        theta_a, theta_b, 
        X_a_0, X_a_1, X_b_0, X_b_1, 
        u_plus, u_minus
    ):
    theta_a_list[ind] = theta_a
    theta_b_list[ind] = theta_b
    util_a, util_b = get_util(theta_a, theta_b, X_a_0, X_a_1, X_b_0, X_b_1, u_plus, u_minus)
    util_a_list[ind] = util_a
    util_b_list[ind] = util_b
    util_total_list[ind] = util_a + util_b
    accepted_a_num = np.sum(X_a_0 > theta_a) + np.sum(X_a_1 > theta_a)
    accepted_b_num = np.sum(X_b_0 > theta_b) + np.sum(X_b_1 > theta_b)
    select_a_list[ind] = accepted_a_num / (len(X_a_0) + len(X_a_1))
    select_b_list[ind] = accepted_b_num / (len(X_b_0) + len(X_b_1))
    select_total_list[ind] = (accepted_a_num + accepted_b_num) / \
                            (len(X_a_0) + len(X_a_1) + len(X_b_0) + len(X_b_1))


def compute_rates_lists(theta_list, X_a_0, X_a_1, X_b_0, X_b_1):
    select_a_each_theta = np.full(len(theta_list), -np.inf)
    select_b_each_theta = np.full(len(theta_list), -np.inf)
    TPR_a_each_theta = np.full(len(theta_list), -np.inf)
    TPR_b_each_theta = np.full(len(theta_list), -np.inf)
    FPR_a_each_theta = np.full(len(theta_list), -np.inf)
    FPR_b_each_theta = np.full(len(theta_list), -np.inf)
    num_a_0 = len(X_a_0)
    num_a_1 = len(X_a_1)
    num_a = num_a_0 + num_a_1
    num_b_0 = len(X_b_0)
    num_b_1 = len(X_b_1)
    num_b = num_b_0 + num_b_1
    for j in range(len(theta_list)):
        theta = theta_list[j]
        select_a_each_theta[j] = (np.sum(X_a_0 > theta) + np.sum(X_a_1 > theta)) / num_a
        select_b_each_theta[j] = (np.sum(X_b_0 > theta) + np.sum(X_b_1 > theta)) / num_b
        TPR_a_each_theta[j] = np.sum(X_a_1 > theta) / num_a_1
        TPR_b_each_theta[j] = np.sum(X_b_1 > theta) / num_b_1
        FPR_a_each_theta[j] = np.sum(X_a_0 > theta) / num_a_0
        FPR_b_each_theta[j] = np.sum(X_b_0 > theta) / num_b_0
    return  select_a_each_theta, select_b_each_theta, \
            TPR_a_each_theta, TPR_b_each_theta, \
            FPR_a_each_theta, FPR_b_each_theta


def get_util(theta_a, theta_b, X_a_0, X_a_1, X_b_0, X_b_1, u_plus, u_minus):
    util_a = np.sum(X_a_1 > theta_a) * u_plus - np.sum(X_a_0 > theta_a) * u_minus
    util_b = np.sum(X_b_1 > theta_b) * u_plus - np.sum(X_b_0 > theta_b) * u_minus
    return util_a, util_b


def theta_MU(theta_list, X_a_0, X_a_1, X_b_0, X_b_1, u_plus, u_minus):
    util_a_MU = np.full(len(theta_list), -np.inf)
    util_b_MU = np.full(len(theta_list), -np.inf)
    for i in range(len(theta_list)):
        theta = theta_list[i]
        util_a, util_b = get_util(theta, theta, X_a_0, X_a_1, X_b_0, X_b_1, u_plus, u_minus)
        util_a_MU[i] = util_a
        util_b_MU[i] = util_b
    return theta_list[np.argmax(util_a_MU)], theta_list[np.argmax(util_b_MU)]


def theta_generic(
        theta_list, 
        X_a_0, X_a_1, 
        X_b_0, X_b_1, 
        u_plus, u_minus, 
        rate_a_each_theta, rate_b_each_theta, 
        diff_bound=0.01
    ):
    '''
    Works for DP/TP/FP/ErR
    '''
    util_total_fix_a = np.full(len(theta_list), -np.inf)
    util_total_fix_b = np.full(len(theta_list), -np.inf)
    for i in range(len(theta_list)):
        ####################
        # fix a
        theta_a_fix_a = theta_list[i]
        rate_a_fix_a = rate_a_each_theta[i]
        # find theta_b w/ closest rate
        rate_diff_abs_fix_a = np.abs(rate_a_fix_a - rate_b_each_theta)
        if np.min(rate_diff_abs_fix_a) <= diff_bound:
            theta_b_fix_a = theta_list[np.argmin(rate_diff_abs_fix_a)]
            # compute util
            util_a_fix_a, util_b_fix_a = get_util(
                theta_a_fix_a, theta_b_fix_a, 
                X_a_0, X_a_1, 
                X_b_0, X_b_1, 
                u_plus, u_minus
            )
            util_total_fix_a[i] = util_a_fix_a + util_b_fix_a
        ####################
        # fix b
        theta_b_fix_b = theta_list[i]
        rate_b_fix_b = rate_b_each_theta[i]
        # find theta_a w/ closest rate
        rate_diff_abs_fix_b = np.abs(rate_b_fix_b - rate_a_each_theta)
        if np.min(rate_diff_abs_fix_b) <= diff_bound:
            theta_a_fix_b = theta_list[np.argmin(rate_diff_abs_fix_b)]
            util_a_fix_b, util_b_fix_b = get_util(
                theta_a_fix_b, theta_b_fix_b, 
                X_a_0, X_a_1, 
                X_b_0, X_b_1, 
                u_plus, u_minus
            )
            util_total_fix_b[i] = util_a_fix_b + util_b_fix_b
    # find thresh w/ highest util
    if np.max(util_total_fix_a) > np.max(util_total_fix_b):
        theta_a_ind = np.argmax(util_total_fix_a)
        rate_a = rate_a_each_theta[theta_a_ind]
        # find theta_b w/ closest rate
        rate_diff_abs = np.abs(rate_a - rate_b_each_theta)
        theta_b_ind = np.argmin(rate_diff_abs)
    else:
        theta_b_ind = np.argmax(util_total_fix_b)
        rate_b = rate_b_each_theta[theta_b_ind]
        # find theta_a w/ closest rate
        rate_diff_abs = np.abs(rate_b - rate_a_each_theta)
        theta_a_ind = np.argmin(rate_diff_abs)
    return theta_list[theta_a_ind], theta_list[theta_b_ind]


def theta_EO(
        theta_list, 
        X_a_0, X_a_1, 
        X_b_0, X_b_1, 
        u_plus, u_minus, 
        TPR_a_each_theta, TPR_b_each_theta, 
        FPR_a_each_theta, FPR_b_each_theta, 
        diff_bound=0.01
    ):
    util_total_fix_a_EO = np.full(len(theta_list), -np.inf)
    util_total_fix_b_EO = np.full(len(theta_list), -np.inf)
    for i in range(len(theta_list)):
        ####################
        # fix a
        theta_a_fix_a = theta_list[i]
        TPR_a_fix_a = TPR_a_each_theta[i]
        FPR_a_fix_a = FPR_a_each_theta[i]
        # find theta_b w/ smallest EO difference
        TPR_diff_abs_fix_a = np.abs(TPR_a_fix_a - TPR_b_each_theta)
        FPR_diff_abs_fix_a = np.abs(FPR_a_fix_a - FPR_b_each_theta)
        EO_diff_abs_fix_a = np.maximum(TPR_diff_abs_fix_a, FPR_diff_abs_fix_a)
        if np.min(EO_diff_abs_fix_a) <= diff_bound:
            theta_b_fix_a = theta_list[np.argmin(EO_diff_abs_fix_a)]
            # compute util
            util_a_fix_a, util_b_fix_a = get_util(
                theta_a_fix_a, theta_b_fix_a, 
                X_a_0, X_a_1, 
                X_b_0, X_b_1, 
                u_plus, u_minus
            )
            util_total_fix_a_EO[i] = util_a_fix_a + util_b_fix_a
        ####################
        # fix b
        theta_b_fix_b = theta_list[i]
        TPR_b_fix_b = TPR_b_each_theta[i]
        FPR_b_fix_b = FPR_b_each_theta[i]
        # find theta_a w/ smallest EO difference
        TPR_diff_abs_fix_b = np.abs(TPR_b_fix_b - TPR_a_each_theta)
        FPR_diff_abs_fix_b = np.abs(FPR_b_fix_b - FPR_a_each_theta)
        EO_diff_abs_fix_b = np.maximum(TPR_diff_abs_fix_b, FPR_diff_abs_fix_b)
        if np.min(EO_diff_abs_fix_b) <= diff_bound:
            theta_a_fix_b = theta_list[np.argmin(EO_diff_abs_fix_b)]
            util_a_fix_b, util_b_fix_b = get_util(
                theta_a_fix_b, theta_b_fix_b, 
                X_a_0, X_a_1, 
                X_b_0, X_b_1, 
                u_plus, u_minus
            )
            util_total_fix_b_EO[i] = util_a_fix_b + util_b_fix_b
    # find thresh w/ highest util
    if np.max(util_total_fix_a_EO) > np.max(util_total_fix_b_EO):
        theta_a_ind = np.argmax(util_total_fix_a_EO)
        TPR_a = TPR_a_each_theta[theta_a_ind]
        FPR_a = FPR_a_each_theta[theta_a_ind]
        # find theta_b w/ closest rate
        TPR_diff_abs = np.abs(TPR_a - TPR_b_each_theta)
        FPR_diff_abs = np.abs(FPR_a - FPR_b_each_theta)
        EO_diff_abs = np.maximum(TPR_diff_abs, FPR_diff_abs)
        theta_b_ind = np.argmin(EO_diff_abs)
    else:
        theta_b_ind = np.argmax(util_total_fix_b_EO)
        TPR_b = TPR_b_each_theta[theta_b_ind]
        FPR_b = FPR_b_each_theta[theta_b_ind]
        # find theta_a w/ closest rate
        TPR_diff_abs = np.abs(TPR_b - TPR_a_each_theta)
        FPR_diff_abs = np.abs(FPR_b - FPR_a_each_theta)
        EO_diff_abs = np.maximum(TPR_diff_abs, FPR_diff_abs)
        theta_a_ind = np.argmin(EO_diff_abs)
    return theta_list[theta_a_ind], theta_list[theta_b_ind]
