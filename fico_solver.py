import numpy as np
import pandas as pd
import fico_util
import sys


def get_pdf_from_cdf(cdf):
    pdf = np.zeros_like(cdf)
    pdf[0] = cdf[0]
    for i in range(len(cdf) - 1):
        pdf[i+1] = (cdf[i+1] - cdf[i])
    return pdf


def get_util_by_thresh_ind(
        util_repay, util_default, 
        thresh_W_ind, thresh_B_ind, 
        repay_num_W, default_num_W, 
        repay_num_B, default_num_B
    ):
    '''
    Output
    ------
    util_W: (accepted White who repay * u+) + (accepted White who default * u-)
    util_B: (accepted Black who repay * u+) + (accepted Black who default * u-)
    '''
    accepted_num_repay_W = np.sum(repay_num_W[thresh_W_ind:])
    accepted_num_default_W = np.sum(default_num_W[thresh_W_ind:])
    util_W = accepted_num_repay_W * util_repay + accepted_num_default_W * util_default
    accepted_num_repay_B = np.sum(repay_num_B[thresh_B_ind:])
    accepted_num_default_B = np.sum(default_num_B[thresh_B_ind:])
    util_B = accepted_num_repay_B * util_repay + accepted_num_default_B * util_default
    return util_W, util_B


def compute_rate_lists(
        cdf_W, cdf_B, scores, 
        repay_num_W, default_num_W, 
        repay_num_B, default_num_B
    ):
    W_select_each_thresh = np.concatenate(([1], 1-cdf_W))
    B_select_each_thresh = np.concatenate(([1], 1-cdf_B))
    W_TPR_each_thresh = np.full(len(scores) + 1, -np.inf)
    B_TPR_each_thresh = np.full(len(scores) + 1, -np.inf)
    W_FPR_each_thresh = np.full(len(scores) + 1, -np.inf)
    B_FPR_each_thresh = np.full(len(scores) + 1, -np.inf)
    total_repay_W = np.sum(repay_num_W)
    total_repay_B = np.sum(repay_num_B)
    total_default_W = np.sum(default_num_W)
    total_default_B = np.sum(default_num_B)
    for i in range(len(scores) + 1):
        W_TPR_each_thresh[i] = np.sum(repay_num_W[i:]) / total_repay_W
        B_TPR_each_thresh[i] = np.sum(repay_num_B[i:]) / total_repay_B
        W_FPR_each_thresh[i] = np.sum(default_num_W[i:]) / total_default_W
        B_FPR_each_thresh[i] = np.sum(default_num_B[i:]) / total_default_B
    return  W_select_each_thresh, B_select_each_thresh, \
            W_TPR_each_thresh, B_TPR_each_thresh, \
            W_FPR_each_thresh, B_FPR_each_thresh


def thresh_MU(
        scores, util_repay, util_default, 
        repay_num_W, default_num_W, 
        repay_num_B, default_num_B
    ):
    util_W_curve_MU = np.full(len(scores) + 1, -np.inf)
    util_B_curve_MU = np.full(len(scores) + 1, -np.inf)
    # brute-force search over thresholds
    for i in range(len(scores) + 1):
        # find accepted and declined then compute utility
        util_W, util_B = get_util_by_thresh_ind(
            util_repay, util_default, 
            i, i, 
            repay_num_W, default_num_W, 
            repay_num_B, default_num_B
        )
        util_W_curve_MU[i] = util_W
        util_B_curve_MU[i] = util_B
    # find thresh w/ highest util
    thresh_MU_W_ind = np.argmax(util_W_curve_MU)
    thresh_MU_B_ind = np.argmax(util_B_curve_MU)
    thresh_MU_W = scores[-1] if thresh_MU_W_ind >= len(scores) else scores[thresh_MU_W_ind]
    thresh_MU_B = scores[-1] if thresh_MU_B_ind >= len(scores) else scores[thresh_MU_B_ind]
    return thresh_MU_W, thresh_MU_B, thresh_MU_W_ind, thresh_MU_B_ind


def thresh_generic(
        scores, util_repay, util_default, 
        repay_num_W, default_num_W, 
        repay_num_B, default_num_B, 
        W_rate_each_thresh, B_rate_each_thresh, 
        diff_bound=0.01
    ):
    '''
    Works for DP/TPR/FPR
    '''
    util_total_fix_W = np.full(len(scores) + 1, -np.inf)
    util_total_fix_B = np.full(len(scores) + 1, -np.inf)
    # brute-force search over thresholds
    for i in range(len(scores) + 1):
        ####################
        ### fixing White ###
        rate_abs_diff = np.abs(W_rate_each_thresh[i] - B_rate_each_thresh)
        if np.min(rate_abs_diff) <= diff_bound:
            B_thresh_ind = np.argmin(rate_abs_diff)
            # compute util
            util_W_fix_W, util_B_fix_W =  get_util_by_thresh_ind(
                util_repay, util_default, 
                i, B_thresh_ind, 
                repay_num_W, default_num_W, 
                repay_num_B, default_num_B
            )
            util_total_fix_W[i] = util_W_fix_W + util_B_fix_W
        ####################
        ### fixing Black ###
        rate_abs_diff = np.abs(B_rate_each_thresh[i] - W_rate_each_thresh)
        if np.min(rate_abs_diff) <= diff_bound:
            W_thresh_ind = np.argmin(rate_abs_diff)
            # compute util
            util_W_fix_B, util_B_fix_B =  get_util_by_thresh_ind(
                util_repay, util_default, 
                W_thresh_ind, i, 
                repay_num_W, default_num_W, 
                repay_num_B, default_num_B
            )
            util_total_fix_B[i] = util_W_fix_B + util_B_fix_B
    # find thresh w/ highest util
    if np.max(util_total_fix_W) > np.max(util_total_fix_B):
        thresh_W_ind = np.argmax(util_total_fix_W)
        rate_abs_diff = np.abs(W_rate_each_thresh[thresh_W_ind] - B_rate_each_thresh)
        thresh_B_ind = np.argmin(rate_abs_diff)
    else:
        thresh_B_ind = np.argmax(util_total_fix_B)
        rate_abs_diff = np.abs(B_rate_each_thresh[thresh_B_ind] - W_rate_each_thresh)
        thresh_W_ind = np.argmin(rate_abs_diff)
    thresh_W = scores[-1] if thresh_W_ind >= len(scores) else scores[thresh_W_ind]
    thresh_B = scores[-1] if thresh_B_ind >= len(scores) else scores[thresh_B_ind]
    return thresh_W, thresh_B, thresh_W_ind, thresh_B_ind


def thresh_EO(
        scores, util_repay, util_default, 
        repay_num_W, default_num_W, 
        repay_num_B, default_num_B, 
        W_TPR_each_thresh, B_TPR_each_thresh, 
        W_FPR_each_thresh, B_FPR_each_thresh, 
        diff_bound=0.01
    ):
    util_total_fix_W_curve_EO = np.full(len(scores) + 1, -np.inf)
    util_total_fix_B_curve_EO = np.full(len(scores) + 1, -np.inf)
    # brute-force search over thresholds
    for i in range(len(scores) + 1):
        ####################
        ### fixing White ###
        TP_rate_diff_abs = np.abs(W_TPR_each_thresh[i] - B_TPR_each_thresh)
        FP_rate_diff_abs = np.abs(W_FPR_each_thresh[i] - B_FPR_each_thresh)
        EO_diff_abs = np.maximum(TP_rate_diff_abs, FP_rate_diff_abs)
        if np.min(EO_diff_abs) <= diff_bound:
            B_thresh_ind = np.argmin(EO_diff_abs)
            # compute util
            util_W_fix_W, util_B_fix_W =  get_util_by_thresh_ind(
                util_repay, util_default, 
                i, B_thresh_ind, 
                repay_num_W, default_num_W, 
                repay_num_B, default_num_B
            )
            util_total_fix_W_curve_EO[i] = util_W_fix_W + util_B_fix_W
        ####################
        ### fixing Black ###
        TP_rate_diff_abs = np.abs(B_TPR_each_thresh[i] - W_TPR_each_thresh)
        FP_rate_diff_abs = np.abs(B_FPR_each_thresh[i] - W_FPR_each_thresh)
        EO_diff_abs = np.maximum(TP_rate_diff_abs, FP_rate_diff_abs)
        if np.min(EO_diff_abs) <= diff_bound:
            W_thresh_ind = np.argmin(EO_diff_abs)
            # compute util
            util_W_fix_B, util_B_fix_B =  get_util_by_thresh_ind(
                util_repay, util_default, 
                W_thresh_ind, i, 
                repay_num_W, default_num_W, 
                repay_num_B, default_num_B
            )
            util_total_fix_B_curve_EO[i] = util_W_fix_B + util_B_fix_B
    # find thresh w/ highest util
    if np.max(util_total_fix_W_curve_EO) > np.max(util_total_fix_B_curve_EO):
        thresh_EO_W_ind = np.argmax(util_total_fix_W_curve_EO)
        TP_rate_diff_abs = np.abs(W_TPR_each_thresh[thresh_EO_W_ind] - B_TPR_each_thresh)
        FP_rate_diff_abs = np.abs(W_FPR_each_thresh[thresh_EO_W_ind] - B_FPR_each_thresh)
        EO_diff_abs = np.maximum(TP_rate_diff_abs, FP_rate_diff_abs)
        thresh_EO_B_ind = np.argmin(EO_diff_abs)
    else:
        thresh_EO_B_ind = np.argmax(util_total_fix_B_curve_EO)
        TP_rate_diff_abs = np.abs(B_TPR_each_thresh[thresh_EO_B_ind] - W_TPR_each_thresh)
        FP_rate_diff_abs = np.abs(B_FPR_each_thresh[thresh_EO_B_ind] - W_FPR_each_thresh)
        EO_diff_abs = np.maximum(TP_rate_diff_abs, FP_rate_diff_abs)
        thresh_EO_W_ind = np.argmin(EO_diff_abs)
    thresh_EO_W = scores[-1] if thresh_EO_W_ind >= len(scores) else scores[thresh_EO_W_ind]
    thresh_EO_B = scores[-1] if thresh_EO_B_ind >= len(scores) else scores[thresh_EO_B_ind]
    return thresh_EO_W, thresh_EO_B, thresh_EO_W_ind, thresh_EO_B_ind
