# -*- coding: utf-8 -*-
import numpy as np


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    FPR = FP / (FP + TN + 0.00001)
    return f1, precision, recall, FPR, TP, TN, FP, FN


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, pred=None, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, pred=pred, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, pred=pred, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True, direction='>'):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).


    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    m_90 = (-1., -1., -1.)
    m_t_90 = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        pred = eval('score{}threshold'.format(direction))
        target = calc_seq(score, label, threshold, pred=pred, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if target[3] <= 0.1 and target[0] > m_90[0]:
            m_t_90 = threshold
            m_90 = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t, m_90, m_t_90)
    print(m, m_t, m_90, m_t_90)
    return m, m_t

#...............................................................................................................................
def blind_bf_search(
        score, label, val, start, end=None, step_num=1, guess=None, display_freq=1, verbose=True, tw=15, normal=0, direction='>'
    ):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`] for an potion of the test set, then evaluate on a 
    hold-out (i.e. blind) set.
    
    Params:
     score: The anomaly detection results 
     label: The target labels (ground truth)
     val: tuple or list of the results and labels to be used for threshold tuning
     start: the minimum threshold 
     end: the maximum threshold
     step_num: the number of steps to search between start and end
     guess: The default threshold to use if no labels were present and no false positives obtained
     display_freq: frequency of printing out current iteration summary
     verbose: whether to print out summary
     tw: The resampling frequency for avoiding overcounting TP & FP or undercounting FN & TN (i.e. batch_size)
     normal: the value of normal behavior 
     direction: directuib of the anomaly from the threshold (< for OMNI)

    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    score_val, label_val = val
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    if guess is None:
        guess = (start + end) / 2  # automatically select guess as the midpoint if not provided
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        pred = eval('score_val{}threshold'.format(direction))
        if np.abs(label_val - normal).max() or pred.max():
            target = calc_twseq(score_val, label_val, normal, threshold, tw, pred=pred)
            if target[0] > m[0]:
                m_t = threshold
                m = target
            if verbose and i % display_freq == 0:
                print("cur in-sample thr: ", threshold, target, m, m_t)
        else:
            continue
    threshold = m_t  # this is the best threhsold found
    if threshold == 0.0:
        threshold = guess
        if verbose:
            print("No true labels or false detections to tune threshold, using a guessed threshold instead...")
    blind_target = calc_twseq(score, label, normal, threshold, tw, pred=eval('score{}threshold'.format(direction)))
    m, m_t = blind_target, threshold
    print('\nOut-of-sample score:')
    print(m, m_t)
    return m, m_t
            
def calc_twseq(score, label, normal, threshold, tw, pred=None):
    """
    Calculate f1 score for a score sequence, resampled at non-rolling time-window
    """
    predict, pred_batch, label_batch = adjust_predicts_tw(score, label, normal, threshold, tw, pred=pred)
    return calc_point2point(pred_batch, label_batch)
    
def adjust_predicts_tw(score, label, normal, threshold, tw, pred=None):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`, where a non-rolling time 
    window (i.e. batch)is used as the basis for adjusting the score. As for adjusting score, only intervals after the first
    true positive detection are adjusted, wheras late detections are not rewarded.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        normal (float): The value of a normal label (not anomaly)
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is higher than the threshold.
        tw (int): the nonrolling interval for adjusting the score
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,

    Returns:
        predict (np.ndarray): adjusted predict labels
        pred_batch (np.ndarray): downsampled (in batches) adjusted predict labels
        score_batch (np.ndarray): downsampled true labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    batched_shape = (int(np.ceil(score.shape[0]/tw)), 1)
    label_batch, pred_batch = np.zeros(batched_shape), np.zeros(batched_shape)
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label != normal
    detect_state = False  # triggered when a True anomaly is detected by model
    anomaly_batch_count = 0
    i, i_tw = 0, 0
    step = tw
    while i < len(score):
        j = min(i+step, len(score))  # end of tw (batch) starting at i
        
        # Adjust step size if needed
        if step > 2 and actual[i:j].sum() > 1:
            if np.diff(np.where(actual[i:j])).max() > 1:  # if it finds an interruption in the true label continuity
                step = min(int((j-i)/2), 2)  # reduce step size
                label_batch, pred_batch = np.append(label_batch, 0), np.append(pred_batch, 0)  # increase size
                j = i + step
            else:
                step = tw
        else:
            step = tw
        
        # start rolling window scoring
        if actual[i:j].max():  # If label = T
            if not actual[i]:  # if first value is normal
                detect_state = False
            s = actual[i:j].argmax()  # this is the index of the first occurance
            if detect_state:  # if anomaly was previously detected by model
                anomaly_batch_count += 1
                pred_batch[i_tw], label_batch[i_tw], predict[i+s:j] = 1, 1, 1
            elif predict[i:j].max():  # if alert was detected with T
                detect_state = True  # turn on detection state
                anomaly_batch_count += 1
                pred_batch[i_tw], label_batch[i_tw], predict[i+s:j] = 1, 1, 1
            else:
                detect_state = False
                label_batch[i_tw] = 1
        else:
            detect_state = False
            if predict[i:j].max():  # if False positive
                pred_batch[i_tw] = 1
        i += step
        i_tw += 1
    return predict, pred_batch, label_batch
