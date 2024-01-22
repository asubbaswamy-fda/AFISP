import numpy as np
import torch
from sklearn.metrics import roc_auc_score
        
                
def clip_predictions(preds, upper_bound=0.99, lower_bound=0.01):
    """Clip probability predictions to be in the (0, 1) open interval.

    :param preds: Array of sample predictions
    :type preds: (num samples,) np array
    :param upper_bound: Upper bound of clipped predictions, defaults to 0.99
    :type upper_bound: float
    :param lower_bound: Lower bound of clipped predictions, defaults to 0.01
    :type lower_bound: float
    :return: Predictions clipped to be in [lower_bound, upper_bound] interval
    :rtype: List[float]
    """
    if upper_bound >= 1.0 or lower_bound <= 0.0:
            raise RuntimeError('upper_bound must be < 1 and lower_bound must be > 0')
    new_preds = np.copy(preds)
    one_inds = np.where(preds > upper_bound)[0]
    zero_inds = np.where(preds < lower_bound)[0]
    
    new_preds[one_inds] = np.repeat(upper_bound, one_inds.shape[0])
    new_preds[zero_inds] = np.repeat(lower_bound, zero_inds.shape[0])
    
    return new_preds


# Sample wise loss functions
def cross_entropy(y,y_pred):
    """Samplewise cross entropy loss for probabilistic classification
    
    :param y: Array of true binary classification labels
    :type y: Numpy array with values in {0, 1}
    :param y_pred: Array of probabilistic predictions (between 0 and 1)
    :type y_pred: Numpy array with values in (0, 1)
    :return: Array of per-sample cross entropy losses
    :rtype: Array[float]
    """
    
    # Note: problems if preds are in {0, 1}
    # Clip predictions before using.
    y_pred = clip_predictions(y_pred)
    return -(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

def entropy(y, y_pred):
    return -np.log(y_pred)

def brier(y, y_pred):
    """Samplewise brier score for probabilistic classification

    param y: Array of true binary classification labels
    :type y: Numpy array with values in {0, 1}
    :param y_pred: Array of probabilistic predictions (between 0 and 1)
    :type y_pred: Numpy array with values in (0, 1)
    :return: Array of per-sample brier scores
    :rtype: Array[float]
    """
    
    return (y-y_pred)**2

def zero_one_loss(y, y_pred):
    """Samplewise Zero-One Loss for binary classification
    
    param y: Array of true binary classification labels
    :type y: Numpy array with values in {0, 1}
    :param y_pred: Array of binary classification predictions {0, 1}
    :type y_pred: Numpy array with values in {0, 1}
    :return: Array of per-sample zero-one losses
    :rtype: Array[float]
    """
    return 1. * (y != y_pred)
    
def mse(y, y_pred):
    """Samplewise mean squared error for regression

    :param y: Array of true regression labels
    :param y_pred: Array of regressin predictions
    :return: Array of samplewise mean squared errors
    """
    
    return (y - y_pred)**2

def logit(p):
    # Map probabilities to the real line
    # Note: requires p to be in (0, 1) exclusive
    clipped = clip_predictions(p)
    return np.log(clipped/(1.-clipped))

def hinge_surrogate(labels, logits):
    positives_term = labels * np.maximum(1.0 - logits, 0)
    negatives_term =  (1.0 - labels) * np.maximum(1.0 + logits, 0)
    
    # for surrogate purposes, this should just be the positive term
    return positives_term + negatives_term

def xent_surrogate(labels, logits):
    softplus_term = np.maximum(-logits, 0.0) + np.log(1.0 + np.exp(-np.abs(logits)))
    
    # for surrogate purposes, this should just be the softplus term
    # because labels are all 1
    
    return logits - labels * logits + softplus_term


# def pfohl_torch_roc_auc_surrogate(y, y_pred, surrogate='xent'):
    
    
#     y_torch = torch.tensor(y)
#     # pfohl used log softmax (so log probabilities
#     logits_torch = torch.tensor(np.log(y_pred))
    
#     logits_difference_torch = logits_torch.unsqueeze(0) - logits_torch.unsqueeze(1)
#     labels_difference_torch = y_torch.unsqueeze(0) - y_torch.unsqueeze(1)
    
#     # matrex which is 1 if label y_i != label y_j
#     abs_label_difference = torch.abs(labels_difference_torch)
    
#     signed_logits_difference_torch = logits_difference_torch * labels_difference_torch
    
#     # TODO: make it 'DRY'
#     if surrogate == 'xent':
#         loss = torch.log(torch.sigmoid(signed_logits_difference_torch))
#         loss = (abs_label_difference * loss).mean(axis=0) * 0.5
#     elif surrogate == 'hinge':
#         loss = torch.maximum(torch.zeros(1), torch.ones(1) - signed_logits_difference_torch)
#         loss = (abs_label_difference * loss).mean(axis=0) * 0.5
        
#     return np.array(loss.tolist())
    

def torch_roc_auc_surrogate(y, y_pred, surrogate='xent'):
    """PyTorch computation of a surrogate samplewise AUROC loss.

    param y: Array of true binary classification labels
    :type y: Numpy array with values in {0, 1}
    :param y_pred: Array of probabilistic predictions (between 0 and 1)
    :type y_pred: Numpy array with values in (0, 1)
    :param surrogate: String specifying which surrogate loss function to use,
        defaults to 'xent'. 'xent': Cross-entropy surrogate. 'hinge': Hinge
        loss surrogate.
    :return: Array of samplewise surrogate AUROC losses.
    """
    
    y_torch = torch.tensor(y)
    logits_torch = torch.tensor(logit(y_pred))
    
    logits_difference_torch = logits_torch.unsqueeze(0) - logits_torch.unsqueeze(1)
    labels_difference_torch = y_torch.unsqueeze(0) - y_torch.unsqueeze(1)
    
    # matrex which is 1 if label y_i != label y_j
    abs_label_difference = torch.abs(labels_difference_torch)
    
    signed_logits_difference_torch = logits_difference_torch * labels_difference_torch
    
    if surrogate == 'xent':
        loss = torch.log(torch.sigmoid(signed_logits_difference_torch))
        loss = (abs_label_difference * loss).mean(axis=0) * 0.5
    elif surrogate == 'hinge':
        loss = torch.maximum(torch.zeros(1), torch.ones(1) - signed_logits_difference_torch)
        loss = (abs_label_difference * loss).mean(axis=0) * 0.5
        
    return np.array(loss.tolist())
    


def roc_auc_surrogate(y, y_pred, surrogate='xent'):
    
    pos_mask = (y == 1)
    neg_mask = (y == 0)
    
    if (np.sum(pos_mask) == 0) or (np.sum(neg_mask) == 0):
        raise Exception("Examples are either all positive or all negative")
                           
    logits = logit(y_pred)
        
    
    logits_difference = np.expand_dims(logits, 0) - np.expand_dims(logits, 1)
    labels_difference = np.expand_dims(y, 0) - np.expand_dims(y, 1)
    
    # if there were weights
    # weights_product = np.expand_dims(weights, 0) * np.expand_dims(weights, 1)
    
    signed_logits_difference = labels_difference * logits_difference
    
    # compute surrogate loss
    if surrogate == 'hinge':
        surr_fn = hinge_surrogate
    elif surrogate == 'xent':
        surr_fn = xent_surrogate
    
    surrogate_loss = surr_fn(np.ones_like(signed_logits_difference), signed_logits_difference)
    # 0 out entries where labels were the same
    proxy_auc_loss = np.abs(labels_difference) * surrogate_loss
    # np.mean(proxy_auc_loss, axis=0)
                             
    
    return proxy_auc_loss

    
    
def bootstrap_ci(y_true, y_pred, n_bootstrap=100, confidence=0.95, loss=roc_auc_score, return_samples=False):
    """Computes non-parametric bootstrap confidence interval for model
    performance.

    :param y_true: True target labels
    :param y_pred: Model predictions (be it regression predictions, probability
        predictions, or classification predictions).
    :param n_bootstrap: Number of bootstrap resamples to perform, defaults to
        100.
    :type n_bootstrap: int
    :param confidence: The confidence level for the interval as a decimal,
        defaults to 0.95
    :type confidence: Float, between 0 and 1
    :param loss: Loss function for computing average model performance. Should
        have signature 'loss(y_true, y_pred)', defaults to 
        sklearn.metrics.roc_auc_score
    :return: The mean performance, the lower interval, and the upper interval
        from the bootstrap samples.
    """
    
    n = y_true.shape[0]
        
    upper_p = 100 * (1. - (1. - confidence)/2)
    lower_p = 100 * ((1. - confidence)/2)
    
    aucs = []
    
    def bootstrap_resample_inds():
        return np.array(np.random.choice(range(n), n, replace=True))
    
    for i in range(n_bootstrap):
        inds = np.array(bootstrap_resample_inds())
        resample_true = y_true[inds]
        resample_pred = y_pred[inds]
        
        if loss==roc_auc_score:
            if (resample_true.mean() == 1) or (resample_true.mean() == 0):
                continue
        
        aucs.append(loss(resample_true, resample_pred))
    
    
    lower, upper= np.percentile(aucs, [lower_p, upper_p])
    if return_samples:
        return aucs
    return np.mean(aucs), lower, upper

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)