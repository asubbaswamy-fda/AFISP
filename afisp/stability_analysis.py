import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.base import clone
from tqdm import tqdm
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from afisp.utils import cohens_d



#Old name: LatentSubgroupShiftEstimator    
class WorstSubsetFinder(BaseEstimator):
    """This is a class for performing a Stability Analysis of a trained machine
    learning (ML) model. Given a test dataset, a WorstSubsetFinder object 
    identifies a data subset of a particular sample size that produces the
    worst performance (i.e., a worst-case data subset of a given size). A worst
    case data subset corresponds to an adversarial covariate shift in the 
    distribution of user-specified features.

    :param subset_fractions: A list of subset fractions between 0 and 1. For 
        each subset fraction, the WorstSubsetFinder identifies the worst
        performing subset of approximately that subset fraction size (i.e., for
        a subset fraction f and a dataset of size N, the subset will be size
        approximately f*N, defaults to [0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0].
    :type subset_fractions: List[float]
    :param conditional_loss_model: A supervised learning model with sklearn 
        interface for estimating the expected conditional loss of the ML model
        to be evaluated, defaults to None which will fit an 
        ExplainableBoostingRegressor.
    :param cv: Number of folds for cross validation to predict the loss for
        each data sample, defaults to 10.
    :type cv: int
    :param verbose: If True, then prints information during calls to self.fit,
        defaults to False.
    :type verbose: bool
    :param eps: Corresponds to the max amount of noise to add to conditional
        loss estimates. It is used to break ties between samples with the same
        expected loss. It should be set to a small positive value if many
        subgroup features are discrete, defaults to 0.
    :type eps: float
    """
    
    def __init__(self,
                 subset_fractions=[0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0],
                 conditional_loss_model=None,
                 cv = 10,
                 verbose=False,
                 eps=0.0):
        """Constructor method
        """
        if eps < 0:
            raise RuntimeError('eps must be a float >= 0')
        self.__dict__.update(locals())
        self.fit_called_ = False
        self._masks = None
        self._effect_sizes_computed = False
        # self.fit_mu_ = False
        
    def fit(self, subgroup_feature_data, 
            samplewise_losses, feature_names=None):
        """Computes the worst-case data subsets for each subset fraction and
        returns the average loss on each subset.

        :param subgroup_features: A numpy array of dim (nsamples, nfeatures)
            containing the subgroup defining feature values for each sample.
        :type subgroup_features: ((nsamples, nfeatures), ndarray)
        :param samplewise_losses: A numpy array of the per-sample observed loss
        :type samplewise_losses: ((nsamples,), array)
        :param feature_names: A list of names for the subgroup
            characterizing features, optional
        :type feature_names: List[string], optional
        :return: An array containing the average loss on each worst-case subset
        :rtype: (# of subset_fractions, array)
        """
        self._samplewise_losses = samplewise_losses
        # by default we will use Explainable Boosting Machines (EBMs) since
        # we have observed that these models work well out of the box, without
        # any hyperparameter tuning.
        self.mu_mdl = self.conditional_loss_model
        if self.conditional_loss_model is None:
            self.mu_mdl = ExplainableBoostingRegressor(
                feature_names=feature_names
            )
        
        X = subgroup_feature_data
        N = X.shape[0]
        self.num_samples_ = N
        num_subsets = len(self.subset_fractions)
        
        # random noise for when subgroup feature data is all discrete
        U = np.random.rand(N,1)*self.eps
        
        # mu hat will contain our expected conditional loss estimates
        # one per sample
        self.mu_hat_ = np.zeros(N)
        
        # eta hat will contain the sample-specific thresholds corresponding
        # to quantiles of the expected conditional loss. Each sample will
        # have a threshold per subset fraction. The quantile is equal to
        # 1 - the sample fraction.
        self.eta_hat_ = np.zeros((num_subsets,N))
        
        # intermediate computation
        self.h_hat_ = np.zeros((num_subsets,N))
        
        # stores the expected conditional loss model for each fold
        self.mu_mdls_ = []
        
        k = 0
        folds = KFold(self.cv, shuffle=False).split(subgroup_feature_data)
        for train_idxs,test_idxs in folds:
            if self.verbose: print(f"k = {k}")
            k += 1
            
            # fit model to fold                        
            mu_mdl_k = clone(self.mu_mdl).fit(X[train_idxs],
                                              samplewise_losses[train_idxs])
            self.mu_hat_[test_idxs] = mu_mdl_k.predict(X[test_idxs])
            self.mu_mdls_.append(mu_mdl_k)
                        
            # Note: for a given alpha value, the eta value for samples will
            # vary depending on which fold the sample is in
            for a, alpha in enumerate(self.subset_fractions):
                # compute the 1-alpha quantile of the expected cond loss
                # this is the eta value for all samples in this fold
                self.eta_hat_[a, test_idxs] = np.repeat(
                            np.quantile(self.mu_hat_[test_idxs], 1.-alpha), 
                            len(test_idxs)
                )
            
        
        
        # R hat will contain the worst case average loss for each subset
        self.R_hats_ = np.zeros(num_subsets)
        
        # variance estimates and confidence intervals for worst case
        # average loss
        self.sigma_hats_ = np.zeros(num_subsets)
        self.cis_ = np.zeros((num_subsets,2))
            
        for a, alpha in enumerate(self.subset_fractions):
            self.h_hat_[a] = 1.0*(self.mu_hat_ + U[:,0] >= self.eta_hat_[a])
        
            psi = np.maximum(self.mu_hat_ + U[:,0] - self.eta_hat_[a],0.0)/alpha + self.eta_hat_[a]
            psi += self.h_hat_[a]*(samplewise_losses - self.mu_hat_)/alpha
    
            self.R_hats_[a] = np.mean(psi)
            self.sigma_hats_[a] = np.sqrt(np.mean((psi - self.R_hats_[a])**2))
            
            # lower confidence interval
            self.cis_[a,0] = self.R_hats_[a] - 1.96*self.sigma_hats_[a]/np.sqrt(N)
            # upper confidence interval
            self.cis_[a,1] = self.R_hats_[a] + 1.96*self.sigma_hats_[a]/np.sqrt(N)
        
        self.fit_called_ = True
        return self.R_hats_
    
    def confidence_intervals(self):
        """Returns analytical confidence intervals for the worst-case loss
        estimates for each subset fraction size.

        :return: Array with first column containing lower confidence interval 
            and second column containing upper confidence interval.
        :rtype: ((# of subset fraction sizes, 2),ndarray)
        """
        
        if not self.fit_called_:
            raise RuntimeError('Must call "fit" on WorstSubsetFinder object first.')
        return self.cis_
    
    def subset_masks(self):
        """Computes a boolean mask for indexing worst-case data subsets.

        :return: List of boolean masks for indexing the worst-case data
            subsets corresponding to each subset fraction
        :rtype: List[(# of samples, boolean ndarray)]
        """
        if not self.fit_called_:
            raise RuntimeError('Must call "fit" on WorstSubsetFinder object first.')
        # find the samples for which expected conditional loss is greater
        # than the quantile.
        # only compute once
        if self._masks is None:
            self._masks = [self.mu_hat_ >= self.eta_hat_[a] 
                for a in range(len(self.subset_fractions))]
        
        return self._masks
    
    def check_subset_sizes(self, plot=True, ax=None):
        """A visual diagnostic for checking subset sizes are correct.

        :param plot: Whether or not to plot the results, defaults to True
        :type plot: bool
        :param ax: Matplotlib axis to plot on if plot is True, defaults to
            None, optional
        :type ax: Matplotlib axis
        :return: Two lists, the first containing the subset fractions and the
            second containing the identified fractions. When plotted against
            each other the relationship should match the line y=x.
        :rtype: (List[float], List[float])
        """
        
        if not self.fit_called_:
            raise RuntimeError('Must call "fit" on WorstSubsetFinder object first.')
        observed_fractions = [np.mean(m) for m in self.subset_masks()]
        
        if plot:
            if ax is None:
                ax = plt.gca()
            ax.plot(self.subset_fractions, observed_fractions)
            ax.plot([0,1],[0,1],'k:', label='Perfect fit')
            ax.set_xlabel('Subset Fraction')
            ax.set_ylabel('Fraction Selected by Worst-Case Mask')
            ax.legend(loc='best')

        
        return self.subset_fractions, observed_fractions
    
    def compute_effect_sizes(self, plot=False, ax=None):
        """Compute the effect size (Cohen's d) of the difference in the average
        loss for each worst-case subset and the full test dataset. Can also
        plot the effect sizes vs subset fraction.

        :param plot: Whether or not to plot the results, defaults to False
        :type plot: bool
        :param ax: Matplotlib axis to plot on if plot is True, defaults to
            None, optional
        :type ax: Matplotlib axis
        :return: Two lists, the first containing the subset fractions and the
            second containing the effect size of the difference in the average
            loss in the worst subset and the full test dataset.
        :rtype: (List[float], List[float])
        """
        if not self.fit_called_:
            raise RuntimeError('Must call "fit" on WorstSubsetFinder object first.')

        cds = []
        # p_vals = []

        for i, a in enumerate(self.subset_fractions):
            idxs = self._masks[i]
            odxs = self._masks[i]
            cds.append(cohens_d(self._samplewise_losses[idxs], 
                                      self._samplewise_losses))
            # could also compute statistical significance if desired
            # pval = ttest_ind(self._samplewise_losses[idxs], 
            #                     x2=self._samplewise_losses[odxs], 
            #                     value=0.,
            #                     alternative='larger',
            #                     usevar='unequal')[1]
            # p_vals.append(pval)

        if plot:
            if ax is None:
                ax = plt.gca()
            ax.plot(self.subset_fractions, cds)
            ax.set_xlabel('Subset Fraction')
            ax.set_ylabel("Cohen's d (Effect Size) of Loss")
        
        self._effect_sizes = cds
        self._effect_sizes_computed = True
        return self.subset_fractions, cds

    def find_max_effect_size(self):
        """Finds the index of the worst subset producing the max effect size
        for the difference in the average loss of the subset and full dataset. 

        :return: The index of the subset producing the max effect size and the
        max effect size.
        :rtype: (int, float)
        """

        if not self._effect_sizes_computed:
            self.compute_effect_sizes(plot=False)
        return np.argmax(self._effect_sizes), max(self._effect_sizes)
    

# This is a naive clustering approach that is an alternative to AFISP
# It theoretically and empirically is worse than AFISP, but may add support
# in a future version
# class ClusteringEstimator(BaseEstimator):

#     def __init__(self, kmax=5, verbose=False):
#         self.__dict__.update(locals())

#         self.fit_called_ = False
#         self.masks_ = None


#     def fit(self, subgroup_feature_data, samplewise_losses, feature_names=None):

#         models = []
#         cluster_inds = []
#         worst_perf = []
#         masks = []
#         def iteration(k):
#             model = KMeans(n_clusters=k, n_init=10)
#             preds = model.fit_predict(subgroup_feature_data)
#             c_losses = np.zeros(k)
#             for i in range(k):
#                 c_losses[i] = samplewise_losses[model.labels_ == i].mean()
#             c_idx = np.argmax(c_losses)
#             worst_perf.append(max(c_losses))
#             cluster_inds.append(preds)
#             models.append(model)
#             masks.append(model.labels_ == c_idx)
        
#         if self.verbose:
#             for k in tqdm(range(2, self.kmax+1)):
#                 iteration(k)
#         else:
            
#             for k in range(2, self.kmax+1):
#                 iteration(k)
                
#         idx = np.argmax(worst_perf)
#         self.model_ = models[idx]
#         self.masks_ = masks[idx]
#         self.fit_called_ = True