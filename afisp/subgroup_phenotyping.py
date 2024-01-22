import numpy as np
from sklearn.base import BaseEstimator
import subprocess
from tqdm import tqdm
from afisp.utils import cohens_d
from statsmodels.stats.weightstats import ttest_ind
from imodels.rule_set.skope_rules import SkopeRulesClassifier
from pathlib import Path
import os


class SubgroupPhenotyper(BaseEstimator):
    """This class performs subgroup phenotyping. After using stability analysis
    to identify a poorly performing data subset, the SubgroupPhenotyper is used
    to find specific data phenotypes or subgroups that are present within the
    data subset.
    """

    def __init__(self):
        """Constructor method
        """
        self.fit_called_ = False

    def fit(self, subgroup_feature_data, subset_labels, test_loss, 
            method="DecisionList", depth=2, cv=False, rule_max=50, 
            p0=0.025, input_fname="data_for_sirus.csv", 
            output_fname="sirus_rules.txt", verbose=0):
        """Computes the subgroup phenotypes using an interpretable classifier.
        The subgroup phenotyper expects categorical features to encoded as
        binary dummy variables.

        :param subgroup_feature_data: Array containing the subgroup feature
            data.
        :param subset_labels: Binary labels for whether each sample is in the
            worst data subset.
        :param test_loss: The observed loss for each sample. Used for filtering
            rules based on statistical significance and effect size.
        :param method: Selects the interpretable classification method used for
            extracting the subgroup phenotypes. "SIRUS" uses the 'Stable and
            Interpretable RUle Set' method implemented in R. It requires a
            working R distribution with the 'sirus' package installed. This is
            recommended. As a python alternative, the "DecisionList" will use
            the SkopeRules DecisionListClassifier, defaults to "DecisionList'.
        :param cv: For method SIRUS, whether or not to use cross-validation to
            select the rule selection threshold. If True, then p0 is ignored.
            Using cross-validation can be very slow, but results are often
            better.
        :param p0: For method SIRUS, the threshold (between 0 and 1) for rule
            selection. Ignored if cv is True, since cv is used to determine a
            good value for p0. The higher the value of p0, the fewer rules that
            will be selected. Recommended to choose a value < 0.1, defaults to
            0.025.
        :type p0: Float in (0, 1)
        :param rule_max: The max number of rules that SIRUS will consider,
            defaults to 50.
        :type rule_max: int > 0
        :param depth:
        :return: The candidate rules extracted by the Subgroup Phenotyper
        :rtype: List[string]
        """
        phenotype_df = subgroup_feature_data.copy()
        phenotype_df['subset_label'] = subset_labels

        if method == "SIRUS":
            # need to check that we have R installed with SIRUS package
            phenotype_df.to_csv(input_fname, index=False)
            package_path = Path(__file__).parent
            command = f"Rscript {package_path}/run_sirus.r"
            command += f" --input {input_fname} --output {output_fname}"
            command += f" --depth {depth}"
            command += f" --rule.max {rule_max}"
            if cv:
                command += f" --cv"
            else:
                command += f" --p0 {p0}"
            # `cwd`: current directory is straightforward
            cwd = Path.cwd()

            # `mod_path`: According to the accepted answer and combine with future power
            # if we are in the `helper_script.py`
            if verbose > 0:
                print("Beginning call to SIRUS. If cv == True this may take a long time.")
            subprocess.call((command), shell=True)
            # subprocess.call((f"Rscript" 
            #                  f" afisp/run_sirus.r" 
            #                  f" --input {df_fname} "
            #                  f" --output {sirus_rules_fname}"
            #                  f" --depth {depth}"
            #                  f" --rule.max {rule_max}"
            #                  f" --p0 0.027"),
            # shell=True)
            if verbose > 0:
                print("Finished call to SIRUS")
            candidate_rules = self._get_sirus_rules(output_fname)
            # clean up temporary files
            if os.path.exists(output_fname):
                os.remove(output_fname)
            if os.path.exists(input_fname):
                os.remove(input_fname)
        elif method == "DecisionList":
            # Add arguments for SkopesRulesClassifier
            if depth == 1:
                md = 1
            else:
                md = list(range(1, depth+1))
            sp = SkopeRulesClassifier(max_depth=md)
            sp.fit(subgroup_feature_data.values, 
                   subset_labels, 
                   feature_names=subgroup_feature_data.columns)
            candidate_rules = sp.rules_

        else:
            raise RuntimeError('Method not implemented. Please choose one of "SIRUS" or "DecisionList"')

        if verbose > 0:
            print("Computing p-values")
        rule_p_values = self._precompute_p_values(candidate_rules, phenotype_df, test_loss)
        significant_rules = self._holm_bonferroni_correction(rule_p_values)
        if verbose > 0:
            print("Effect size filtering")
        extracted_rules = self._effect_size_filtering(significant_rules, phenotype_df, test_loss, 
                                                        effect_threshold=0.3)
        self.fit_called_ = True
        self._extracted_rules = extracted_rules
        return self._extracted_rules

    def _negate_simple_rule(self, rule):
        """Negates a rule string of the form "if ARG [<|>|<=|>=] VAL"

        Args:
            rule: A string of a rule of the form "if ARG [<|>|<=|>=] VAL"
        Returns:
            String in which the condition [>|<|>=|<=] has been negated
        """
        if ">=" in rule:
            return(rule.replace(">=", "<"))
        if "<=" in rule:
            return(rule.replace("<=", ">"))
        if ">" in rule:
            return(rule.replace(">", "<="))
        if "<" in rule:
            return(rule.replace("<", ">="))

    def _get_sirus_rules(self, filename):

        with open(filename) as f:
            filelines = [line for line in f]
        
        sirus_rules = set()
        
        for line in filelines:
            if " then " not in line:
                continue
            # every rule reads as 'if RULE then ...'
            end = line.index(" then ")
            rule = line[3:end].strip()
            if_prob = float(line.split('then')[1].split()[0])
            else_prob = float(line.split('else')[1].split()[0])
                
            if if_prob > else_prob:# can take the rule as is
                sirus_rules.add(rule)
            elif '&' in rule: # negate a compound rule
                # ~(X & Y) = ~X or ~Y
                # new_rules = []
                for r in rule.split('&'): 
                    # new_rules.append(negate_simple_rule(r.strip()))
                    sirus_rules.add(self._negate_simple_rule(r.strip()))
                # sirus_rules.add(' | '.join(new_rules))
            else: # negate a simple rule
                sirus_rules.add(self._negate_simple_rule(rule))

        return(sirus_rules)
    
    def _precompute_p_values(self, sirus_rules, phenotype_df, test_loss, alpha=0.05):
        # precompute p_values
        rule_p_values = []
        for rule in tqdm(sirus_rules):
            rows = phenotype_df.eval(str(rule))
                
                
            # two independent sample t test that brier score is larger in group than out of group
            pval = ttest_ind(test_loss[rows], 
                            x2=test_loss[~rows], 
                            value=0.,
                            alternative='larger',
                            usevar='unequal')[1]
            # check for nan
            
            # print(rule, rows.sum())
            # print(f"Mean AUC {np.mean(bootstrap_aucs):.3f} Threshold AUC {performance_threshold:.3f} p value {pval:.2f} ")
            rule_p_values.append((rule, pval, rows.sum()))
        # rule_p_values = sorted(rule_p_values, key=lambda x: x[1])
        # filter out nans
        rule_p_values = sorted([x for x in rule_p_values if not np.isnan(x[1])], key=lambda x: x[1])
        return(rule_p_values)
    
    def _holm_bonferroni_correction(self, rule_p_values, sig_value=0.05):
        # expect p-values to be sorted, smallest to largest
        
        m = len(rule_p_values)
        
        significant_rules = []
        
        for k in range(1, m+1):
            rule_k = rule_p_values[k-1]
            # reject with adaptive significance level
            if rule_k[1] < sig_value / (m + 1 - k):
                significant_rules.append(rule_k)
                continue
            # print(k, rule_k, sig_value / (m + 1 - k))
            break

        return(significant_rules)
    
    def _effect_size_filtering(self, significant_rules, phenotype_df, 
                               test_loss, effect_threshold = 0.4, 
                               verbose=False):
        rules = []
        for rule in tqdm(significant_rules):
            r = rule[0]
            rows = phenotype_df.eval(str(r))
            cd = cohens_d(test_loss[rows], test_loss[~rows])
            if verbose:
                print(f"{r} Cohen's d {cd:.2f}")
            if cd > effect_threshold:
                rules.append((r, cd))
        # sort from highest effect size to lowest
        rules = sorted(rules, key=lambda x: x[1], reverse=True)
        rules = [r[0] for r in rules] # drop the effect sizes
        return(rules)


