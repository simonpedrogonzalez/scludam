import os
import sys
from copy import deepcopy
from typing import Union

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from attr import attrs, attrib, validators
from hdbscan import HDBSCAN, all_points_membership_vectors
from sklearn.base import ClassifierMixin, ClusterMixin, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    normalized_mutual_info_score,
    pairwise_distances,
)
import pandas as pd
from sklearn.preprocessing import RobustScaler
from astropy.table.table import Table
from astropy.stats.sigma_clipping import sigma_clipped_stats

from itertools import permutations

sys.path.append(os.path.join(os.path.dirname("scludam"), "."))

from scludam.hkde import HKDE, PluginSelector
from scludam.utils import Colnames
from scludam.masker import RangeMasker, DistanceMasker, CrustMasker
from scludam.plot_gauss_err import plot_kernels
# from scludam.plots import plot3d_s, membership_3d_plot, membership_plot, probaplot


@attrs(auto_attribs=True)
class DBME:

    n_iters: int = 2
    iteration_atol: float = 0.01
    pdf_estimator: HKDE = HKDE()

    kde_leave1out: bool = True

    atol: float = 1e-2
    rtol: float = 1
    labeltol: int = 0

    kernel_calculation_mode: str = attrib(
        validator=validators.in_(["same", "per_class", "per_class_per_iter"]),
        default="per_class",
    )
    # internal attrs
    n: int = None
    d: int = None
    n_classes: int = None

    unique_labels: np.ndarray = None
    labels: np.ndarray = None
    priors: np.ndarray = None
    counts: np.ndarray = None
    posteriors: np.ndarray = None
    data: np.ndarray = None
    neff_iters: int = None

    estimators: list = attrib(factory=list)

    iter_priors: list = attrib(factory=list)
    iter_counts: list = attrib(factory=list)

    iter_log_likelihood_diff: list = attrib(factory=list)
    iter_log_likelihood_perc: list = attrib(factory=list)
    log_likelihoods: list = attrib(factory=list)
    iter_label_diff: list = attrib(factory=list)
    iter_labels: list = attrib(factory=list)
    """     
    noise_mask: np.ndarray = None
    permanent_noise: int = 0 """

    mixing_error: float = 1

    iter_dists: list = attrib(factory=list)
    dist: np.ndarray = None
    iter_test_proba: list = attrib(factory=list)

    diff_parametric: list = attrib(factory=list)

    def get_labels(self, posteriors):
        labels = np.argmax(posteriors, axis=1) - 1
        return labels

    def get_log_likelihood(self, densities):
        # calculate likelihood function from posteriors
        # should be L(X) = prod(i=1...N)(p(xi))
        # p(x)=sum(j=1...K)(wj*p(x|j))
        # in practice, the multiplication of small probabilities will underflow the float precision
        # so we calculate log(L(X)) using log multiplication property
        # log(a*b) = log(a)+log(b)
        # TODO: double and triple check, with normalized data also
        # return np.log((densities * (self.counts/self.n_obs)).sum(axis=1)).sum()
        total_density = densities.sum(axis=1, keepdims=True)
        cond_prob = densities / total_density
        return np.sum(np.log(np.sum(cond_prob * self.priors, axis=1)))

    def get_log_likelihood_diff(self):
        if len(self.log_likelihoods) > 1:
            ll_t_minus_1 = self.log_likelihoods[-2]
            ll_t = self.log_likelihoods[-1]
            # TODO: check
            log_diff = ll_t_minus_1 - ll_t
            increase_perc = (ll_t_minus_1 - ll_t) * 100.0 / ll_t_minus_1
            return log_diff, increase_perc
        return np.inf, np.inf

    def update_dist(self, posteriors):
        if self.dist is None:
            self.dist = pairwise_distances(RobustScaler().fit_transform(self.data))
        dists = list()
        l = self.labels.copy() + 1
        for i in range(1, posteriors.shape[1]):
            center = ((self.data.T * posteriors[:, i].ravel()).T).sum(
                axis=0
            ) / posteriors[:, i].sum()
            mean_mem_dist_to_center = (
                np.sum(
                    pairwise_distances(self.data, center.reshape(1, -1))
                    * np.atleast_2d(posteriors[:, i]).T
                )
                / posteriors[:, i].sum()
            )
            dists.append(mean_mem_dist_to_center)
        self.iter_dists.append(np.array(dists))

    def update_log_likelihood(self, densities):
        self.log_likelihoods.append(self.get_log_likelihood(densities))
        log_diff, increase_perc = self.get_log_likelihood_diff()
        self.iter_log_likelihood_perc.append(increase_perc)
        self.iter_log_likelihood_diff.append(log_diff)

    def update_diff_parametric(self, parametric):
        self.diff_parametric.append(
            np.abs(self.posteriors[:, 1] - parametric[:, 1]).mean()
        )

    def update_diff_test_proba(self, test_proba):
        interesting = np.zeros_like(test_proba).astype(bool)
        interesting[(test_proba > 0.02) & (test_proba < 0.98)] = True

        self.iter_test_proba.append(
            np.abs(self.posteriors[:, 1][interesting] - test_proba[interesting]).max()
        )

    def is_stopping_criteria_achieved(self):
        # (ln(LLt) - ln(LLt-1)) < atol
        # or increase_percentage < ptol
        if not self.iter_log_likelihood_diff or not self.iter_log_likelihood_perc:
            return False
        log_diff = self.iter_log_likelihood_diff[-1]
        increase_perc = self.iter_log_likelihood_perc[-1]
        if self.atol is not None and log_diff < self.atol:
            return True
        if self.rtol is not None and increase_perc < self.rtol:
            return True
        if (
            self.labeltol is not None
            and self.iter_label_diff
            and self.iter_label_diff[-1] == self.labeltol
        ):
            return True
        return False

    def update_class_mixtures(self, posteriors: np.ndarray):
        # test
        if self.mixing_error >= 0 and len(self.iter_priors) > 1:
            if (
                np.abs((self.iter_priors[0] - self.iter_priors[-1]))[0]
                < self.mixing_error
            ):
                self.labels = np.argmax(posteriors, axis=1) - 1
                self.iter_labels.append(self.labels)
                if len(self.iter_labels) > 1:
                    label_diff = (
                        (self.iter_labels[-1] != self.iter_labels[-2]).astype(int).sum()
                    )
                    self.iter_label_diff.append(label_diff)
                self.counts = posteriors.sum(axis=0)
                self.priors = self.counts / self.n
                self.iter_counts.append(self.counts)
                self.iter_priors.append(self.priors)
            else:
                self.iter_counts.append(self.counts)
                self.iter_priors.append(self.priors)
                self.labels = np.argmax(posteriors, axis=1) - 1
                self.iter_labels.append(self.labels)
                if len(self.iter_labels) > 1:
                    label_diff = (
                        (self.iter_labels[-1] != self.iter_labels[-2]).astype(int).sum()
                    )
                    self.iter_label_diff.append(label_diff)
        else:
            # end test
            self.labels = np.argmax(posteriors, axis=1) - 1
            self.iter_labels.append(self.labels)
            if len(self.iter_labels) > 1:
                label_diff = (
                    (self.iter_labels[-1] != self.iter_labels[-2]).astype(int).sum()
                )
                self.iter_label_diff.append(label_diff)
            self.counts = posteriors.sum(axis=0)
            self.priors = self.counts / self.n
            self.iter_counts.append(self.counts)
            self.iter_priors.append(self.priors)

    def get_posteriors(self, densities):
        # probability calculation
        # P(Ci|x) = Di(x) * P(Ci) / sumj(Dj(x) * P(Cj))
        total_density = (
            (densities * self.counts)
            .sum(axis=1, keepdims=True)
            .repeat(self.n_classes, axis=1)
        )
        posteriors = densities * self.counts / total_density
        return posteriors

    def get_posteriors2(self, densities):
        # probability calculation
        # P(Ci|x) = Di(x) / sumj(Dj(x))
        total_density = densities.sum(axis=1, keepdims=True).repeat(
            self.n_classes, axis=1
        )
        posteriors = densities / total_density
        return posteriors

    def get_posteriors3(self, densities):
        # probability calculation
        # P(Ci|x) = Di(x) / sumj(Dj(x))
        total_density = densities.sum(axis=1, keepdims=True).repeat(
            self.n_classes, axis=1
        )
        posteriors = densities / total_density

        yfie1 = HKDE().fit(self.data, weights=posteriors[:, 0]).pdf(self.data)
        yclu = 1 - yfie1
        new_posteriors = np.concatenate(
            (yfie1.reshape(-1, 1), yclu.reshape(-1, 1)), axis=1
        )

        return new_posteriors

    def get_densities(self, data: np.ndarray, err, corr, weights: np.ndarray):
        densities = np.zeros((self.n, self.n_classes))

        # estimator(s) fitting
        if not self.estimators or self.kernel_calculation_mode == "per_class_per_iter":
            if self.kernel_calculation_mode == "same":
                self.estimators = [self.pdf_estimator.fit(data, err, corr)]
            else:
                self.estimators = []
                for i in range(self.n_classes):
                    self.estimators.append(
                        deepcopy(self.pdf_estimator).fit(
                            data=data,
                            err=err,
                            corr=corr,
                            weights=weights[:, i],
                        ),
                    )

        # pdf estimation
        for i in range(self.n_classes):
            if self.kernel_calculation_mode == "same":
                class_estimator = self.estimators[0]
            else:
                class_estimator = self.estimators[i]
            densities[:, i] = class_estimator.set_weights(weights[:, i]).pdf(
                data, leave1out=self.kde_leave1out
            )

        return densities

    def fit(
        self,
        data: np.ndarray,
        init_proba: np.ndarray,
        err: np.ndarray = None,
        corr: Union[np.ndarray, str] = None,
        test_proba=None,
    ):

        self.n, self.d = data.shape
        self.data = data
        self.labels = np.argmax(init_proba, axis=1) - 1
        self.unique_labels = np.unique(self.labels)
        self.n_classes = len(self.unique_labels)

        self.posteriors = init_proba
        self.update_class_mixtures(posteriors=init_proba)

        # case no clusters found
        if self.n_classes == 1:
            # there are no populations to fit
            return self

        # p_parametric, mix = parametric(data, self.labels, self.priors)
        # f = lambda x: mix.predict_proba(x)[:,0]
        # testing
        # self.update_dist(self.posteriors)
        # self.update_diff_test_proba(test_proba)
        # end testing

        # self.update_diff_parametric(p_parametric)

        for i in range(self.n_iters):
            # is copy actually needed?
            previous_posteriors = self.posteriors.copy()
            weights = previous_posteriors
            # bad idea
            """ if self.noise_mask is not None:
                weights[self.noise_mask] = np.array([1.] + [0.]*(self.n_classes - 1)) """
            densities = self.get_densities(data, err, corr, weights)
            # self.posteriors = self.get_posteriors(densities)
            self.posteriors = self.get_posteriors(densities)
            self.update_class_mixtures(self.posteriors)
            # self.update_log_likelihood(densities)
            # testing
            # self.update_dist(self.posteriors)
            # self.update_diff_test_proba(test_proba)
            # grid = self.membership_plot()
            # lll = self.posteriors[:,1] > .5
            # mem_plot_kernels(index=1, dbme=self, ax=grid.axes[1,0], data=data, n=10000, labels=lll)
            # print(f'iter {i}')
            # plt.show()
            # plot3d_s(self.data, self.posteriors[:,1])
            # plt.show()
            # self.probaplot()
            # plt.show()

            # self.update_diff_parametric(p_parametric)
            """ nn = self.n_obs
            self.n_obs = 50**3
            g = lambda x: self.get_posteriors(x, None, None, weights=weights)[0][:,1]
            total_density = lambda x: np.atleast_2d((self.get_posteriors(x, None, None, weights=weights)[1] * self.priors).sum(axis=1, keepdims=True)).T
            total_density_only_sum = lambda x: np.atleast_2d((self.get_posteriors(x, None, None, weights=weights)[1]).sum(axis=1, keepdims=True)).T
            p_only_sum = lambda x: np.atleast_2d(np.atleast_2d(self.get_posteriors(x, None, None, weights=weights)[1][:,1]).T / total_density_only_sum(x)).T
            # p_check = lambda x: np.atleast_2d(np.atleast_2d(self.get_posteriors(x, None, None, weights=weights)[1][:,1]).T * self.priors[1] / total_density(x)).T
            gfig, gaxes = pair_density_plot(data, g, gr=50)
            gfig.suptitle('pc')
            tdfig, tdaxes = pair_density_plot(data, total_density, gr=50)
            tdfig.suptitle('cum_d_per_priors')
            tdosfig, tdosaxes = pair_density_plot(data, total_density_only_sum, gr=50)
            tdosfig.suptitle('cum_d_only')
            fig, ax = self.estimators[1].density_plot()
            fig.suptitle('dens_cum')
            #fig2, ax2 = pair_density_plot(data, p_check, gr=50)
            #fig2.suptitle('p_check')
            fig3, ax3 = pair_density_plot(data, p_only_sum, gr=50)
            fig3.suptitle('p_only_sum')
            plt.show()
            self.n_obs = nn
             """
            # end testing
            # self.is_stopping_criteria_achieved()
            # break
            # testing
            # self.membership_3d_plot(marker='o', palette='viridis_r', marker_size=100**(self.posteriors[:,1]))
            # end test
            self.neff_iters = i + 1
        return self

    def iter_plot(self, **kwargs):

        sns.set_style("darkgrid")

        df = pd.DataFrame(np.array(self.iter_counts))
        df.columns = [f"N{str(i)}" for i in range(self.iter_counts[0].shape[0])]
        df["t"] = np.arange(self.neff_iters + 1)

        fig, axes = plt.subplots(7, 1, sharex=True)

        axes[0].set_xticks(range(self.neff_iters + 1))

        for i in range(len(axes)):
            axes[i].axvline(1, color="black")
            axes[i].axvline(self.neff_iters, color="black")

        sns.lineplot(
            ax=axes[0], x="t", y="value", hue="variable", data=pd.melt(df, ["t"])
        )

        df["log_likelihood"] = np.array([np.nan] + self.log_likelihoods)
        df["ll_%"] = np.array([np.nan] + self.iter_log_likelihood_perc)
        df["ll_diff"] = np.array([np.nan] + self.iter_log_likelihood_diff)
        df["label_diff"] = np.array([np.nan] + self.iter_label_diff)

        sns.lineplot(ax=axes[1], x=df.t, y=df["log_likelihood"])

        sns.lineplot(ax=axes[2], x=df.t, y=df["ll_%"])
        if self.rtol is not None:
            axes[2].axhline(self.rtol, color="red")

        sns.lineplot(ax=axes[3], x=df.t, y=df["ll_diff"])
        if self.atol is not None:
            axes[3].axhline(self.atol, color="red")

        sns.lineplot(ax=axes[4], x=df.t, y=df["label_diff"])
        if self.labeltol is not None:
            axes[4].axhline(self.labeltol, color="red")

        # test
        dists = np.array(self.iter_dists)
        for i in range(self.n_classes - 1):
            df[f"dist{i}"] = dists[:, i]
            sns.lineplot(ax=axes[5], x=df.t, y=df[f"dist{i}"])

        """ df['diff_parametric'] = np.array(self.diff_parametric)
        sns.lineplot(ax=axes[6], x=df.t, y=df.diff_parametric) """
        # end test
        df["proba_diff"] = self.iter_test_proba
        sns.lineplot(ax=axes[6], x=df.t, y=df.proba_diff)

        return fig, axes

    # def membership_plot(self, label=0, **kwargs):
    #     return membership_plot(self.data, self.posteriors[:, label + 1], **kwargs)

    # def class_plot(self, **kwargs):
    #     return membership_plot(self.data, self.labels + 1, self.labels, **kwargs)

    # def membership_3d_plot(self, label=0, **kwargs):
    #     return membership_3d_plot(self.data, self.posteriors[:, label + 1], **kwargs)

    # def probaplot(self, **kwargs):
    #     return probaplot(
    #         data=self.data, proba=self.posteriors, labels=self.labels, **kwargs
    #     )


# def parametric(data, labels, priors):
#     import pomegranate as pg

#     f = pg.IndependentComponentsDistribution(
#         [
#             pg.UniformDistribution.from_samples(data[:, 0][labels == -1]),
#             pg.UniformDistribution.from_samples(data[:, 1][labels == -1]),
#             pg.UniformDistribution.from_samples(data[:, 2][labels == -1]),
#         ]
#     )

#     c = pg.IndependentComponentsDistribution(
#         [
#             pg.NormalDistribution.from_samples(data[:, 0][labels == 0]),
#             pg.NormalDistribution.from_samples(data[:, 1][labels == 0]),
#             pg.NormalDistribution.from_samples(data[:, 2][labels == 0]),
#         ]
#     )

#     mix = pg.GeneralMixtureModel([c, f])
#     mix.fit(data)
#     return mix.predict_proba(data), mix


# def test_membership():
#     np.random.seed(0)
#     df = one_cluster_sample_small(cluster_size=50, field_size=int(1e4))
#     data = df[["pmra", "pmdec", "parallax"]].to_numpy()

#     real_pmp = df["p_pm_cluster1"].to_numpy()
#     real_pmlabels = np.zeros_like(real_pmp)
#     real_pmlabels[real_pmp > 0.5] = 1

#     estimator = DensityBasedMembershipEstimator(
#         min_cluster_size=50,
#         n_iters=5,
#         pdf_estimator=HKDE(bw=PluginSelector(diag=True)),
#         iter_pdf_update=False,
#         mixing_error=1,
#     )
#     result = estimator.fit_predict(data)

#     calculated_pmp = result.p[:, 1]
#     calculated_labels = np.zeros_like(calculated_pmp)
#     calculated_labels[calculated_pmp > 0.5] = 1

#     acc = accuracy_score(real_pmlabels, calculated_labels)
#     conf = confusion_matrix(real_pmlabels, calculated_labels)
#     minfo = normalized_mutual_info_score(real_pmlabels, calculated_labels)

#     print("minfo")
#     print(minfo)
#     print("acc")
#     print(acc)
#     print("conf")
#     print(conf)
#     print("end")


# def mem_plot_kernels(dbme, ax, data, index=1, n=10, nstd=3, labels=None):
#     n = min(n, data.shape[0])
#     e = dbme.estimators[index]
#     if labels is None:
#         condition = dbme.labels == index - 1
#     else:
#         condition = labels
#     means = data[condition][:n]
#     cov = e.covariances[condition][:n]
#     return plot_kernels(
#         means=means,
#         cov_matrices=cov,
#         ax=ax,
#         alpha=1,
#         linewidth=0.5,
#         edgecolor="k",
#         facecolor="none",
#         nstd=nstd,
#     )


# # it just does not fucking work:
# # log_likelihood decreases instead of increasing
# # mixtures do not stabilize
# def test_membership_real():
#     s1_5 = "tests/data/clusters_phot/ngc2527.xml"
#     s2 = "ng2527_x2.xml"
#     s2_5 = "ng2527_x2.5.xml"
#     s3 = "ng2527_x3.xml"
#     s3_5 = "ng2527_x3.5.xml"
#     s2_5_phot = "ng2527_phot_x2.5.xml"
#     s15_mag = "scripts/data/clusters_phot/ngc2527bright1.csv"
#     s7_5 = "ngc2527_select_9_sigmas.xml"
#     s2_5_cured = "ng2527_cured_x2.5.xml"
#     s7_5_cured = "ng2527_cured_x7.5.xml"

#     print("reading")
#     df = Table.read(s7_5_cured).to_pandas()
#     cnames = Colnames(df.columns.to_list())
#     fiveparameters = ["pmra", "pmdec", "parallax", "ra", "dec"]
#     threeparameters = ["pmra", "pmdec", "parallax"]
#     twoparameters = ["pmra", "pmdec"]
#     datanames = cnames.get_data_names(fiveparameters)
#     errornames, missing_err = cnames.get_error_names(datanames)
#     corrnames, missing_corr = cnames.get_corr_names(datanames)
#     data = df[datanames].to_numpy()
#     err = df[errornames].to_numpy()
#     if missing_corr:
#         corr = None
#     else:
#         corr = df[corrnames].to_numpy()
#     n, d = data.shape
#     w = np.ones(n)
#     print("calculating")

#     scaled = RobustScaler().fit_transform(data)
#     mask = DistanceMasker(center='geometric', percentage=10).mask(data=scaled)
#     mask2 = CrustMasker(percentage=10).mask(data=scaled)
#     sns.scatterplot(data[:,0], data[:,1], hue=mask)

#     normal = 190
#     cured = 167
#     dbme = DensityBasedMembershipEstimator(
#         min_cluster_size=cured,
#         n_iters=10,
#         pdf_estimator=HKDE(bw=PluginSelector(diag=True)),
#         kernel_calculation_mode="per_class",
#         mixing_error=1,
#     )
#     result = dbme.fit_predict(data)  # , err=err, corr=corr)
#     dbme.iter_plot()
#     plt.show()
#     dbme.membership_plot(0, palette="viridis", density_intervals=10, colnames=datanames)

#     df["p"] = result.p[:, 1]
#     mems = df[df.p > 0.5]
#     nonmems = df[df.p <= 0.5]
#     sns.scatterplot(
#         mems.bp_rp, mems.phot_g_mean_mag, hue=mems.p, hue_norm=(0, 1)
#     ).invert_yaxis()
#     sns.scatterplot(
#         nonmems.bp_rp, nonmems.phot_g_mean_mag, hue=nonmems.p, hue_norm=(0, 1)
#     ).invert_yaxis()

#     plt.show()
#     print("coso")


# def test_cluster_selection():

#     np.random.seed(0)
#     df = three_clusters_sample(cluster_size=50, field_size=int(1e3))
#     data = df[["pmra", "pmdec", "parallax"]].to_numpy()
#     real_pmp = df["p_pm_cluster1"].to_numpy()
#     real_pmlabels = np.zeros_like(real_pmp)
#     real_pmlabels[real_pmp > 0.5] = 1

#     estimator = DensityBasedMembershipEstimator(
#         min_cluster_size=50,
#         min_samples=30,
#         cluster_centers=np.array([(8, 8, 5), (5, 5, 5)]),
#         # allow_single_cluster=True,
#         n_iters=30,
#         kernel_calculation_mode="per_class",
#         mixing_error=1,
#     )

#     estimator.fit_predict(data)
#     # estimator.iter_plot()
#     estimator.clustering_plot()
#     print("coso")


# # test_membership()
# # test_membership_real()
# # test_cluster_selection()
# # test_simul()
# from scludam.synthetic import case2_sample0c, case2_sample1c, case2_sample2c
# from scludam.shdbscan import SHDBSCAN, one_hot_encode


# def test_1c_pm():
#     fmix = 0.9
#     clu_size = int(1000 * (1 - fmix))
#     df = case2_sample1c(fmix)
#     test_proba = df["p_pm_cluster1"].to_numpy()

#     test_labels = np.zeros_like(test_proba)
#     test_labels[test_proba > 0.5] = 1
#     test_labels = test_labels - 1
#     init_proba = one_hot_encode(test_labels)

#     data = df[["pmra", "pmdec"]].to_numpy()

#     # clu = SHDBSCAN(min_cluster_size=clu_size, auto_allow_single_cluster=True, outlier_quantile=.9).fit(data)
#     # init_proba = clu.proba

#     dbme = DBME(n_iters=20).fit(data, init_proba, test_proba=test_proba)

#     print("coso")


# def test_1c_space():
#     fmix = 0.9
#     clu_size = int(1000 * (1 - fmix))
#     df = case2_sample1c(fmix)
#     test_proba = df["p_pm_cluster1"].to_numpy()
#     all_test_proba = df[["p_pm_field", "p_pm_cluster1"]].to_numpy()

#     test_labels = np.zeros_like(test_proba)
#     test_labels[test_proba > 0.5] = 1
#     test_labels = test_labels - 1
#     init_proba = one_hot_encode(test_labels)

#     data = df[["pmra", "pmdec"]].to_numpy()

#     # clu = SHDBSCAN(min_cluster_size=clu_size, auto_allow_single_cluster=True, outlier_quantile=.9).fit(data)
#     # init_proba = clu.proba

#     dbme = DBME(n_iters=20).fit(data, init_proba, test_proba=test_proba)

#     print("coso")


# # test_1c_space()
