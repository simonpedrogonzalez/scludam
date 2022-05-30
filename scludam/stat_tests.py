import os
import sys
from abc import abstractmethod

from astropy.stats import RipleysKEstimator
from attrs import define, validators, field
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from unidip.dip import diptst
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from warnings import warn
from scipy.stats import ks_2samp, multivariate_normal

sys.path.append(os.path.join(os.path.dirname("scludam"), "."))
from scludam.synthetic import (
    Cluster,
    Field,
    Synthetic,
    UniformSphere,
    polar_to_cartesian,
    stats,
    case2_sample0c,
    case2_sample1c,
    case2_sample2c,
    BivariateUnifom,
)
from scludam.utils import combinations


@define(auto_attribs=True)
class TestResult:
    passed: bool = None


class StatTest:
    @abstractmethod
    def test(self, data: np.ndarray, *args, **kwargs) -> TestResult:
        pass


@define(auto_attribs=True)
class HopkinsTestResult(TestResult):
    value: float = None
    pvalue: float = None


@define(auto_attribs=True)
class RipleyKTestResult(TestResult):
    value: float = None
    radii: np.ndarray = None
    l_function: np.ndarray = None


@define(auto_attribs=True)
class DipDistTestResult(TestResult):
    pvalue: float = None
    dist: np.ndarray = None


@define(auto_attribs=True)
class HopkinsTest(StatTest):
    n_samples: int = None
    metric: str = "euclidean"
    n_iters: int = 100
    # reduction: Callable = np.median
    # interpretation:
    # H0: data comes from uniform distribution
    # H1: data does not come from uniform distribution
    # if h = u/(u+w) ~ 1 => w = 0 luego hay estructura
    # if h = u/(u+w) ~ .5 => w ~ u luego no hay estructura
    # if h > .75 => reject H0, and in general  indicates a clustering
    # tendency at the 90% confidence level.
    threshold: float = None
    pvalue_threshold: float = 0.05

    def get_pvalue(self, value, n_samples):
        """
        Parameters
        ----------
        value : float
            The hopkins score of the dataset (between 0 and 1)
        n_samples : int
            The number of samples used to compute the hopkins score

        Returns
        ---------------------
        pvalue : float
            The pvalue of the hopkins score
        """
        beta = stats.beta(n_samples, n_samples)
        if value > 0.5:
            return 1 - (beta.cdf(value) - beta.cdf(1 - value))
        else:
            return 1 - (beta.cdf(1 - value) - beta.cdf(value))

    def test(self, data: np.ndarray, *args, **kwargs):
        """Assess the clusterability of a dataset. A score
        between 0 and 1, a score around 0.5 express
        no clusterability and a score tending to 1
        express a high cluster tendency.

        Parameters
        ----------
        data : numpy array
            The input dataset

        Returns
        ---------------------
        score : float
            The hopkins score of the dataset (between 0 and 1)

        Examples
        --------
        >>> from sklearn import datasets
        >>> from pyclustertend import hopkins
        >>> X = datasets.load_iris().data
        >>> hopkins(X,150)
        0.16
        """
        assert len(data.shape) == 2

        obs, dims = data.shape

        if self.n_samples is None:
            n_samples = int(obs * 0.1)
        else:
            n_samples = min(obs, self.n_samples)

        results = []
        for i in range(self.n_iters):
            sample = resample(data, n_samples=n_samples, replace=False)
            if self.metric == "mahalanobis":
                kwargs["V"] = np.cov(sample, rowvar=False)
            tree = BallTree(sample, leaf_size=2, metric=self.metric, *args, **kwargs)
            dist, _ = tree.query(sample, k=2)
            sample_nn_distance = dist[:, 1]

            max_data = data.max(axis=0)
            min_data = data.min(axis=0)
            uniform_sample = np.random.uniform(
                low=min_data, high=max_data, size=(n_samples, dims)
            )

            dist, _ = tree.query(uniform_sample, k=1)
            uniform_nn_distance = dist[:, 0]

            sample_sum = np.sum(sample_nn_distance**dims)
            uniform_sum = np.sum(uniform_nn_distance**dims)
            # sample_sum = self.reduction(sample_nn_distance)
            # uniform_sum = self.reduction(uniform_nn_distance)
            if sample_sum + uniform_sum == 0:
                raise Exception("The denominator of the hopkins statistics is null")
            results.append(uniform_sum / (uniform_sum + sample_sum))

        value = np.median(np.array(results))
        pvalue = self.get_pvalue(value, n_samples)
        if self.threshold is not None:
            passed = value >= self.threshold
        else:
            passed = pvalue <= self.pvalue_threshold
        return HopkinsTestResult(value=value, passed=passed, pvalue=pvalue)


@define(auto_attribs=True)
class DipDistTest(StatTest):
    n_samples: int = None
    metric: str = "euclidean"
    pvalue_threshold: float = 0.05

    def test(self, data: np.ndarray, *args, **kwargs):
        """dip test of unimodality over multidimensional
        data based on distance metric"""
        assert len(data.shape) == 2

        obs, dims = data.shape

        if self.n_samples is None:
            n_samples = min(obs, 100)
        else:
            n_samples = min(obs, self.n_samples)

        sample = resample(data, n_samples=n_samples, replace=False)
        dist = np.ravel(np.tril(pairwise_distances(sample, metric=self.metric)))
        dist = np.msort(dist[dist > 0])
        _, pval, _ = diptst(dist, *args, **kwargs)
        # sns.histplot(dist).set(title=str(pval))
        # plt.show()
        # print(pval)
        passed = pval < self.pvalue_threshold
        return DipDistTestResult(pvalue=pval, passed=passed, dist=dist)


@define
class RipleysKTest(StatTest):
    rk_estimator: RipleysKEstimator = None

    scaler = MinMaxScaler()

    ripley_factors = {
        0.05: 1.42,
        0.01: 1.68,
    }

    chiu_factors = {
        0.1: 1.31,
        0.05: 1.45,
        0.01: 1.75,
    }

    mode: str = field(
        validator=validators.in_(["ripley", "chiu", "ks"]), default="ripley"
    )

    factor: float = None

    pvalue_threshold: float = field(default=0.05)

    @pvalue_threshold.validator
    def _check_pvalue_threshold(self, attribute, value):
        if self.factor is None:
            if self.mode == "ripley" and value not in self.ripley_factors.keys():
                raise ValueError(
                    f"{value} is not a valid pvalue threshold for {self.mode} rule. Must be one of {self.ripley_factors.keys()}"
                )
            elif self.mode == "chiu" and value not in self.chiu_factors.keys():
                raise ValueError(
                    f"{value} is not a valid pvalue threshold for {self.mode} rule. Must be one of {self.chiu_factors.keys()}"
                )
        elif value <= 0 or value >= 1:
            raise ValueError(
                f"{value} is not a valid pvalue threshold. Must be between 0 and 1."
            )

    def empirical_csr_rule(self, l_function, radii, area, n):
        supremum = np.max(np.abs(l_function - radii))
        if self.factor:
            factor = self.factor
        elif self.mode == "ripley":
            factor = self.ripley_factors[self.pvalue_threshold]
        else:
            factor = self.chiu_factors[self.pvalue_threshold]
        return supremum, supremum >= factor * np.sqrt(area) / n

    def ks_rule(self, l_function, radii):
        pvalue = ks_2samp(l_function, radii).pvalue
        return pvalue, pvalue <= self.pvalue_threshold

    def test(self, data: np.ndarray, radii: np.ndarray = None, *args, **kwargs):

        data_unique = np.unique(data, axis=0)
        if data_unique.shape[0] != data.shape[0]:
            warn(
                "There are repeated data points that cause astropy.stats.RipleysKEstimator to break, they will be removed."
            )
            data = data_unique

        obs, dims = data.shape
        if dims != 2:
            raise ValueError("Data must be bidimensional.")

        if self.scaler is not None:
            data = self.scaler.fit_transform(data)

        x_min = data[:, 0].min()
        x_max = data[:, 0].max()
        y_min = data[:, 1].min()
        y_max = data[:, 1].max()

        if radii is None:
            # considers rectangular window
            # based on spatstat rmax.rule
            short_side = min(x_max - x_min, y_max - y_min)
            radii_max_ripley = short_side / 4
            radii_max_large = np.sqrt(1000 / (np.pi * obs))
            radii_max = min(radii_max_ripley, radii_max_large)
            step = radii_max / 128 / 4
            radii = np.arange(0, radii_max + step, step)

        if self.rk_estimator is None:
            # area = ConvexHull(points=data).volume
            area = (x_max - x_min) * (y_max - y_min)
            self.rk_estimator = RipleysKEstimator(
                area=area,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
        else:
            area = self.rk_estimator.area

        if kwargs.get("mode") is None:
            # best mode for rectangular window
            kwargs["mode"] = "ripley"

        l_function = self.rk_estimator.Lfunction(data, radii, *args, **kwargs)

        if self.mode == "ks":
            value, passed = self.ks_rule(l_function, radii)
        else:
            value, passed = self.empirical_csr_rule(l_function, radii, area, obs)

        print(self.ks_rule(l_function, radii))
        print(f"{value} {passed}")

        return RipleyKTestResult(
            value=value, passed=passed, radii=radii, l_function=l_function
        )


def test_dip():
    ns = [100, 1000, int(1e4)]
    # case uniform
    uniforms = [
        UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=10).rvs(n)
        for n in ns
    ]

    results_euclidean = np.array(
        [DipTest(metric="euclidean").test(u) for u in uniforms]
    )
    results_mahalanobis = np.array([DipTest().test(u) for u in uniforms])

    print(results_euclidean)
    print(results_mahalanobis)
    assert np.all(results_euclidean > 0.05)
    assert np.all(results_mahalanobis > 0.05)

    # case 1 k
    any_pm = stats.multivariate_normal(mean=(7.5, 7), cov=1.0 / 35)
    c_f_mixes = [0.1, 0.5, 0.9]
    cov_diag = 0.5
    random_matrix = np.random.rand(3, 3)
    cov_full = [np.dot(rm, rm.T) for rm in random_matrix]
    covs = [cov_diag, cov_full]
    metrics = ["euclidean", "mahalanobis"]
    parameters = combinations([ns, c_f_mixes, covs, metrics])
    oneclusters = [
        Synthetic(
            field=Field(
                space=UniformSphere(
                    center=polar_to_cartesian((120.5, -27.5, 5)), radius=10
                ),
                pm=any_pm,
                star_count=int(p[0] * (1 - p[1])),
            ),
            clusters=[
                Cluster(
                    space=stats.multivariate_normal(
                        mean=polar_to_cartesian([120.7, -28.5, 5]),
                        cov=p[2],
                    ),
                    pm=any_pm,
                    star_count=int(p[0] * p[1]),
                ),
            ],
            representation_type="cartesian",
        )
        .rvs()[["x", "y", "z"]]
        .to_numpy()
        for p in parameters
    ]
    results = [DipTest(metric=p[3]).test(oc) for oc, p in zip(oneclusters, parameters)]
    a = [(p[0], p[1], r) for p, r in zip(parameters, results)]
    print(results)
    print("coso")
    assert np.all(results < 0.2)

    # NOTE: "fails" when mix is too imbalanced, e.g. .1 to .9, or viceversa
    # Meaning:
    # if dip > .1
    # there is unimodal tendency, there are no clusters or there is only
    # one cluster and no noise
    # if dip < .1
    # if there is multimodal tendency
    #   there are several clusters
    #   or one cluster + noise

    # case 2 k
    any_pm = stats.multivariate_normal(mean=(7.5, 7), cov=1.0 / 35)
    c_f_mixes = [0.1, 0.5, 0.9]
    cov_diag = 0.5
    random_matrix = np.random.rand(3, 3)
    cov_full = [np.dot(rm, rm.T) for rm in random_matrix]
    covs = [cov_diag, cov_full]
    metrics = ["euclidean", "mahalanobis"]
    parameters = combinations([ns, c_f_mixes, covs, metrics])

    twoclusters = [
        Synthetic(
            field=Field(
                space=UniformSphere(
                    center=polar_to_cartesian((120.5, -27.5, 5)), radius=10
                ),
                pm=any_pm,
                star_count=int(p[0] * (1 - p[1])),
            ),
            clusters=[
                Cluster(
                    space=stats.multivariate_normal(
                        mean=polar_to_cartesian([119.5, -28.5, 4.8]),
                        cov=p[2],
                    ),
                    pm=any_pm,
                    star_count=int(p[0] * p[1]),
                ),
                Cluster(
                    space=stats.multivariate_normal(
                        mean=polar_to_cartesian([121.5, -26.5, 5.2]),
                        cov=p[2],
                    ),
                    pm=any_pm,
                    star_count=int(p[0] * p[1]),
                ),
            ],
            representation_type="cartesian",
        )
        .rvs()[["x", "y", "z"]]
        .to_numpy()
        for p in parameters
    ]
    results = [DipTest(metric=p[3]).test(oc) for oc, p in zip(twoclusters, parameters)]
    a = [(p[0], p[1], r) for p, r in zip(parameters, results)]
    print(a)
    print(results)
    print("coso")


def test_hopkins():
    ns = [100, 1000, int(1e4)]
    # case uniform
    uniforms = [
        UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=10).rvs(n)
        for n in ns
    ]
    metrics = ["euclidean", "mahalanobis"]
    reductions = [np.median, np.mean, np.sum, lambda x: np.sum(x**3)]
    cases = combinations([metrics, reductions, uniforms])
    results = [
        (len(d), m, f, HopkinsTest(metric=m, reduction=f).test(d)) for m, f, d in cases
    ]

    false_positives = [r for r in results if r[3].passed]
    df = pd.DataFrame([(r[3].value, r[2].__name__) for r in false_positives])
    df.columns = ["v", "f"]
    sns.kdeplot(df.v, hue=df.f)
    plt.show()

    print(results)

    # assert np.all(np.array([r.passed for _,_,r in results]) == False)

    # all good except for np.sum(x**3), which makes the test pass in
    # some cases while it should not

    any_pm = stats.multivariate_normal(mean=(7.5, 7), cov=1.0 / 35)
    c_f_mixes = [0.1, 0.5, 0.9]
    cov_diag = 0.5
    random_matrix = np.random.rand(3, 3)
    cov_full = [np.dot(rm, rm.T) for rm in random_matrix]
    covs = [cov_diag, cov_full]
    cases = combinations([metrics, reductions, ns, c_f_mixes, covs])

    results = [
        (
            n,
            p,
            m,
            f,
            c,
            HopkinsTest(metric=m, reduction=f).test(
                Synthetic(
                    field=Field(
                        space=UniformSphere(
                            center=polar_to_cartesian((120.5, -27.5, 5)),
                            radius=10,
                        ),
                        pm=any_pm,
                        star_count=int(n * (1 - p)),
                    ),
                    clusters=[
                        Cluster(
                            space=stats.multivariate_normal(
                                mean=polar_to_cartesian([120.7, -28.5, 5]),
                                cov=c,
                            ),
                            pm=any_pm,
                            star_count=int(n * p),
                        ),
                    ],
                    representation_type="cartesian",
                )
                .rvs()[["x", "y", "z"]]
                .to_numpy()
            ),
            Synthetic(
                field=Field(
                    space=UniformSphere(
                        center=polar_to_cartesian((120.5, -27.5, 5)), radius=10
                    ),
                    pm=any_pm,
                    star_count=int(n * (1 - p)),
                ),
                clusters=[
                    Cluster(
                        space=stats.multivariate_normal(
                            mean=polar_to_cartesian([120.7, -28.5, 5]),
                            cov=c,
                        ),
                        pm=any_pm,
                        star_count=int(n * p),
                    ),
                ],
                representation_type="cartesian",
            )
            .rvs()[["x", "y", "z"]]
            .to_numpy(),
        )
        for m, f, n, p, c in cases
    ]

    # 1/3 of results are false negatives !!!
    # whats wrong here
    false_negatives = [r for r in results if r[5].passed is False]
    print(false_negatives)
    df = pd.DataFrame([(r[5].value, r[3].__name__) for r in false_positives])
    df.columns = ["v", "f"]
    sns.kdeplot(df.v, hue=df.f)
    plt.show()
    assert np.all(np.array([r.passed for _, _, r in results]) is True)


def test_hopkins2():
    f_ratios = [0.6, 0.7, 0.8, 0.9]

    space = ["ra", "dec"]
    pm = ["pmra", "pmdec"]
    space_plx = ["ra", "dec", "parallax"]
    pm_plx = ["pmra", "pmdec", "parallax"]
    pm_plx_space = ["ra", "dec", "pmra", "pmdec", "parallax"]
    no_plx = ["ra", "dec", "pmra", "pmdec"]
    xyz = ["x", "y", "z"]

    cols_2d = [space, pm]
    cols_3d = [space_plx, pm_plx]
    cols_5d = [pm_plx_space]
    cols_all = [space, pm, space_plx, pm_plx, pm_plx_space, no_plx]
    cols_all = [pm, space]

    funcs = [case2_sample0c, case2_sample1c, case2_sample2c]

    cs = combinations([funcs, f_ratios, cols_all])
    from rpy2.robjects import r
    from scludam.rutils import rhardload, pyargs2r, rclean

    rhardload(r, "hopkins")
    res = []
    for c in cs:

        sample = c[0](c[1])[c[2]].to_numpy()
        sample = RobustScaler().fit_transform(sample)

        rclean(r, "var")
        _, rparams = pyargs2r(r, data=sample, n=100)
        r_res = np.asarray(r(f"hv = hopkins(data,n)"))
        r_res_pval = np.asarray(r(f"hopkins.pval(hv,n)"))
        # r_res2 = np.asarray(r(f'get_clust_tendency(data,n)$hopkins_stat'))

        test_result = HopkinsTest().test(sample)
        if c[0].__name__ == "case2_sample0c":
            n_clu = 0
            passed = False
        else:
            passed = True
            if c[0].__name__ == "case2_sample1c":
                n_clu = 1
            else:
                n_clu = 2
        err = passed == test_result.passed
        r_passed = r_res_pval.ravel()[0] <= 0.05
        r_err = r_passed == passed
        res.append(
            (
                n_clu,
                c[0].__name__,
                c[1],
                str(c[2]),
                len(c[2]),
                test_result.value,
                test_result.pvalue,
                test_result.passed,
                passed,
                test_result.passed == passed,
                r_res.ravel()[0],
                r_res_pval.ravel()[0],
                err,
                r_err,
            )
        )

    df = pd.DataFrame(res)
    df.columns = [
        "n_clu",
        "func",
        "f_ratio",
        "cols",
        "dims",
        "value",
        "pvalue",
        "passed",
        "expected",
        "passed_equal",
        "r_value",
        "r_pval",
        "err",
        "r_err",
    ]
    df["err"] = df.err.astype(int)
    df["r_err"] = df.err.astype(int)
    print(df)
    # if p value > .05, the likelihood of the null hypothesis is > .05, then there is no enough evidence of the alternative hypothesis
    # H0: sample comes from random distribution

    # NO FUNCIONA BIEN EN 2D PORQUE ES EL UNICO CASO DONDE NO ESTA EL PARALAJE QUE ES LO QUE MEJOR AGRUPA
    # DA FALSO POSITIVO CUANDO SE USA PARALAJE PORQUE NO ESTA UNIFORMEMENTE DISTRIBUIDO
    # DA FALSO NEGATIVO CUANDO NO SE USA PARALAJE PORQUE ESTA MENOS UNIFORMEMENTE DISTRIBUIDO
    # SACAR EL PARALAJE SOLUCIONARIA LOS DOS PROBLEMAS
    # ALTERNATIVAMENTE, SE PUEDE PROBAR CON XYZ EN LUGAR DE RA DEC PLX


def test_hopkins3():
    from sklearn.datasets import load_iris

    data = load_iris().data
    np.random.seed(0)
    test_result = HopkinsTest().test(data)
    print(test_result)


def rhopkins():
    from rpy2.robjects import r
    from scludam.rutils import rhardload, pyargs2r, rclean

    rhardload(r, "hopkins")
    rhardload(r, "pdist")
    r(
        """hopkins2 <- function (X, m=ceiling(nrow(X)/10), d=ncol(X), U=NULL) {
  if (!(is.matrix(X)) & !(is.data.frame(X))) 
    stop("X must be data.frame or matrix")

  if (m > nrow(X)) 
    stop("m must be no larger than num of samples")

  if(missing(U)) {
    # U is a matrix of column-wise uniform values sampled from the space of X
    colmin <- apply(X, 2, min)
    colmax <- apply(X, 2, max)    
    U <- matrix(0, ncol = ncol(X), nrow = m)
    for (i in 1:ncol(X)) {
      U[, i] <- runif(m, min = colmin[i], max = colmax[i])
    }
  } else {
    # The user has provided the uniform values.
  }

  # Random sample of m rows in X (without replacement)
  k <- sample(1:nrow(X), m)
  W <- X[k, , drop=FALSE]   # Need 'drop' in case X is single-column
  
  # distance between each row of W and each row of X
  dwx <- as.matrix(pdist(W,X))
  # Caution: W[i,] is the same point as X[k[i],] and the distance between them is 0,
  # but we do not want to consider that when calculating the minimum distance
  # between W[i,] and X, so change the distance from 0 to Inf
  for(i in 1:m) dwx[i,k[i]] <- Inf
  # distance from each row of W to the NEAREST row of X
  dwx <- apply(dwx, 1, min)
  
  # distance between each row of U and each row of X
  dux <- as.matrix(pdist(U,X)) # rows of dux refer to U, cols refer to X
  # distance from each row of U to the NEAREST row of X
  dux <- apply(dux, 1, min)

  # You would think this would be faster, but it is not for our test cases:
  # stat = 1 / (1 + sum(dwx^d) / sum( dux^d ) )
  
  return( sum(dux^d) / sum( dux^d + dwx^d ) )
}"""
    )

    from sklearn import datasets

    iris = datasets.load_iris()
    min_data = np.min(iris.data, axis=0)
    max_data = np.max(iris.data, axis=0)
    dims = 4
    np.random.seed(0)
    uniform_sample = np.random.uniform(low=min_data, high=max_data, size=(150, dims))
    pyargs2r(r, U=uniform_sample)
    pyargs2r(r, X=iris.data)
    r_res = np.asarray(r(f"hv = hopkins2(X,m=150,U=U)")).ravel()[0]
    r_pval = np.asarray(r("hvpval = hopkins.pval(hv, 150)")).ravel()[0]
    np.random.seed(0)
    test_result = HopkinsTest(n_iters=1, n_samples=150).test(iris.data)
    print(test_result.value)
    print(r_res)
    print(r_pval)


def uniform_sample():
    return BivariateUnifom(locs=(0, 0), scales=(1, 1)).rvs(1000)


def uni_scipy():
    x = stats.uniform(loc=0, scale=1).rvs(size=1000)
    y = stats.uniform(loc=0, scale=1).rvs(size=1000)
    return np.vstack((x, y)).T


def uni_numpy():
    x = np.random.uniform(low=0, high=1, size=1000)
    y = np.random.uniform(low=0, high=1, size=1000)
    return np.vstack((x, y)).T


def uni_r():
    from rpy2.robjects import r
    from scludam.rutils import rhardload, pyargs2r, rclean

    r("x = runif(1000); y = runif(1000)")
    return np.vstack((np.asarray(r("x")).ravel(), np.asarray(r("y")).ravel())).T


def cluster_structure_sample():
    sample = BivariateUnifom(locs=(0, 0), scales=(1, 1)).rvs(500)
    sample2 = stats.multivariate_normal(mean=(0.5, 0.5), cov=1.0 / 200).rvs(500)
    return np.concatenate((sample, sample2))


def harder_cluster():
    sample = BivariateUnifom(locs=(0, 0), scales=(1, 1)).rvs(800)
    sample2 = stats.multivariate_normal(mean=(0.5, 0.5), cov=1.0 / 200).rvs(200)
    return np.concatenate((sample, sample2))


def bimodal_sample():
    sample = BivariateUnifom(locs=(0, 0), scales=(1, 1)).rvs(500)
    sample2 = stats.multivariate_normal(mean=(0.75, 0.75), cov=1.0 / 200).rvs(250)
    sample3 = stats.multivariate_normal(mean=(0.25, 0.25), cov=1.0 / 200).rvs(250)
    return np.concatenate((sample, sample2, sample3))


def test_ripleys():
    us = uniform_sample()
    usr = RipleysKTest().test(data=us)
    cl = cluster_structure_sample()
    clr = RipleysKTest().test(data=cl)
    prin


def test_dip2():
    us = uniform_sample()
    usr = DipTest().test(data=us)
    cl = cluster_structure_sample()
    clr = DipTest().test(data=cl)
    print("coso")


def are_unis_equal():
    from rpy2.robjects import r
    from scludam.rutils import rhardload, pyargs2r, rclean

    rhardload(r, "spatstat")
    loser = []
    for i in range(1000):
        ur = uni_r()
        un = uni_numpy()
        us = uni_scipy()
        pyargs2r(r, ur=ur, un=un, us=us)
        r("W <- owin(c(0,1), c(0,1))")
        r(
            "ur <- as.ppp(as.matrix(ur), W=W); un <- as.ppp(as.matrix(un), W=W); us <- as.ppp(as.matrix(us), W=W)"
        )
        r(
            'ler = Lest(ur, correction="Ripley"); ln = Lest(un, correction="Ripley"); ls = Lest(us, correction="Ripley")'
        )
        radir = np.asarray(r("ler$r"))
        lfr = np.asarray(r("ler$iso"))
        lfn = np.asarray(r("ln$iso"))
        lfs = np.asarray(r("ls$iso"))
        df = pd.DataFrame({"r": radir, "fr": lfr, "fn": lfn, "fs": lfs})
        rval = np.abs(lfr - radir).max()
        nval = np.abs(lfn - radir).max()
        sval = np.abs(lfs - radir).max()
        names = ["r", "n", "s"]
        loser.append(names[np.argmax([rval, nval, sval])])

    print(loser)


def rripley():
    from rpy2.robjects import r
    from scludam.rutils import rhardload, pyargs2r, rclean
    from sklearn.datasets import load_iris

    iris = np.unique(load_iris().data[:, :2], axis=0)
    x_min = np.min(iris[:, 0])
    x_max = np.max(iris[:, 0])
    y_min = np.min(iris[:, 1])
    y_max = np.max(iris[:, 1])

    rhardload(r, "spatstat")
    pyargs2r(r, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    pyargs2r(r, iris=iris)
    r("W <- owin(c(x_min, x_max), c(y_min, y_max))")
    r("iris <- as.ppp(iris, W=W)")
    r('le = Lest(iris, correction="Ripley")')
    radii = np.asarray(r("le$r"))
    lf = np.asarray(r("le$iso"))
    diff = np.abs(lf - radii)
    value = diff.max()

    print(RipleysKTest().test(iris))
    n = 1000

    mine = []
    rrr = []
    mine2 = []
    for i in range(100):
        u = uniform_sample()
        pyargs2r(r, u=u)
        r("W <- owin(c(0,1), c(0,1))")
        r("u <- as.ppp(as.matrix(u), W=W)")
        r('le = Lest(u, correction="Ripley")')
        radii = np.asarray(r("le$r"))
        theo = np.asarray(r("le$theo"))
        lf = np.asarray(r("le$iso"))
        threshold = 1.45 * np.sqrt(1) / n
        value = np.max(np.abs(lf - radii))
        passed = value >= threshold
        correct = not passed
        rrr.append(correct)
        correct_mine = not RipleysKTest().test(u).passed
        correct_mine2 = not RipleysKTest(mode="ks").test(u).passed
        if not correct:
            print("stop")
        mine.append(correct_mine)
        mine2.append(correct_mine2)

    rrr1 = []
    mine1 = []
    mine11 = []
    for i in range(100):
        u = cluster_structure_sample()
        pyargs2r(r, u=u)
        r("W <- owin(c(0,1), c(0,1))")
        r("u <- as.ppp(as.matrix(u), W=W)")
        r('le = Lest(u, correction="Ripley")')
        radii = np.asarray(r("le$r"))
        lf = np.asarray(r("le$iso"))
        threshold = 1.45 * np.sqrt(1) / n
        value = np.max(np.abs(lf - radii))
        passed = value >= threshold
        correct = passed
        rrr1.append(correct)
        correct_mine1 = RipleysKTest().test(u).passed
        correct_mine3 = RipleysKTest(mode="ks").test(u).passed
        mine1.append(correct_mine)
        mine11.append(correct_mine3)

    mine1h = []
    rrrh = []
    mine2h = []
    for i in range(100):
        u = harder_cluster()
        pyargs2r(r, u=u)
        r("W <- owin(c(0,1), c(0,1))")
        r("u <- as.ppp(as.matrix(u), W=W)")
        r('le = Lest(u, correction="Ripley")')
        radii = np.asarray(r("le$r"))
        lf = np.asarray(r("le$iso"))
        threshold = 1.45 * np.sqrt(1) / n
        value = np.max(np.abs(lf - radii))
        passed = value >= threshold
        correct = passed
        rrrh.append(correct)
        correct_mine = RipleysKTest().test(u).passed
        correct_mine2 = RipleysKTest(mode="ks").test(u).passed
        if not correct_mine2:
            print("stop")
        mine1h.append(correct_mine)
        mine2h.append(correct_mine2)

    mineks = np.concatenate([np.array(mine2), np.array(mine11)])
    mineer = np.concatenate([np.array(mine), np.array(mine1)])
    rrrr = np.concatenate([np.array(rrr), np.array(rrr1)])
    ks = len(mineks[mineks])
    er = len(mineer[mineer])
    rr = len(rrrr[rrrr])
    u = uniform_sample()
    pyargs2r(r, u=u)
    r("W <- owin(c(0,1), c(0,1))")
    r("u <- as.ppp(as.matrix(u), W=W)")
    r("plot(u)")
    r("dev.off()")
    r('le = Lest(u, correction="Ripley")')
    radii = np.asarray(r("le$r"))
    lf = np.asarray(r("le$iso"))
    diff = np.abs(lf - radii)
    value = diff.max()
    threshold = 1.68 * np.sqrt(1) / n
    passed = value >= threshold
    correct = not passed
    print(f"uniform {correct}")

    lfa = RipleysKEstimator(area=1, x_max=1, x_min=0, y_min=0, y_max=1).Lfunction(
        u, radii, mode="ripley"
    )
    diffa = np.abs(lfa - radii)
    valuea = diffa.max()
    thresholda = 1.68 * np.sqrt(1) / n
    passeda = valuea >= thresholda
    correcta = not passeda

    clu = cluster_structure_sample()
    pyargs2r(r, clu=clu)
    r("clu <- as.ppp(as.matrix(clu), W=W)")
    r("plot(clu)")
    r("dev.off()")
    r('le = Lest(clu, correction="Ripley")')
    radii = np.asarray(r("le$r"))
    lf = np.asarray(r("le$iso"))
    diff = np.abs(lf - radii)
    value = diff.max()
    threshold = 1.68 * np.sqrt(1) / n
    passed = value >= threshold
    correct = passed
    print(f"clu {correct}")

    print(r_res)


def rdip():
    from rpy2.robjects import r
    from scludam.rutils import rhardload, pyargs2r, rclean
    from sklearn.datasets import load_iris

    rhardload(r, "diptest")
    us = uniform_sample()
    cl = cluster_structure_sample()
    bi = bimodal_sample()

    sus = resample(us, n_samples=500, replace=False)
    clss = resample(cl, n_samples=500, replace=False)
    bis = resample(bi, n_samples=500, replace=False)

    distu = np.ravel(np.tril(pairwise_distances(sus)))
    distu = np.msort(distu[distu > 0])

    distc = np.ravel(np.tril(pairwise_distances(clss)))
    distc = np.msort(distc[distc > 0])

    distb = np.ravel(np.tril(pairwise_distances(bis)))
    distb = np.msort(distb[distb > 0])

    pyargs2r(r, u=distu, clu=distc, bi=distb)
    r("du = dip.test(u)")
    r("dc = dip.test(clu)")
    r("db = dip.test(bi)")
    upval = np.asarray(r("du$p.value")).ravel()[0]
    clpval = np.asarray(r("dc$p.value")).ravel()[0]
    bipval = np.asarray(r("db$p.value")).ravel()[0]
    ust = np.asarray(r("du$statistic")).ravel()[0]
    clt = np.asarray(r("dc$statistic")).ravel()[0]
    bit = np.asarray(r("db$statistic")).ravel()[0]
    corru = upval > 0.05
    corrc = clpval > 0.05
    corrb = bipval <= 0.05
    # sns.histplot(dist).set(title=str(pval))
    # plt.show()
    # print(pval)

    np.random.seed(0)
    usr = DipTest().test(data=us)
    clr = DipTest().test(data=cl)


# test_dip2()
# rripley()
# rdip()
# are_unis_equal()
""" pops = [uniform_sample, cluster_structure_sample, bimodal_sample]
tests = [HopkinsTest(), RipleysKTest(), DipDistTest()]

tres = []
for t in tests:
    for i,p in enumerate(pops):
        np.random.seed(0)
        tres.append({
            'test': t.__class__.__name__,
            'pop': i+1,
            'passed': t.test(p()).passed,
        })

print(tres) """

""" passeds = []
for i in range(100):
    passeds.append(RipleysKTest(pvalue_threshold=.01).test(uniform_sample()).passed)
len(passeds[passeds])

print(passeds) """
# uht = HopkinsTest(metric="mahalanobis", n_iters=100).test(data=uniform_sample).passed


def test_ripleysk_empirical_rule():
    radii = np.array([1, 2, 3])
    lf = np.array([1.01, 2.01, 3.02])
    area = 1.5
    n = 100
    value1, passed1 = RipleysKTest(factor=1.65).empirical_csr_rule(
        radii=radii, l_function=lf, area=area, n=n
    )
    assert passed1
    assert np.isclose(value1, 0.02)
    value2, passed2 = RipleysKTest(pvalue_threshold=0.01).empirical_csr_rule(
        radii=radii, l_function=lf, area=area, n=n
    )
    assert not passed2
    assert np.isclose(value2, 0.02)


def uniform_sample():
    return BivariateUnifom(locs=(0, 0), scales=(1, 1)).rvs(1000)


def one_cluster_sample():
    sample = BivariateUnifom(locs=(0, 0), scales=(1, 1)).rvs(500)
    sample2 = multivariate_normal(mean=(0.5, 0.5), cov=1.0 / 200).rvs(500)
    return np.concatenate((sample, sample2))


def two_clusters_sample():
    sample = BivariateUnifom(locs=(0, 0), scales=(1, 1)).rvs(500)
    sample2 = multivariate_normal(mean=(0.75, 0.75), cov=1.0 / 200).rvs(250)
    sample3 = multivariate_normal(mean=(0.25, 0.25), cov=1.0 / 200).rvs(250)
    return np.concatenate((sample, sample2, sample3))


def test_dip_uniform():
    assert not DipDistTest().test(uniform_sample()).passed


def test_dip_one_cluster():
    assert not DipDistTest().test(one_cluster_sample()).passed


def test_dip_two_clusters():
    assert DipDistTest().test(data=two_clusters_sample()).passed


def plot(tr):
    palette = sns.color_palette("flare", 20)
    ran_color = palette[np.random.choice(len(palette))]
    return sns.lineplot(np.arange(0, len(tr.dist)), tr.dist, color=ran_color)


""" for i in range(10):
    np.random.seed(i)
    u = uniform_sample()
    np.random.seed(i)
    c = one_cluster_sample()
    np.random.seed(i)
    b = two_clusters_sample()
    ur = DipDistTest(n_samples=200).test(u)
    cr = DipDistTest(n_samples=200).test(c)
    br = DipDistTest(n_samples=200).test(b)

    if not ur.passed and cr.passed and br.passed:
        print('Correct:', i)
    else:
        print('Incorrect:', i)
        print(ur.passed, cr.passed, br.passed)
        print('coso')

print('coso') """
