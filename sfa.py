import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
import sys
import os


class SparseFactorAnalyzer:
    def __init__(self, k_components=0, data=None):
        # dimensions
        n_features, m_samples = data.shape[:2]

        self.k_components = k_components
        self.n_features = n_features
        self.m_samples = m_samples

        # initialize parameters
        self.Lambda = np.ones((self.n_features, k_components))  # loadings
        self.F = np.random.random((k_components, m_samples))  # factors
        # ard parameters
        self.sigma2 = np.random.random((n_features, k_components))

        self.scale_F_sigma2()

        self.psi = np.asarray(data.var(axis=1))

        # data
        self.data = np.asarray(data)

        # moments of loading matrix
        self.Lambda = np.zeros((n_features, k_components))
        self.Lambda2 = np.zeros((n_features, k_components, k_components))

        # covariance matrix for samples
        self.Gamma = \
            np.zeros((self.k_components, self.m_samples, self.m_samples))
        self.Gamma = np.array([make_spd_matrix(self.m_samples)
                              for k in range(k_components)])

        # hyper parameters
        self.alpha = 1
        self.beta = 20.0 / m_samples
        self.penalty = 0

        self.lls = [-np.inf]

    def ecme(self, iters, structured=False, scale=True, verbose=False):
        """
        iters = number of iterations
        structured = whether or not to consider relationships between sampled
        encoded in inverse covariance matrices Gamma
        """
        if verbose is False:
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        for i in range(iters):
            print('iter:', i)

            #print('updating lambda')
            self.update_lambda()

            #print('updating psi')
            self.update_psi()

            #print('updating sigma2')
            self.update_lambda()
            self.update_sigma2()

            self.update_lambda()
            if structured:
                #print('updating structured F')
                self.update_structured_F()
            else:
                #print('updating F')
                self.update_F()

            if scale:
                #print('scaling sigma2 and F')
                self.scale_F_sigma2()

            if i % 1 == 0:
                #print('computing log likelihood')
                self.lls.append(self.log_likelihood())
                change = self.lls[-1] - self.lls[-2]
                print('marginal log likelihood has improved by:', change)

                if (change >= 0 and change <= 1e-8):
                    print('converged')
                    break

        if verbose is False:
            sys.stdout = original_stdout

    def update_lambda(self):

        omega = np.zeros((self.n_features, self.k_components, self.m_samples))
        Lambda = np.zeros((self.n_features, self.k_components))
        Lambda2 = np.zeros((self.n_features, self.k_components,
                            self.k_components))

        for i in range(self.n_features):
            sigma2 = np.diag(self.sigma2[i])
            psi = self.psi[i] * np.eye(self.m_samples)

            omega[i] = np.linalg.multi_dot([
                sigma2,
                self.F,
                woodbury(psi, self.F.T, sigma2, self.F)
            ])

            Lambda[i] = np.dot(omega[i], self.data[i])
            Lambda2[i] = sigma2 \
                + np.outer(Lambda[i], Lambda[i]) \
                - np.linalg.multi_dot([omega[i], self.F.T, sigma2])

            self.Lambda = Lambda
            self.Lambda2 = Lambda2

    def update_F(self):
        Z = np.zeros((self.k_components, self.k_components))
        newF = np.zeros(self.F.shape)
        for i in range(self.n_features):
            Z += self.psi[i] * self.Lambda2[i]
            # W += self.psi[i] * np.outer(self.Lambda[i], self.data[i])

        Z = np.linalg.pinv(Z)

        for j in range(self.m_samples):
            W = np.zeros(self.k_components)
            for i in range(self.n_features):
                W += self.psi[i] * self.data[i, j] * self.Lambda[i]

            newF[:, j] = np.dot(Z, W)
        self.F = newF

    def update_structured_F(self):
        dim = self.k_components * self.m_samples
        X = np.zeros((dim, dim))
        C = np.zeros(dim)

        Z = np.zeros((self.k_components, self.k_components))
        for i in range(self.n_features):
            Z += self.psi[i] * self.Lambda2[i]

        idx = 0
        for k in range(self.k_components):
            for j in range(self.m_samples):
                idx_in_factor = \
                    np.arange(self.m_samples) + (k * self.m_samples)
                idx_in_sample = \
                    j + (np.arange(self.k_components) * self.m_samples)

                C[idx] = \
                    np.inner(self.psi * self.data[:, j], self.Lambda[:, k])

                X[idx, idx_in_factor] = \
                    X[idx, idx_in_factor] + self.Gamma[k, j]
                X[idx, idx_in_sample] = \
                    X[idx, idx_in_sample] + Z[k]
                idx += 1

        newF = np.linalg.solve(X, C).reshape(self.k_components, -1)
        self.F = newF
        return X, C, Z

    def update_psi(self):
        newPsi = np.zeros(self.psi.shape)
        for i in range(self.n_features):
            YtY = np.inner(self.data[i], self.data[i])
            YgFtLg = np.linalg.multi_dot([
                    self.data[i],
                    self.F.T,
                    self.Lambda[i]
                ])
            FtLLtFt = np.einsum(
                    'ij,ji->i',
                    np.dot(self.F.T, self.Lambda2[i].T), self.F
                ).sum()
            const = (self.m_samples + 2 * self.m_samples * (self.alpha - 1))
            newPsi[i] = const / (YtY - (2 * YgFtLg) + FtLLtFt + 2 / self.beta)
        self.psi = newPsi
    """
    def update_sigma2(self):
        newSigma2 = self.sigma2

        for i in range(self.n_features):
            psi_inv = (1 / self.psi[i]) * np.eye(self.m_samples)
            for k in range(self.k_components):
                sigma2_delk = newSigma2[i]
                sigma2_delk[k] = 0
                sigma2_delk = np.diag(sigma2_delk)

                B = np.linalg.multi_dot([self.F.T, sigma2_delk, self.F]) + \
                    psi_inv
                Binv = np.linalg.pinv(B)
                q = np.linalg.multi_dot([self.F[k].T, Binv, self.data[i]])
                s = np.linalg.multi_dot([self.F[k].T, Binv, self.F[k]])

                sig = ((q ** 2) - s) / (s ** 2)

                if sig <= 0:
                    newSigma2[i, k] = 0
                else:
                    newSigma2[i, k] = sig

            if np.all(np.isclose(newSigma2[i, :], 0)):
                newSigma2[i, :] += 1e-5

        self.sigma2 = newSigma2
    """

    def update_sigma2(self):
        newSigma2 = self.sigma2
        for i in range(self.n_features):
            sigma2 = np.diag(newSigma2[i])
            psi_inv = (1 / self.psi[i]) * np.eye(self.m_samples)
            B = np.linalg.multi_dot([self.F.T, sigma2, self.F]) + \
                psi_inv
            Binv = np.linalg.pinv(B)

            for k in range(self.k_components):
                sigma2k = newSigma2[i, k]
                if sigma2k == 0:
                    Binv_k = Binv
                else:
                    x = self.F[k] * np.sqrt(sigma2k)
                    Binv_k = sherman_morrison(Binv, -1*x, x)

                q = np.linalg.multi_dot([self.F[k].T, Binv_k, self.data[i]])
                s = np.linalg.multi_dot([self.F[k].T, Binv_k, self.F[k]])

                sig = ((q ** 2) - s) / (s ** 2)

                # threshold new sig estimate, update sigma and Binv
                if sig <= 0:
                    newSigma2[i, k] = 0
                    Binv = Binv_k
                else:
                    newSigma2[i, k] = sig
                    x = self.F[k] * np.sqrt(sig)
                    Binv = sherman_morrison(Binv_k, x, x)

            if np.all(np.isclose(newSigma2[i, :], 0)):
                newSigma2[i, :] += 1e-10

        self.sigma2 = newSigma2
    

    def scale_F_sigma2(self):
        """
        divide rows of F by their standard deve derivative of
        multiply corresponding columns of Sigma by variance
        """
        stdv = self.F.std(axis=1)
        self.F = self.F / stdv[:, np.newaxis]
        self.sigma2 = self.sigma2 * (stdv ** 2)[np.newaxis, :]
        return stdv

    def log_likelihood(self):
        """
        up to a constant
        """
        ll = 0

        for i in range(self.n_features):
            sigma2 = np.diag(self.sigma2[i])
            psi_inv = (1 / self.psi[i]) * np.eye(self.m_samples)

            cov = np.linalg.multi_dot([
                self.F.T, sigma2, self.F
            ]) + (psi_inv)

            ll += multivariate_normal.logpdf(
                x=self.data[i], mean=np.zeros(self.m_samples), cov=cov)

        return ll

    def scale_Y():
        pass

    def update_psi_sfa(void):
        pass

    def update_mu_rows_sfa(void):
        pass

    def update_mu_columns_sfa(void):
        pass

    def update_mu_additive_sfa(void):
        pass

    def objective_function(void):
        pass

    def write_matrices(path):
        pass

    def single_iteration(path):
        pass

    def marginal_log_likelihood():
        pass

    def test_matrix_inverse():
        pass

    def residual_variance():
        pass

    def residual_sum_squares():
        pass


def woodbury(Ainv, U, C, V):
    """
    (A + UCV)^-1 = Ainv - Ainv U(Cinv + V Ainv U)^-1 V A inv
    """
    Cinv = np.linalg.pinv(C)

    temp = Cinv + np.linalg.multi_dot([
        V, Ainv, U
    ])

    result = Ainv - np.linalg.multi_dot([
        Ainv,
        U,
        np.linalg.pinv(temp),
        V,
        Ainv
    ])

    return result


def sherman_morrison(Ainv, u, v):
    """
    (A + uv^T)^-1 = Ainv - Ainv u v^T Ainv/ (1 + v^T Ainv u)
    """
    u = u.reshape(-1, 1)
    v = v.reshape(-1, 1)

    term = np.linalg.multi_dot([
            Ainv,
            np.outer(u, v),
            Ainv
        ])

    const = 1 + np.linalg.multi_dot([v.T, Ainv, u])
    inverse = Ainv - (term / const)

    return inverse
