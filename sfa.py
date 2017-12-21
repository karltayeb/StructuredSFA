import numpy as np
from scipy.stats import multivariate_normal
# from sklearn.datasets import make_spd_matrix
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
        self.sigma2 = np.ones((n_features, k_components))

        self.scale_F_sigma2()

        # data
        self.data = np.asarray(data)

        # moments of loading matrix
        self.Lambda = np.zeros((n_features, k_components))
        self.Lambda2 = np.zeros((n_features, k_components, k_components))

        self.psi = np.asarray(data.var(axis=1))

        # covariance matrix for samples
        self.Gamma_inverse = \
            np.zeros((self.k_components, self.m_samples, self.m_samples))

        # hyper parameters
        self.alpha = 1
        self.beta = 20.0 / m_samples
        self.penalty = 0

        self.lls = [-np.inf]

    def ecme(self, iters, structured=False, scale=True, verbose=True):
        """
        iters = number of iterations
        structured = whether or not to consider relationships between sampled
        encoded in inverse covariance matrices Gamma_inverse
        """
        if verbose is False:
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        for i in range(iters):
            print('updating lambda')
            self.update_lambda()

            print('updating psi')
            self.sfa_update_psi()

            print('updating sigma2')
            self.sfa_update_sigma2()

            print('updating F')
            if structured:
                self.update_structured_F()
            else:
                self.update_F()

            if scale:
                print('scaling F, sigma2')
                self.scale_F_sigma2()

            if i % 5 == 0:
                # print('computing log likelihood')
                self.sfa_log_likelihood()
                change = self.lls[-1] - self.lls[-2]
                print(i, 'expected log likelihood has improved by:', change)

                if (change >= 0 and change <= 1e-8):
                    print('converged')
                    break

        if verbose is False:
            sys.stdout = original_stdout

    def update_lambda(self):
        """
        """
        Lambda = np.zeros((self.n_features, self.k_components))
        Lambda2 = \
            np.zeros((self.n_features, self.k_components, self.k_components))

        for i in range(self.n_features):
            sigma2 = self.sigma2[i]
            psi = self.psi[i] * np.eye(self.m_samples)
            psi_inv = (1 / self.psi[i]) * np.eye(self.m_samples)

            if not np.isclose(sigma2.sum(), 0):
                sigma2 = np.diag(sigma2)

                """
                omega = np.linalg.multi_dot([
                    sigma2,
                    self.F,
                    woodbury(psi, self.F.T, sigma2, self.F)
                ])
                """
                omega = np.linalg.multi_dot([
                    sigma2,
                    self.F,
                    np.linalg.inv(psi_inv + np.dot(np.dot(self.F.T, sigma2), self.F))
                ])

                Lambda[i] = np.dot(omega, self.data[i])
                Lambda2[i] = sigma2 \
                    - np.linalg.multi_dot([omega, self.F.T, sigma2]) \
                    + np.outer(Lambda[i], Lambda[i])

        self.Lambda = Lambda
        self.Lambda2 = Lambda2

    def update_F(self):
        """
        """
        F = np.zeros(self.F.shape)

        S = np.zeros((self.k_components, self.k_components))
        for i in range(self.n_features):
            S = S + (self.psi[i] * self.Lambda2[i])
        S = np.linalg.inv(S)

        for j in range(self.m_samples):
            T = np.zeros(self.k_components)
            for i in range(self.n_features):
                T = T + (self.psi[i] * self.data[i, j] * self.Lambda[i])

            F[:, j] = np.dot(S, T)

        self.F = F

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
            const = self.m_samples + (2 * self.m_samples * (self.alpha - 1))
            newPsi[i] = const / \
                (YtY - (2 * YgFtLg) + FtLLtFt + (2 / self.beta))
        self.psi = newPsi

    def sfa_update_psi(self):
        YtY = 0
        YgFt = np.zeros(self.k_components)
        YgFtLg = 0
        Yg = np.zeros(self.m_samples)
        FtLLtF = 0

        FFt = np.dot(self.F, self.F.T)

        for i in range(self.n_features):
            Yg = self.data[i]
            YtY = np.inner(Yg, Yg)
            YgFt = np.dot(Yg, self.F.T)
            YgFtLg = np.inner(YgFt, self.Lambda[i])

            FFtLLt = np.dot(FFt, self.Lambda2[i])
            FtLLtF = np.sum(np.diagonal(FFtLLt).sum())
            self.psi[i] = 1 / \
                (((YtY - (2.0 * YgFtLg) + FtLLtF) + (2.0 / self.beta)) /
                 self.m_samples)

    def update_structured_F(self):
        #self.scale_F_sigma2()
        #self.update_lambda()

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
                    X[idx, idx_in_factor] + self.Gamma_inverse[k, j]
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
        
    def sfa_update_sigma2(self):
        for i in range(self.n_features):
            self.sigma2[i] = self._update_sigma2_row(i)

    def _update_sigma2_row(self, i):
        psi_inv = (1 / self.psi[i]) * np.eye(self.m_samples)
        sigma2g = self.sigma2[i]

        for k in range(self.k_components):
            Binv = np.linalg.inv(
                psi_inv + np.dot(self.F.T * sigma2g, self.F)
            )
            omega = np.dot(self.F, Binv)

            X = np.dot(omega, self.F.T)
            Qm = np.dot(omega, self.data[i])
            sigma2gk = sigma2g[k]
            sm = X[k, k]
            qm = Qm[k]

            if np.isclose(sigma2gk, 0):
                q2 = qm * qm
            else:
                q2 = (1.0 / sigma2gk) * qm / ((1.0 / sigma2gk) - sm)
                q2 = q2 * q2
                sm = (1.0 / sigma2gk) * sm / ((1.0 / sigma2gk) - sm)

            new_sigma2gk = max(0, (q2 - sm) / (sm * sm))
            sigma2g[k] = new_sigma2gk

        return sigma2g

    def sfa_update_sigma22(self):
        for i in range(self.n_features):
            psi_inv = (1 / self.psi[i]) * np.eye(self.m_samples)
            sigma2g = np.diag(self.sigma2[i])

            if np.isclose(sigma2g.sum(), 0):
                Binv = self.psi[i] * np.eye(self.m_samples)
            else:
                Binv = np.linalg.inv(
                    psi_inv + np.linalg.multi_dot([self.F.T, sigma2g, self.F])
                )

            for k in range(self.k_components):
                omega = np.dot(self.F, Binv)
                X = np.dot(omega, self.F.T)
                Qm = np.dot(omega, self.data[i])
                sigma2gk = sigma2g[k, k]
                sm = X[k, k]
                qm = Qm[k]

                if np.isclose(sigma2gk, 0):
                    q2 = qm * qm
                else:
                    q2 = (1.0 / sigma2gk) * qm / ((1.0 / sigma2gk) - sm)
                    q2 = q2 * q2
                    sm = (1.0 / sigma2gk) * sm / ((1.0 / sigma2gk) - sm)

                new_sigma2gk = max(0, (q2 - sm) / (sm * sm))
                sigma2g[k, k] = new_sigma2gk

                # update omega for new sigma
                x = self.F[k] * np.sqrt(sigma2gk)
                Binv = sherman_morrison(Binv, -x, x)

                x = self.F[k] * np.sqrt(new_sigma2gk)
                Binv = sherman_morrison(Binv, x, x)

            self.sigma2[i] = np.diagonal(sigma2g)

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
        return marginal log likelihood
        up to a constant
        """
        ll = 0

        for i in range(self.n_features):

            sigma2 = self.sigma2[i]
            psi_inv = (1 / self.psi[i]) * np.eye(self.m_samples)

            if np.all(np.isclose(sigma2, 0)):
                cov = psi_inv

            else:
                cov = np.linalg.multi_dot([
                    self.F.T, np.diag(sigma2), self.F
                ]) + (psi_inv)

            psi_inv = (1 / self.psi[i]) * np.eye(self.m_samples)

            ll += multivariate_normal.logpdf(
                x=self.data[i], mean=np.zeros(self.m_samples), cov=cov)

        self.lls.append(ll)

        return ll

    def expected_log_likelihood(self):
        """
        return marginal log likelihood
        up to a constant
        """
        ll = 0

        for i in range(self.n_features):
            temp = 0
            for j in range(self.m_samples):
                dat = self.data[i, j]
                temp += dat * dat
                temp += -2 * dat * np.inner(self.Lambda[i], self.F[:, j])
                temp += np.linalg.multi_dot(
                    [self.F[:, j], self.Lambda2[i], self.F[:, j]])

            ll += -0.5 * temp * self.psi[i]

        ll += self.m_samples * np.log(self.psi).sum()
        ll += -1 * self.psi.sum() / self.beta
        return ll

    def sfa_log_likelihood(self):
        """
        expected complete log likelihood
        """
        mean = np.dot(self.Lambda, self.F)
        ll = 0
        psiterm = 0
        psiprior = 0
        for i in range(self.n_features):
            Y = self.data[i]
            mean = np.dot(self.Lambda[i], self.F)
            residual = Y - mean
            ll += -0.5 * np.inner(residual, residual) * self.psi[i]
            psiterm += self.m_samples * np.log(self.psi[i])
            psiprior += self.psi[i] / self.beta

        log_likelihood = (((-1 * (self.m_samples * self.n_features) / 2.0)
                 * np.log(2*np.pi)) + (0.5 * psiterm) + ll + psiprior)

        self.lls.append(log_likelihood)

        return log_likelihood

    def sfa_log_likelihood2(self):
        """
        expected complete log likelihood
        """
        mean = np.dot(self.Lambda, self.F)
        ll = 0
        psiterm = 0
        psiprior = 0
        for i in range(self.n_features):
            Y = self.data[i]
            mean = np.dot(self.Lambda[i], self.F)
            residual = Y - mean
            ll += -0.5 * (np.inner(residual, residual) + self.sigma2[i].sum()) * self.psi[i]
            psiterm += self.m_samples * np.log(self.psi[i])
            psiprior += self.psi[i] / self.beta

        return (((-1 * (self.m_samples * self.n_features) / 2.0)
                 * np.log(2*np.pi)) + (0.5 * psiterm) + ll + psiprior)


    def sfa_marginal_log_likelihood(self):
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
