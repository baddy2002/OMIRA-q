import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel

class QuantumKernelClassifier:
    """
    Calcola la matrice di Gram quantistica (Fidelity Kernel):
    K_ij = |<psi(m_i) | psi(m_j)>|^2
    e addestra un classificatore a margine (SVM / Ridge).
    """
    def __init__(self, c_param=1.0):
        self.clf = SVC(kernel="precomputed", C=c_param)
        self.train_states = []

    def fit(self, train_states, y_train):
        self.train_states = train_states
        K_train = self._compute_gram_matrix(train_states, train_states)
        self.clf.fit(K_train, y_train)
        return self

    def predict(self, test_states):
        K_test = self._compute_gram_matrix(test_states, self.train_states)
        return self.clf.predict(K_test)

    @staticmethod
    def _compute_gram_matrix(states_A, states_B):
        n_a, n_b = len(states_A), len(states_B)
        K = np.zeros((n_a, n_b), dtype=np.float32)
        for i in range(n_a):
            for j in range(n_b):
                # Fidelity tra due statevector QuTiP
                psi_a = states_A[i]
                psi_b = states_B[j]
                K[i, j] = np.abs(np.vdot(psi_a, psi_b)) ** 2
        return K