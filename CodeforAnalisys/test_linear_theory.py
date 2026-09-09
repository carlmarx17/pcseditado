"""Independent physical and numerical checks of the parallel solver."""

import unittest
import numpy as np
from scipy.special import gamma, roots_genlaguerre

from linear_theory import ParallelDispersion, Z, Z_kappa


def make_dispersion(kappa=None, **overrides):
    params = dict(beta_par_i=3.0, A_i=2.0, beta_par_e=1.0, A_e=1.0,
                  mass_ratio=200.0, c_over_va=176.7767, kappa=kappa)
    return ParallelDispersion(**(params | overrides))


class LinearTheoryTests(unittest.TestCase):
    def test_kappa_response_matches_loader_gaussian_mixture(self):
        # The PSC loader mixes Gaussians with Y~Gamma(kappa-1/2,1).
        # Susceptibility is linear in f, so averaging Maxwellian responses
        # is independent of the implemented Z_kappa velocity quadrature.
        kappa = 3.0
        model = make_dispersion(kappa)
        maxwell = make_dispersion()
        nodes, weights = roots_genlaguerre(256, kappa - 1.5)
        weights /= gamma(kappa - 0.5)
        for cyclotron, base_speed, anisotropy in [(1.0, np.sqrt(3.0), 2.0), (-200.0, np.sqrt(200.0), 1.0)]:
            speeds = base_speed * np.sqrt((kappa - 1.5) / nodes)
            terms = [maxwell._species_term(0.4 + 0.3j, 0.5, 1.0, anisotropy, speed, cyclotron)
                     for speed in speeds]
            mixture = np.dot(weights, terms)
            direct = model._species_term(0.4 + 0.3j, 0.5, 1.0, anisotropy,
                                         base_speed * np.sqrt(0.5), cyclotron)
            self.assertAlmostEqual(abs(direct - mixture), 0.0, delta=2e-6)

    def test_finite_kappa_is_not_replaced_by_maxwellian_at_50(self):
        zeta = 0.7 + 0.4j
        errors = [abs(Z_kappa(zeta, k) - Z(zeta)) for k in [40, 80, 160]]
        self.assertTrue(errors[0] > errors[1] > errors[2] > 0)
        self.assertLess(max(e * k for e, k in zip(errors, [40, 80, 160])) /
                        min(e * k for e, k in zip(errors, [40, 80, 160])), 1.03)

    def test_cold_isotropic_response_and_circular_sign(self):
        model = make_dispersion(beta_par_i=1e-7, beta_par_e=1e-7, A_i=1.0)
        omega, k = 0.3 + 0.1j, 0.2
        for channel, sign in [("plus", 1), ("minus", -1)]:
            expected = k*k - omega**2/model.c_over_va**2 + omega/(omega-sign) + 200*omega/(omega+sign*200)
            self.assertAlmostEqual(abs(model(omega, k, channel) - expected), 0.0, delta=1e-7)

    def test_unconverged_iterate_is_not_a_root(self):
        model = make_dispersion()
        result = model.solve(0.4, max_iter=1)
        self.assertFalse(np.isfinite(result))
        self.assertFalse(model.last_solve["converged"])

    def test_accepted_root_has_small_residual(self):
        model = make_dispersion(A_i=0.1, beta_par_i=10.0)
        root = model.solve(0.1)
        self.assertTrue(np.isfinite(root))
        self.assertLess(abs(model(root, 0.1)) / 0.1**2, 1e-8)

    def test_invalid_parameters_and_damped_kappa_are_explicit(self):
        for kappa in [1.5, -1.0, np.nan]:
            with self.assertRaises(ValueError):
                make_dispersion(kappa)
        self.assertFalse(np.isfinite(Z_kappa(0.3 - 0.1j, 3.0)))
        with self.assertRaises(ValueError):
            make_dispersion().solve(0.0)


if __name__ == "__main__":
    unittest.main()
