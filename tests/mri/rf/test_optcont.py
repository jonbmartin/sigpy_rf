import time
import unittest
import numpy as np
import numpy.testing as npt
import jax.numpy as jnp
from jax import jit
from sigpy.mri.rf import optcont
import sigpy.mri.rf as rf

if __name__ == '__main__':
    unittest.main()


class TestOptcont(unittest.TestCase):

    def test_rf_autodiff(self):
        # test parameters (can be changed)
        dt = 1e-6
        b1 = np.arange(0, 3, 0.1)  # gauss, b1 range to sim over
        nb1 = np.size(b1)
        pbc = 1.5  # b1 (Gauss)
        pbw = 0.4  # b1 (Gauss)

        # generate INPUT rf pulse

        rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=2, ndes=256, ptype='ex', flip=np.pi / 2,
                                           pbw=pbw,
                                           pbc=[pbc], d1e=0.01, d2e=0.01,
                                           rampfilt=True, bs_offset=5000)
        full_pulse = (rfp_bs + rfp_ss) * 2 * np.pi * 4258 * dt  # scaled

        # simulate with target function to generate magnetization profile
        rfp_abs = abs(full_pulse)
        rfp_angle = np.angle(full_pulse)
        nt = np.size(rfp_abs)
        rf_op = np.append(rfp_abs, rfp_angle)

        w = np.ones(nb1)  # weight

        Mxd = np.zeros(nb1)
        Myd = np.zeros(nb1)
        Mzd = np.zeros(nb1)

        for ii in range(nb1):
            Mxd[ii], Myd[ii], Mzd[ii] = rf.sim.arb_phase_b1sel_np(rf_op, b1[ii], 0, 0, 1.0, nt)

        # optimize the pulse with its original target profile as sanity check
        Mxd = np.array(Mxd)
        Myd = np.array(Myd)
        Mzd = np.array(Mzd)

        # huge step size in order to really push if the loss goes to 0 for the same profile
        rf_test_1 = optcont.rf_autodiff(rf_op, b1, Mxd, Myd, Mzd, w, niters=20, step=0.1, mx0=0,
                                       my0=0, mz0=1.0)

        for ii in range(nb1):
            Mxd[ii], Myd[ii], Mzd[ii] = rf.sim.arb_phase_b1sel_np(rf_test_1, b1[ii], 0, 0, 1.0, nt)

        # compare results
        npt.assert_almost_equal(rf_op, rf_test_1, decimal=2)

    # def test_optcont1d(self):
    #     print('Test not fully implemented')
    #
    #     try:
    #         gamgdt, pulse = optcont.optcont1d(4, 256, 2, 8)
    #     finally:
    #         dt = 4e-6
    #         gambar = 4257  # gamma/2/pi, Hz/g
    #         [a, b] = rf.optcont.blochsim(pulse, x / (gambar * dt * gmag), gamgdt)
    #         Mxy = 2 * np.conj(a) * b
    #
    #         pyplot.figure()
    #         pyplot.figure(np.abs(Mxy))
    #         pyplot.show()
    #
    #         # TODO: compare with target Mxy, take integration
    #
    #         alpha = rf.b2a(db)
    #
    # def test_blochsim(self):
    #     print('Test not implemented')
    #     # TODO: insert testing
    #
    # def test_deriv(self):
    #     print('Test not implemented')
    #     # TODO: insert testing
