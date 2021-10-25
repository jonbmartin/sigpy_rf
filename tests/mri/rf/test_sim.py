import unittest

import numpy as np
import numpy.testing as npt
import sigpy.mri.rf as rf

import matplotlib.pyplot as pyplot

if __name__ == '__main__':
    unittest.main()


class TestSim(unittest.TestCase):

    def test_arb_phase_b1sel(self):
        print('Test of arb_phase_b1sel(rf_op, b1, mx_0, my_0, mz_0, nt) started. Please '
              'temporarily change jnp to np in function for time efficiency.')

        # test parameters (can be changed)
        dt = 1e-6
        b1 = np.arange(0, 2, 0.01)  # gauss, b1 range to sim over
        nb1 = np.size(b1)
        pbc = 1.5  # b1 (Gauss)
        pbw = 0.4  # b1 (Gauss)

        # generate rf pulse
        rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=2, ndes=256, ptype='ex', flip=np.pi / 2,
                                           pbw=pbw,
                                           pbc=[pbc], d1e=0.01, d2e=0.01,
                                           rampfilt=True, bs_offset=7500)
        full_pulse = (rfp_bs + rfp_ss) * 2 * np.pi * 4258 * dt  # scaled

        # use another simulation function to generate magnetization profile
        a, b = rf.abrm_hp(full_pulse.reshape((1, np.size(full_pulse))),
                          np.zeros(np.size(full_pulse)),
                          np.array([[1]]), 0, b1.reshape(np.size(b1), 1))
        Mxyfull = 2 * np.conj(a) * b

        # simulate with target function to generate magnetization profile
        rfp_abs = abs(full_pulse)
        rfp_angle = np.angle(full_pulse)
        nt = np.size(rfp_abs)
        rf_op = np.append(rfp_abs, rfp_angle)

        Mxd = np.zeros(nb1)
        Myd = np.zeros(nb1)
        Mzd = np.zeros(nb1)

        for ii in range(nb1):
            Mxd[ii], Myd[ii], Mzd[ii] = rf.sim.arb_phase_b1sel_loop(rf_op, b1[ii], 0, 0, 1.0, nt)

        # graphs (temp)
        pyplot.figure()
        pyplot.plot(np.sqrt(Mxd ** 2 + Myd ** 2))
        pyplot.show()

        # compare results
        npt.assert_almost_equal(abs(Mxd + 1j * Myd), abs(Mxyfull.flatten()), decimal=2)


    def test_abrm(self):
        #  also provides testing of SLR excitation. Check ex profile sim.
        tb = 8
        N = 128
        d1 = 0.01
        d2 = 0.01
        ptype = 'ex'
        ftype = 'ls'

        pulse = rf.slr.dzrf(N, tb, ptype, ftype, d1, d2, False)
        [a, b] = rf.sim.abrm(pulse, np.arange(-2 * tb, 2 * tb, 0.01), True)
        Mxy = 2 * np.multiply(np.conj(a), b)

        pts = np.array([Mxy[int(len(Mxy) / 2 - len(Mxy) / 3)],
                        Mxy[int(len(Mxy) / 2)],
                        Mxy[int(len(Mxy) / 2 + len(Mxy) / 3)]])

        npt.assert_almost_equal(abs(pts), np.array([0, 1, 0]), decimal=2)
