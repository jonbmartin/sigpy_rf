import time
import unittest
import numpy as np
import numpy.testing as npt
import jax.numpy as jnp
from jax import jit
from sigpy.mri.rf import optcont
from matplotlib import pyplot
import sigpy.mri.rf as rf
import scipy.io as sio

if __name__ == '__main__':
    unittest.main()


class TestOptcont(unittest.TestCase):

    def test_rf_autodiff(self):
        t0 = time.time()
        print('Tests start.')

        # Experiment 1: optimize the pulse with its original target profile as sanity check
        excute = input("Start experiment 1: optimize the pulse with its original target profile "
                       "as sanity check (y/n):\n")
        if excute == 'y':
            # test parameters (can be changed)
            dt = 1e-6
            b1 = np.arange(0, 2, 0.05)  # gauss, b1 range to sim over
            nb1 = np.size(b1)
            pbc = 1.5  # b1 (Gauss)
            pbw = 0.4  # b1 (Gauss)

            # generate rf pulse
            rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=2, ndes=256, ptype='ex', flip=np.pi / 2,
                                               pbw=pbw,
                                               pbc=[pbc], d1e=0.01, d2e=0.01,
                                               rampfilt=True, bs_offset=7500)
            full_pulse = (rfp_bs + rfp_ss) * 2 * np.pi * 4258 * dt  # scaled
            print('Finish Generate rf pulse. Time: {:f}'.format(time.time() - t0))

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
                Mxd[ii], Myd[ii], Mzd[ii] = rf.sim.arb_phase_b1sel_loop(rf_op, b1[ii], 0, 0, 1.0,
                                                                        nt)

            print('Finish Simulate magnetization profile. Time: {:f}'.format(time.time() - t0))

            # huge step size to show difference
            rf_test_1 = optcont.rf_autodiff(full_pulse, b1, Mxd, Myd, Mzd, w, niters=1, step=0.1,
                                            mx0=0, my0=0, mz0=1.0)

            # compare results
            npt.assert_almost_equal(full_pulse, rf_test_1, decimal=2)
            print('Test passed. Time: {:f}'.format(time.time() - t0))

        # Experiment 2: Generate rf pulse from a pulse with different pass bandwidth and center
        excute = input("Start experiment 2: Generate rf pulse from a pulse with different pass "
                       "bandwidth and center (y/n):\n")
        if excute == 'y':
            # test parameters (can be changed)
            dt = 1e-6
            b1 = np.arange(0, 2, 0.05)  # gauss, b1 range to sim over
            nb1 = np.size(b1)
            pbc = 1.5  # b1 (Gauss)
            pbw = 0.4  # b1 (Gauss)

            # generate rf pulse
            rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=2, ndes=256, ptype='ex', flip=np.pi / 2,
                                               pbw=pbw,
                                               pbc=[pbc], d1e=0.01, d2e=0.01,
                                               rampfilt=True, bs_offset=7500)
            full_pulse = (rfp_bs + rfp_ss) * 2 * np.pi * 4258 * dt  # scaled
            print('Finish Generate rf pulse. Time: {:f}'.format(time.time() - t0))

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
                Mxd[ii], Myd[ii], Mzd[ii] = rf.sim.arb_phase_b1sel_loop(rf_op, b1[ii], 0, 0, 1.0,
                                                                        nt)

            print('Finish Simulate magnetization profile. Time: {:f}'.format(time.time() - t0))

            # generate new target profile
            pbc = 1.5  # b1 (Gauss)
            pbw = 0.2  # b1 (Gauss)
            rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=2, ndes=256, ptype='ex', flip=np.pi / 2,
                                               pbw=pbw,
                                               pbc=[pbc], d1e=0.01, d2e=0.01,
                                               rampfilt=True, bs_offset=5000)
            full_pulse_shifted = (rfp_bs + rfp_ss) * 2 * np.pi * 4258 * dt  # scaled
            nt = np.size(full_pulse_shifted)

            # visualize input pulse magnetization
            rfp_abs = abs(full_pulse_shifted)
            rfp_angle = np.angle(full_pulse_shifted)
            rf_op_test_2_ini = np.append(rfp_abs, rfp_angle)

            Mx_ini = np.zeros(nb1)
            My_ini = np.zeros(nb1)
            Mz_ini = np.zeros(nb1)

            for ii in range(nb1):
                Mx_ini[ii], My_ini[ii], Mz_ini[ii] = rf.sim.arb_phase_b1sel_loop(rf_op_test_2_ini,
                                                                                 b1[ii], 0, 0, 1.0,
                                                                                 nt)

            # optimize test pulse
            rf_test_2 = optcont.rf_autodiff(full_pulse_shifted, b1, Mxd, Myd, Mzd, w, niters=20,
                                            step=0.0001,
                                            mx0=0, my0=0, mz0=1.0)

            rfp_abs = abs(rf_test_2)
            rfp_angle = np.angle(rf_test_2)
            rf_op_test_2 = np.append(rfp_abs, rfp_angle)

            # generate magnetization profile with acquired pulse
            Mxi = np.zeros(nb1)
            Myi = np.zeros(nb1)
            Mzi = np.zeros(nb1)
            for ii in range(nb1):
                Mxi[ii], Myi[ii], Mzi[ii] = rf.sim.arb_phase_b1sel_loop(rf_op_test_2, b1[ii], 0, 0,
                                                                        1.0,
                                                                        nt)

            # # graphs (temp)
            # pyplot.figure()
            # pyplot.plot(np.sqrt(Mxd ** 2 + Myd ** 2))
            # pyplot.plot(np.sqrt(Mxi ** 2 + Myi ** 2))
            # pyplot.plot(np.sqrt(Mx_ini ** 2 + My_ini ** 2))
            # pyplot.show()

            # compare results
            # npt.assert_almost_equal(rf_op, rf_test_1, decimal=2)
            npt.assert_almost_equal(Mxi, Mxd, decimal=2)
            npt.assert_almost_equal(Myi, Myd, decimal=2)
            npt.assert_almost_equal(Mzi, Mzd, decimal=2)
            print('Test passed. Time: {:f}'.format(time.time() - t0))

        # Experiment 3: Generate rf pulse for a large flat pass band
        excute = input("Start experiment 3: Generate rf pulse for a large flat pass band"
                       " (y/n):\n")
        if excute == 'y':
            # generate initial pulse
            dt = 1e-6
            b1 = np.arange(0, 1, 0.02)  # gauss, b1 range to sim over
            nb1 = np.size(b1)
            pbc = b1[np.floor(b1.size / 2).astype(int)]  # b1 (Gauss)
            pbw = b1[-1] / 1.5  # b1 (Gauss)
            rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=2, ndes=256, ptype='st', flip=np.pi / 4,
                                               pbw=pbw,
                                               pbc=[pbc], d1e=0.01, d2e=0.01,
                                               rampfilt=True, bs_offset=5000)
            full_pulse = (rfp_bs + rfp_ss) * 2 * np.pi * 4258 * dt  # scaled

            # simulate with target function to generate magnetization profile
            rfp_abs = abs(full_pulse)
            rfp_angle = np.angle(full_pulse)
            nt = np.size(rfp_abs)
            rf_op = np.append(rfp_abs, rfp_angle)

            Mx_ini = np.zeros(nb1)
            My_ini = np.zeros(nb1)
            Mz_ini = np.zeros(nb1)

            for ii in range(nb1):
                Mx_ini[ii], My_ini[ii], Mz_ini[ii] = rf.sim.arb_phase_b1sel_loop(rf_op, b1[ii], 0,
                                                                                 0, 1.0,
                                                                                 nt)
            print('Finish setting up initial pulse and profile. Pulse duration: {:f}ms. '
                  'Time: {:f}'.format(np.size(full_pulse) * dt * 1000, (time.time() - t0)))

            # set up target profile
            # use bir4 to generate target Mx, My and Mz
            n = 1176
            dt_bir = 4e-6
            dw0 = 100 * np.pi / dt_bir / n
            beta = 10
            kappa = np.arctan(20)
            flip = np.pi / 4
            [am_bir, om_bir] = rf.adiabatic.bir4(n, beta, kappa, flip, dw0)

            bsrf = am_bir * np.exp(1j * dt * 2 * np.pi * np.cumsum(om_bir))

            rfp_abs = abs(bsrf)
            rfp_angle = np.angle(bsrf)
            nt = np.size(rfp_abs)
            rf_op = np.append(rfp_abs, rfp_angle)

            Mxd = np.zeros(nb1)
            Myd = np.zeros(nb1)
            Mzd = np.zeros(nb1)

            for ii in range(nb1):
                Mxd[ii], Myd[ii], Mzd[ii] = rf.sim.arb_phase_b1sel_loop(rf_op, b1[ii], 0,
                                                                        0, 1.0, nt)

            w = np.ones(nb1)  # weight
            print('Finish setting up target. Target bir4 duration: {:f}ms. Time: {:f}'.format(
                n * dt_bir * 1000, (time.time() - t0)))

            # Mxyd = np.ones(nb1)
            # w = np.append(np.ones(3)*0.5, np.ones(nb1-6))
            # w = np.append(w, np.ones(3)*0.5)    # weight

            # pyplot.figure()
            # pyplot.plot(np.sqrt(Mxd ** 2 + Myd ** 2))
            # pyplot.show()

            # optimize test pulse
            rf_test_3, loss = optcont.rf_autodiff(full_pulse, b1, Mxd, Myd, Mzd, w, niters=30,
                                                  step=0.000001,
                                                  mx0=0, my0=0, mz0=1.0)

            # rf_test_3 = optcont.rf_autodiff_mxy(full_pulse, b1, Mxyd, w, niters=1,
            #                                 step=0.0001,
            #                                 mx0=0, my0=0, mz0=1.0)

            rfp_abs = abs(rf_test_3)
            rfp_angle = np.angle(rf_test_3)
            nt = np.size(rfp_abs)
            rf_op_test_3 = np.append(rfp_abs, rfp_angle)
            print('Finish optimization. Time: {:f}'.format(time.time() - t0))

            # generate magnetization profile with acquired pulse
            Mxi = np.zeros(nb1)
            Myi = np.zeros(nb1)
            Mzi = np.zeros(nb1)
            for ii in range(nb1):
                Mxi[ii], Myi[ii], Mzi[ii] = rf.sim.arb_phase_b1sel_loop(rf_op_test_3, b1[ii], 0, 0,
                                                                        1.0, nt)

            # graphs (temp)
            pyplot.figure()
            # pyplot.plot(b1, Mxyd, '-b', label= 'desired Mxy')
            # pyplot.plot(b1, np.sqrt(Mxi ** 2 + Myi ** 2), '-g', label= 'final Mxy')
            # pyplot.plot(b1, np.sqrt(Mx_ini ** 2 + My_ini ** 2), '-r', label= 'initial Mxy')
            pyplot.plot(b1, np.sqrt(Mxi ** 2 + Myi ** 2), '-r', label='Mxy')
            pyplot.plot(b1, np.sqrt(Mx_ini ** 2 + My_ini ** 2), '-g', label='Mxy initial')
            pyplot.plot(b1, np.sqrt(Mxd ** 2 + Myd ** 2), '-b', label='Mxy desired')
            pyplot.plot(b1, Mzi, '-c', label='Mz')
            pyplot.legend()
            pyplot.show()

            pyplot.figure()
            pyplot.plot(loss, '-c', label='loss')
            pyplot.legend()
            pyplot.show()

            pyplot.figure()
            pyplot.plot(abs(full_pulse).T, '-r', label='rf pulse initial')
            pyplot.plot(abs(rf_test_3).T, '-g', label='rf pulse final')
            # pyplot.plot(abs(bsrf).T, '-b', label='bir4 pulse')
            pyplot.legend()
            pyplot.show()

            # compare results
            # npt.assert_almost_equal(rf_op, rf_test_1, decimal=2)
            npt.assert_almost_equal(np.sqrt(Mxi ** 2 + Myi ** 2), np.sqrt(Mxd ** 2 + Myd ** 2),
                                    decimal=1)
            print('Test passed. Time: {:f}'.format(time.time() - t0))

        # Experiment 4: Refine an inversion pulse
        excute = input("Start experiment 4: Refine an inversion pulse"
                       " (y/n):\n")
        if excute == 'y':
            # load test pulse
            dict = sio.loadmat(
                '/nas/home/sunh11/inversion_pulse_refinement/inversion_pulse_to_refine_PBC1d4PBW0d3.mat')
            dt = 4e-6
            b1 = np.squeeze(dict['b1_grid'])  # gauss, b1 range to sim over
            nb1 = np.size(b1)
            w = np.squeeze(dict['w'])
            Mzd = np.squeeze(dict['ideal_mz'])
            full_pulse = np.squeeze(dict['full_pulse']) * 2 * np.pi * 4258 * dt

            # simulate with target function to generate magnetization profile
            rfp_abs = abs(full_pulse)
            rfp_angle = np.angle(full_pulse)
            nt = np.size(rfp_abs)
            rf_op = np.append(rfp_abs, rfp_angle)

            Mx_ini = np.zeros(nb1)
            My_ini = np.zeros(nb1)
            Mz_ini = np.zeros(nb1)

            for ii in range(nb1):
                Mx_ini[ii], My_ini[ii], Mz_ini[ii] = rf.sim.arb_phase_b1sel_loop(rf_op, b1[ii], 0,
                                                                                 0, 1.0, nt)
            print('Finish setting up initial pulse and profile. Pulse duration: {:f}ms. '
                  'Time: {:f}'.format(np.size(full_pulse) * dt * 1000, (time.time() - t0)))

            # optimize test pulse
            niters = 150
            step_size = 0.00010  # JBM was 0.00005
            rf_test_4, loss = optcont.rf_autodiff_mz(full_pulse, b1, Mzd, w, niters,
                                                     step_size,
                                                     mx0=0, my0=0, mz0=1.0)

            rfp_abs = abs(rf_test_4)
            rfp_angle = np.angle(rf_test_4)
            nt = np.size(rfp_abs)
            rf_op_test_4 = np.append(rfp_abs, rfp_angle)
            print('Finish optimization. Time: {:f}'.format(time.time() - t0))

            # generate magnetization profile with acquired pulse
            Mxi = np.zeros(nb1)
            Myi = np.zeros(nb1)
            Mzi = np.zeros(nb1)
            for ii in range(nb1):
                Mxi[ii], Myi[ii], Mzi[ii] = rf.sim.arb_phase_b1sel_loop(rf_op_test_4, b1[ii], 0, 0,
                                                                        1.0, nt)

            # graphs (temp)
            pyplot.figure()
            pyplot.plot(b1, np.sqrt(Mxi ** 2 + Myi ** 2), '-r', label='Mxy')
            pyplot.plot(b1, np.sqrt(Mx_ini ** 2 + My_ini ** 2), '-g', label='Mxy initial')
            pyplot.plot(b1, Mzd, '-b', label='Mz desired')
            pyplot.plot(b1, Mzi, '-c', label='Mz')
            pyplot.plot(b1, Mz_ini, '-m', label='Mz initial')
            pyplot.plot(b1, w, '-y', label='weight')
            pyplot.legend()
            pyplot.show()

            pyplot.figure()
            pyplot.plot(loss, '-c', label='loss')
            pyplot.legend()
            pyplot.show()

            pyplot.figure()
            pyplot.plot(abs(full_pulse).T, '-r', label='rf pulse initial')
            pyplot.plot(abs(rf_test_4).T, '-g', label='rf pulse final')
            pyplot.legend()
            pyplot.show()

            # save the pulse regardless, no prompting
            # excute = input("Save the results? (y/n):\n")
            # if excute == 'y':
            print('Saving Results')
            dict_out = {'Mx_ini': Mx_ini, 'My_ini': My_ini, 'Mz_ini': Mz_ini,
                        'Mx_fin': Mxi, 'My_fin': Myi, 'Mz_fin': Mzi,
                        'pulse_ini': full_pulse, 'pulse_ref': rf_test_4,
                        'niters': niters, 'step_size': step_size,
                        'weight': w, 'loss': loss, 'run_time': time.time() - t0}
            sio.savemat('/nas/home/sunh11/inversion_pulse_refinement'
                        '/after_refine_50_pbc1d4pbw0d3'
                        '.mat'
                        , dict_out)

            # compare results
            npt.assert_almost_equal(Mzi, Mzd, decimal=1)
            print('Test passed. Time: {:f}'.format(time.time() - t0))

        # Experiment 5: Generate rf pulse for a large flat pass band (iterative)
        excute = input("Start experiment 5: Generate rf pulse for a large flat pass band (iters)"
                       " (y/n):\n")
        if excute == 'y':
            excute = input("Start with new pulse? (y/n):\n")
            if excute == 'y':
                # generate initial pulse
                dt = 1e-6
                b1 = np.arange(0, 1, 0.02)  # gauss, b1 range to sim over
                nb1 = np.size(b1)
                pbc = b1[np.floor(b1.size / 2).astype(int)]  # b1 (Gauss)
                pbw = b1[-1] / 1.5  # b1 (Gauss)
                rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=2, ndes=256, ptype='st', flip=np.pi / 4,
                                               pbw=pbw,
                                               pbc=[pbc], d1e=0.01, d2e=0.01,
                                               rampfilt=True, bs_offset=5000)
                full_pulse = (rfp_bs + rfp_ss) * 2 * np.pi * 4258 * dt  # scaled

                # simulate with target function to generate magnetization profile
                rfp_abs = abs(full_pulse)
                rfp_angle = np.angle(full_pulse)
                nt = np.size(rfp_abs)
                rf_op = np.append(rfp_abs, rfp_angle)

                Mx_ini = np.zeros(nb1)
                My_ini = np.zeros(nb1)
                Mz_ini = np.zeros(nb1)

                for ii in range(nb1):
                    Mx_ini[ii], My_ini[ii], Mz_ini[ii] = rf.sim.arb_phase_b1sel_loop(rf_op, b1[ii], 0,
                                                                                 0, 1.0, nt)
                print('Finish setting up initial pulse and profile. Pulse duration: {:f}ms. '
                  'Time: {:f}'.format(np.size(full_pulse) * dt * 1000, (time.time() - t0)))

                # set up target
                Mxyd = np.ones(nb1)
                Mxd = np.zeros(nb1)
                Myd = np.zeros(nb1)

                w = np.append(np.zeros(10), np.ones(nb1-10-10))
                w = np.append(w, np.zeros(10))      # weight
                niters = 0
                step_size = 0.00010

            else:
                # load test pulse
                name = input("Load previous pulse (file name only, not including .mat):\n")
                dict = sio.loadmat(
                    '/nas/home/sunh11/test_pulses/'+name+'.mat')
                dt = np.squeeze(dict['dt'])
                b1 = np.squeeze(dict['b1'])  # gauss, b1 range to sim over
                nb1 = np.size(b1)

                full_pulse = np.squeeze(dict['pulse_fin'])
                Mx_ini = np.squeeze(dict['Mx_fin'])
                My_ini = np.squeeze(dict['My_fin'])
                Mz_ini = np.squeeze(dict['Mz_fin'])

                w = np.squeeze(dict['weight'])
                Mxd = np.squeeze(dict['Mx_dsr'])
                Myd = np.squeeze(dict['My_dsr'])
                Mxyd = np.squeeze(dict['Mxy_dsr'])
                niters = np.squeeze(dict['niters'])
                step_size = np.squeeze(dict['step_size'])

            # loop parameters
            loop_count = 0
            proceed = 'y'
            new_iters = 1
            Mxi = np.zeros(nb1)
            Myi = np.zeros(nb1)
            Mzi = np.zeros(nb1)
            rf_test_5 = full_pulse
            loss = 0

            while proceed == 'y':
                # get new iter numbers
                # new_iters = int(input("Enter the number of new iterations to perform (y/n):\n"))

                # get optimize direction
                direction = input("Optimize Mx or My? (x/y):\n")
                if direction == 'x':
                    Mxd = np.sqrt(Mxyd ** 2 - My_ini ** 2)
                    Myd = My_ini
                else:
                    Myd = np.sqrt(Mxyd**2-Mx_ini**2)
                    Mxd = Mx_ini

                # optimize test pulse
                rf_test_5, loss = optcont.rf_autodiff_mx_my(full_pulse, b1, Mxd, Myd, w, new_iters,
                                                      step_size,
                                                      mx0=0, my0=0, mz0=1.0)

                rfp_abs = abs(rf_test_5)
                rfp_angle = np.angle(rf_test_5)
                nt = np.size(rfp_abs)
                rf_op_test_5 = np.append(rfp_abs, rfp_angle)
                print('Finish optimization. Time: {:f}'.format(time.time() - t0))

                # generate magnetization profile with acquired pulse
                for ii in range(nb1):
                    Mxi[ii], Myi[ii], Mzi[ii] = rf.sim.arb_phase_b1sel_loop(rf_op_test_5, b1[ii], 0, 0,
                                                                            1.0, nt)

                # graphs (temp)
                pyplot.figure()
                pyplot.plot(b1, np.sqrt(Mxi ** 2 + Myi ** 2), '-r', label='Mxy')
                pyplot.plot(b1, np.sqrt(Mx_ini ** 2 + My_ini ** 2), '-g', label='Mxy initial')
                pyplot.plot(b1, np.sqrt(Mxd ** 2 + Myd ** 2), '-b', label='Mxy desired')
                pyplot.plot(b1, Mzi, '-c', label='Mz')
                pyplot.legend()
                pyplot.show()

                pyplot.figure()
                pyplot.plot(loss, '-c', label='loss')
                pyplot.legend()
                pyplot.show()

                pyplot.figure()
                pyplot.plot(abs(full_pulse).T, '-r', label='rf pulse initial')
                pyplot.plot(abs(rf_test_5).T, '-g', label='rf pulse final')
                pyplot.legend()
                pyplot.show()

                # record loop and prompt for another
                loop_count += 1
                proceed = input("Start another loop (y/n):\n")

            # save the pulse regardless
            # excute = input("Save the results? (y/n):\n")
            # if excute == 'y':
            print('Saving Results')
            dict_out = {'Mx_ini': Mx_ini, 'My_ini': My_ini, 'Mz_ini': Mz_ini,
                        'Mx_fin': Mxi, 'My_fin': Myi, 'Mz_fin': Mzi,
                        'Mx_dsr': Mxd, 'My_dsr': Myd, 'Mxy_dsr': Mxyd,
                        'pulse_ini': full_pulse, 'pulse_fin': rf_test_5,
                        'niters': niters + new_iters * loop_count, 'step_size': step_size,
                        'weight': w, 'loss': loss, 'run_time': time.time() - t0,
                        'dt': dt, 'b1': b1}
            sio.savemat('/nas/home/sunh11/test_pulses'
                        '/mxyrefine1119'
                        '.mat'
                        , dict_out)

            # # compare results
            # # npt.assert_almost_equal(rf_op, rf_test_1, decimal=2)
            # npt.assert_almost_equal(np.sqrt(Mxi ** 2 + Myi ** 2), np.sqrt(Mxd ** 2 + Myd ** 2),
            #                         decimal=1)
            # print('Test passed. Time: {:f}'.format(time.time() - t0))

    def test_optcont1d(self):
        print('Test not fully implemented')

        try:
            gamgdt, pulse = optcont.optcont1d(4, 256, 2, 8)
        finally:
            dt = 4e-6
            gambar = 4257  # gamma/2/pi, Hz/g
            [a, b] = rf.optcont.blochsim(pulse, x / (gambar * dt * gmag), gamgdt)
            Mxy = 2 * np.conj(a) * b

            pyplot.figure()
            pyplot.figure(np.abs(Mxy))
            pyplot.show()

            # TODO: compare with target Mxy, take integration

            alpha = rf.b2a(db)

    def test_blochsim(self):
        print('Test not implemented')
        # TODO: insert testing

    def test_deriv(self):
        print('Test not implemented')
        # TODO: insert testing
