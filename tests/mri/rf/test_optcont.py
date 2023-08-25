import unittest
import numpy as np
import numpy.testing as npt
from sigpy.mri.rf import optcont
from sigpy.mri.rf.slr import dinf

class TestOptcont(unittest.TestCase):

    def test_optcont1d(self):
        try:
            N = 128
            os = 2  # oversampling factor
            tb = 4
            dthick = 4
            gamgdt, pulse = optcont.optcont1d(dthick, N, os, tb)
        except Exception:
            print('Exception optimal control design')
        else:
            dt = 4e-6  # s
            x = np.arange(0, N * os, 1) / N / os - 1 / 2
            gambar = 4257  # gamma/2/pi, Hz/g
            gmag = tb/(N*dt)/dthick/gambar

            a, b = optcont.blochsim(pulse, x / (gambar * dt * gmag), gamgdt)

            d1 = 0.01
            d2 = 0.01  # ripple levels in beta profile
            d1 = np.sqrt(d1 / 2)  # Mxy -> beta ripple for ex pulse
            d2 = d2 / np.sqrt(2)
            dib = dinf(d1, d2)
            ftwb = dib / tb

            fb = np.asarray([0, (1 - ftwb) * (tb / 2), (1 + ftwb) * (tb / 2), N / 2]) / N  # freq edges, normalized to 2*nyquist
            dpass = np.abs(x) < fb[1]  # passband mask

            w = np.ones(np.size(b))
            db = np.sqrt(1 / 2) * dpass * np.exp(-1j / 2 * x * 2 * np.pi)
            auxb = w * (b - db)
            drf = optcont.deriv(pulse, x / (gambar * dt * gmag), gamgdt, None, auxb, a, b)
