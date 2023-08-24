import unittest
import numpy as np
import numpy.testing as npt
from sigpy.mri.rf import optcont
from sigpy.mri.rf.slr import b2a

if __name__ == '__main__':
    unittest.main()


class TestOptcont(unittest.TestCase):

    def test_optcont1d(self):
        print('Test not fully implemented')

        try:
            gamgdt, pulse = optcont.optcont1d(4, 256, 2, 8)
        finally:
            dt = 4e-6
            gambar = 4257  # gamma/2/pi, Hz/g
            [a, b] = optcont.blochsim(pulse, x / (gambar * dt * gmag), gamgdt)
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
