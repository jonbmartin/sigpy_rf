import os
import struct
import unittest
import numpy as np
import numpy.testing as npt
from sigpy.mri.rf import io

if __name__ == '__main__':
    unittest.main()


class TestIo(unittest.TestCase):

    def test_signa(self):
        # TODO: finish testing, might add test for scale param
        test_filename = 'temp'
        real_wav = np.array(
            [327.8703, 17.8558, 424.5647, 466.9966, 339.3676, 378.8701, 371.5662, 196.1135,
             327.7389, 85.5933, 353.0230, 15.9164, 138.4615, 23.0857, 48.5659, 411.7289,
             347.4143, 158.5497, 475.1110, 17.2230])
        expected_real = (22612, 1232, 29280, 32206, 23404, 26128, 25626, 13524, 22602, 5902, 24346,
                         1098, 9548, 1592, 3350, 28394, 23960, 10934, 32766, 1188)
        complex_wav = np.array(
            [426.52 + 390.13 * 1j, 311.03 + 194.87 * 1j, 175.48 + 120.85 * 1j, 256.62 + 201.96 * 1j,
             200.90 + 48.23 * 1j, 37.98 + 65.99 * 1j, 119.96 + 471.03 * 1j, 61.66 + 478.07 * 1j,
             91.95 + 287.60 * 1j, 119.98 + 29.89 * 1j, 208.63 + 117.39 * 1j, 24.83 + 176.58 * 1j,
             451.36 + 410.60 * 1j, 472.39 + 7.70 * 1j, 245.43 + 21.51 * 1j, 244.63 + 84.50 * 1j,
             168.86 + 324.56 * 1j, 450.03 + 365.86 * 1j, 184.62 + 323.87 * 1j, 55.60 + 225.46 * 1j]
        )
        expected_comp_real = (29232, 21318, 12028, 17588, 13770, 2604, 8222, 4226, 6302, 8224,
                              14300, 1702, 30936, 32376, 16822, 16766, 11574, 30844, 12654, 3810)
        expected_comp_imag = (26738, 13356, 8282, 13842, 3306, 4522, 32284, 32766, 19712, 2048,
                              8046, 12102, 28142, 528, 1474, 5792, 22244, 25076, 22198, 15452)
        contents = ()
        contents_real = ()
        contents_imag = ()

        try:
            io.signa(real_wav, test_filename)
            with open(test_filename, mode='rb') as file:
                file_content = file.read(2)
                while file_content:
                    data = struct.unpack('>h', file_content)
                    contents = contents + data
                    file_content = file.read(2)
        except FileNotFoundError as e:
            print('Fail to create/open: ' + test_filename)
        finally:
            os.remove(test_filename)
        self.assertEqual(contents, expected_real, "GE output not as expected for real input.")

        try:
            io.signa(complex_wav, test_filename)
            with open(test_filename + '.r', mode='rb') as file:
                file_content = file.read(2)
                while file_content:
                    data = struct.unpack('>h', file_content)
                    contents_real = contents_real + data
                    file_content = file.read(2)
            with open(test_filename + '.i', mode='rb') as file:
                file_content = file.read(2)
                while file_content:
                    data = struct.unpack('>h', file_content)
                    contents_imag = contents_imag + data
                    file_content = file.read(2)
        except OSError as e:
            print('Fail to create/open: ' + test_filename + '.r or .i')
        finally:
            os.remove(test_filename + '.r')
            os.remove(test_filename + '.i')
        self.assertEqual(contents_real, expected_comp_real,
                         "GE output not as expected for complex input.")
        self.assertEqual(contents_imag, expected_comp_imag,
                         "GE output not as expected for complex input.")
        