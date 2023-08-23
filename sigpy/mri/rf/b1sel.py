# -*- coding: utf-8 -*-
""":math:`B_1^{+}`-selective RF Pulse Design functions.

"""
import numpy as np

from sigpy.mri.rf import slr as slr
from sigpy.mri.rf.util import dinf, b12wbs, calc_kbs
from scipy.interpolate import interp1d

__all__ = ['dz_bssel_rf', 'bssel_bs', 'dz_b1_rf', 'bssel_ex_slr',
           'dz_b1_gslider_rf', 'dz_b1_hadamard_rf']


def dz_bssel_rf(dt=2e-6, 
                tb=4, 
                short_rat=1, 
                ndes=128, 
                ptype='ex', 
                flip=np.pi/4,
                pbw=0.25, 
                pbc=[1], 
                d1e=0.01, 
                d2e=0.01,
                rampfilt=True, 
                bs_offset=20000,
                fa_correct=True,):
    """Design a math:`B_1^{+}`-selective pulse following J Martin's
    Bloch Siegert method.

    Args:
        dt (float): hardware sampling dwell time in s.
        tb (int): time-bandwidth product.
        short_rat (float): ratio of duration of desired pulse to duration
            required by nyquist. Can shorten pulse at expense of profile.
        ndes (int): number of taps in filter design.
        ptype (string): pulse type, 'st' (small-tip excitation), 'ex' (pi/2
            excitation pulse), 'se' (spin-echo pulse), 'inv' (inversion), or
            'sat' (pi/2 saturation pulse).
        flip (float): flip angle, in radians. Only required for ptype 'st',
            implied for other ptypes.
        pbw (float): width of passband in Gauss.
        pbc (list of floats): center of passband(s) in Gauss.
        d1e (float): passband ripple level in :math:`M_0^{-1}`.
        d2e (float): stopband ripple level in :math:`M_0^{-1}`.
        rampfilt (bool): option to directly design the modulated filter, to
            compensate b1 variation across a slice profile.
        bs_offset (float): (Hz) constant offset during pulse.
        fa_correct (bool): option to apply empirical flip angle correction.

    Returns:
        3-element tuple containing

        - **bsrf** (*array*): complex bloch-siegert gradient waveform.
        - **rfp** (*array*): complex slice-selecting waveform.
        - **rw** (*array*): complex bloch-siegert rewinder

    References:
        Martin, J., Vaughn, C., Griswold, M., & Grissom, W. (2021).
        Bloch-Siegert |B 1+ |-Selective Excitation Pulses.
        Proc. Intl. Soc. Magn. Reson. Med.
    """

    beta = 0.5  # AM waveform parameter for fermi sweeps # JBM was 0.5
    nsw = np.round(1250e-6 / dt)  # number of time points in sweeps
    kappa = np.arctan(2)  # FM waveform parameter

    # calculate bandwidth and pulse duration using lowest PBC of bands. Lower
    # PBC's require a longer pulse, so lowest constrains our pulse length
    upper_b1 = min(pbc) + pbw / 2
    lower_b1 = min(pbc) - pbw / 2

    # using Ramsey's BS shift equation pre- w_rf >> gam*b1 approximation
    B = b12wbs(bs_offset, upper_b1) - b12wbs(bs_offset,lower_b1)
    Tex = (tb / B) * short_rat  # seconds, the entire pulse duration

    # perform the design of the BS far off resonant pulse
    bsrf, rw, phi_bs = bssel_bs(Tex, dt, bs_offset)

    # design pulse for number of bands desired
    if len(pbc) == 1:
        rfp, phi_ex = bssel_ex_slr(Tex, dt, tb, ndes, ptype, flip, pbw, pbc[0],
                                   d1e, d2e, rampfilt, bs_offset, fa_correct)

    # repeat design for multiple bands of excitation
    else:
        rfp = np.zeros((1, np.int(np.ceil(Tex / dt / 2) * 2)), dtype=complex)
        for ii in range(0, len(pbc)):
            upper_b1 = pbc[ii] + pbw / 2
            lower_b1 = pbc[ii] - pbw / 2
            B_i = bs_offset * ((1 + (4258 * upper_b1) ** 2 / bs_offset ** 2) ** (
                        1 / 2) - 1) - \
                bs_offset * ((1 + (4258 * lower_b1) ** 2 / bs_offset ** 2) ** (
                        1 / 2) - 1)
            T_i = tb / B_i  # seconds, the entire pulse duration
            ex_subpulse = bssel_ex_slr(T_i, dt, tb, ndes, ptype, flip, pbw,
                                       pbc[ii], d1e, d2e, rampfilt,
                                       bs_offset)

            # zero pad to match the length of the longest pulse
            if ii > 0:
                zpad = np.zeros((1, np.size(rfp)-np.size(ex_subpulse)))
                zp1 = zpad[:, :np.size(zpad)//2]
                zp2 = zpad[:, (np.size(zpad))//2:]

                ex_subpulse = np.concatenate([zp1, ex_subpulse, zp2], axis=1)
            rfp += ex_subpulse

    # zero-pad it to the same length as bs
    nsw = int(np.ceil((np.size(bsrf) - np.size(rfp))/2))
    rfp = np.concatenate([np.zeros((1, np.int(nsw))), rfp], axis=1)
    rfp = np.concatenate([rfp,np.zeros((1,np.size(bsrf)-np.size(rfp)))], axis=1)

    # return the subpulses. User should superimpose bsrf and rfp if desired
    return bsrf, rfp, rw


def bssel_bs(T, dt, bs_offset):
    """Design the Bloch-Siegert shift inducing component pulse for a
     math:`B_1^{+}`-selective pulse following J Martin's Bloch Siegert method.

        Args:
            T (float): total pulse duration (s).
            dt (float): hardware sampling dwell time (s).
            bs_offset (float): constant offset during pulse (Hz).

        Returns:
            2-element tuple containing

            - **bsrf** (*array*): complex BS pulse.
            - **bsrf_rew** (*array*): FM waveform (radians/s).

        References:
            Martin, J., Vaughn, C., Griswold, M., & Grissom, W. (2021).
            Bloch-Siegert |B 1+ |-Selective Excitation Pulses.
            Proc. Intl. Soc. Magn. Reson. Med.
        """

    a = 0.00006
    Bth = 0.95
    t0 = T/2 - a*np.log((1-Bth)/Bth)
    T_full = 2*t0 +13.81 * a
    t = np.arange(-T_full/2, T_full/2, dt)
    bs_am = 1 / (1 + np.exp((np.abs(t)-t0)/a))
    if np.mod(np.size(bs_am), 2) != 0:
        bs_am = bs_am[:-1]

    A_half = bs_am[0:int(np.size(bs_am)/2)]
    gam = 4258
    k = 0.2
    t_v = np.arange(dt, T_full/2*dt+dt, dt)
    om = (gam*A_half)/np.sqrt((1-(gam*A_half*abs(t_v))/k)**(-2)-1)

    om -= np.max(abs(om))
    om = np.expand_dims(om*1,0)
    bs_fm = np.concatenate([-om, np.fliplr(-om)],axis=1) + bs_offset
    kbs_bs = calc_kbs(bs_am, bs_fm, T)

    bsrf = bs_am * np.exp(1j * dt * 2 * np.pi * np.cumsum(bs_fm))
    bsrf = np.expand_dims(bsrf,0)
    phi_bs = np.cumsum((4258*bs_am)**2/(2*bs_fm))

    # Build an RF rewinder, same amplitude but shorter duration to produce -0.5
    # the Kbs. Pull middle samples until duration matched
    bs_am_rew = np.ndarray.tolist(np.squeeze(bs_am))
    bs_fm_rew = np.ndarray.tolist(np.squeeze(-bs_fm))
    kbs_rw = -kbs_bs
    while abs(kbs_rw) > 0.5 * abs(kbs_bs):
        mid = len(bs_am_rew)//2
        bs_am_rew = bs_am_rew[:mid] + bs_am_rew[mid+1:]
        bs_fm_rew = bs_fm_rew[:mid] + bs_fm_rew[mid+1:]
        kbs_rw = calc_kbs(bs_am_rew, bs_fm_rew, len(bs_am_rew)*dt)

    # adjust amplitude to precisely give correct Kbs
    bs_am_rew = np.array(bs_am_rew) * np.sqrt(abs(kbs_bs/(2*kbs_rw)))
    kbs_rw = calc_kbs(bs_am_rew, bs_fm_rew, len(bs_am_rew) * dt)
    bsrf_rew = np.array(bs_am_rew) * np.exp(1j * dt * 2 * np.pi * np.cumsum(np.array(bs_fm_rew)))
    print('RW kbs = {}'.format(kbs_rw))

    return bsrf, bsrf_rew, phi_bs


def bssel_ex_slr(T, dt=2e-6, tb=4, ndes=128, ptype='ex', flip=np.pi/2,
                 pbw=0.25, pbc=1, d1e=0.01, d2e=0.01, rampfilt=True,
                 bs_offset=20000, fa_correct=True):

    n = np.int(np.ceil(T / dt / 2) * 2)  # samples in final pulse, force even

    if not rampfilt:
        # straightforward SLR design, no ramp
        rfp = slr.dzrf(ndes, tb, ptype, 'ls', d1e, d2e)
        rfp = np.expand_dims(rfp, 0)
    else:
        # perform a filtered design that compensates the b1 variation across
        # the slice. Here, calc parameter relations
        bsf, d1, d2 = slr.calc_ripples(ptype, d1e, d2e)

        # create a beta that corresponds to a ramp
        b = slr.dz_ramp_beta(ndes, T, ptype, pbc, pbw, bs_offset, tb, d1, d2, dt)

        if ptype == 'st':
            rfp = b
        else:
            # inverse SLR transform to get the pulse
            b = bsf * b
            rfp = slr.b2rf(np.squeeze(b))
            rfp = np.expand_dims(rfp, 0)

    # interpolate to target dwell time
    rfinterp = interp1d(np.linspace(-T / 2, T / 2, ndes), rfp, kind='cubic')
    trf = np.linspace(-T / 2, T / 2, n)
    rfp = rfinterp(trf)
    rfp = rfp * ndes / n

    # scale for desired flip if ptype 'st'
    if ptype == 'st':
        rfp = rfp / np.sum(rfp) * flip / (2 * np.pi * 4258 * dt)  # gauss
    else:  # rf is already in radians in other cases
        rfp = rfp / (2 * np.pi * 4258 * dt)

    # slice select modulation is middle of upper and lower b1
    upper_b1 = pbc + pbw / 2
    lower_b1 = pbc - pbw / 2
    rfp_modulation = 0.5*(b12wbs(bs_offset, upper_b1) + b12wbs(bs_offset, lower_b1))
    print(f'SS modulation = {rfp_modulation} Hz')

    # empirical correction factor for scaling
    if fa_correct:

        scalefact = pbc*(0.3323*np.exp(-0.9655*(rfp_modulation/bs_offset))
                         + 0.6821*np.exp(-0.02331*(rfp_modulation/bs_offset)))
        rfp = rfp / scalefact
    else:
        rfp = rfp / pbc

    # modulate RF to be centered at the passband. complex modulation => 1 band!
    t = np.linspace(- np.int(T / dt / 2), np.int(T / dt / 2), np.size(rfp))
    rfp = rfp * np.exp(-1j * 2 * np.pi * rfp_modulation * t * dt)

    phi_bs = np.cumsum((4258*np.real(rfp))**2/(2*rfp_modulation))
    return rfp, phi_bs


__all__ = ["dz_b1_rf", "dz_b1_gslider_rf", "dz_b1_hadamard_rf"]


def dz_b1_rf(
    dt=2e-6,
    tb=4,
    ptype="st",
    flip=np.pi / 6,
    pbw=0.3,
    pbc=2,
    d1=0.01,
    d2=0.01,
    os=8,
    split_and_reflect=True,
):
    """Design a :math:`B_1^{+}`-selective excitation pulse following Grissom \
    JMR 2014

    Args:
        dt (float): hardware sampling dwell time in s.
        tb (int): time-bandwidth product.
        ptype (string): pulse type, 'st' (small-tip excitation), 'ex' (pi/2
            excitation pulse), 'se' (spin-echo pulse), 'inv' (inversion), or
            'sat' (pi/2 saturation pulse).
        flip (float): flip angle, in radians.
        pbw (float): width of passband in Gauss.
        pbc (float): center of passband in Gauss.
        d1 (float): passband ripple level in :math:`M_0^{-1}`.
        d2 (float): stopband ripple level in :math:`M_0^{-1}`.
        os (int): matrix scaling factor.
        split_and_reflect (bool): option to split and reflect designed pulse.

    Split-and-reflect preserves pulse selectivity when scaled to excite large
    tip-angles.

    Returns:
        2-element tuple containing

        - **om1** (*array*): AM waveform.
        - **dom** (*array*): FM waveform (radians/s).

    References:
        Grissom, W., Cao, Z., & Does, M. (2014).
        :math:`B_1^{+}`-selective excitation pulse design using the Shinnar-Le
        Roux algorithm. Journal of Magnetic Resonance, 242, 189-196.
    """

    # calculate beta filter ripple
    [_, d1, d2] = slr.calc_ripples(ptype, d1, d2)

    # calculate pulse duration
    b = 4257 * pbw
    pulse_len = tb / b

    # calculate number of samples in pulse
    n = int(np.ceil(pulse_len / dt / 2) * 2)

    if pbc == 0:
        # we want passband as close to zero as possible.
        # do my own dual-band filter design to minimize interaction
        # between the left and right bands

        # build system matrix
        A = np.exp(
            1j
            * 2
            * np.pi
            * np.outer(
                np.arange(-n * os / 2, n * os / 2), np.arange(-n / 2, n / 2)
            )
            / (n * os)
        )

        # build target pattern
        ii = np.arange(-n * os / 2, n * os / 2) / (n * os) * 2
        w = dinf(d1, d2) / tb
        f = np.asarray([0, (1 - w) * (tb / 2), (1 + w) * (tb / 2), n / 2]) / (
            n / 2
        )
        d = np.double(np.abs(ii) < f[1])
        ds = np.double(np.abs(ii) > f[2])

        # shift the target pattern to minimum center position
        pbc = int(np.ceil((f[2] - f[1]) * n * os / 2 + f[1] * n * os / 2))
        dl = np.roll(d, pbc)
        dr = np.roll(d, -pbc)
        dsl = np.roll(ds, pbc)
        dsr = np.roll(ds, -pbc)

        # build error weight vector
        w = dl + dr + d1 / d2 * np.multiply(dsl, dsr)

        # solve for the dual-band filter
        AtA = A.conj().T @ np.multiply(np.reshape(w, (np.size(w), 1)), A)
        Atd = A.conj().T @ np.multiply(w, dr - dl)
        h = np.imag(np.linalg.pinv(AtA) @ Atd)

    else:  # normal design
        # design filter
        h = slr.dzls(n, tb, d1, d2)

        # dual-band-modulate the filter
        om = 2 * np.pi * 4257 * pbc  # modulation frequency
        t = np.arange(0, n) * pulse_len / n - pulse_len / 2
        h = 2 * h * np.sin(om * t)

    if split_and_reflect:
        # split and flip fm waveform to improve large-tip accuracy
        dom = np.concatenate((h[n // 2 :: -1], h, h[n : n // 2 : -1])) / 2
    else:
        dom = np.concatenate((0 * h[n // 2 :: -1], h, 0 * h[n : n // 2 : -1]))

    # scale to target flip, convert to Hz
    dom = dom * flip / (2 * np.pi * dt)

    # build am waveform
    om1 = np.concatenate((-np.ones(n // 2), np.ones(n), -np.ones(n // 2)))

    return om1, dom


def dz_b1_gslider_rf(
    dt=2e-6,
    g=5,
    tb=12,
    ptype="st",
    flip=np.pi / 6,
    pbw=0.5,
    pbc=2,
    d1=0.01,
    d2=0.01,
    split_and_reflect=True,
):
    """Design a :math:`B_1^{+}`-selective excitation gSlider pulse following
     Grissom JMR 2014.

    Args:
        dt (float): hardware sampling dwell time in s.
        g (int): number of slabs to be acquired.
        tb (int): time-bandwidth product.
        ptype (string): pulse type, 'st' (small-tip excitation), 'ex' (pi/2
            excitation pulse), 'se' (spin-echo pulse), 'inv' (inversion), or
            'sat' (pi/2 saturation pulse).
        flip (float): flip angle, in radians.
        pbw (float): width of passband in Gauss.
        pbc (float): center of passband in Gauss.
        d1 (float): passband ripple level in :math:`M_0^{-1}`.
        d2 (float): stopband ripple level in :math:`M_0^{-1}`.
        split_and_reflect (bool): option to split and reflect designed pulse.

    Split-and-reflect preserves pulse selectivity when scaled to excite large
     tip-angles.

    Returns:
        2-element tuple containing

        - **om1** (*array*): AM waveform.
        - **dom** (*array*): FM waveform (radians/s).

    References:
        Grissom, W., Cao, Z., & Does, M. (2014).
        :math:`B_1^{+}`-selective excitation pulse design using the Shinnar-Le
        Roux algorithm. Journal of Magnetic Resonance, 242, 189-196.
    """

    # calculate beta filter ripple
    [_, d1, d2] = slr.calc_ripples(ptype, d1, d2)
    # if ptype == 'st':
    bsf = flip

    # calculate pulse duration
    b = 4257 * pbw
    pulse_len = tb / b

    # calculate number of samples in pulse
    n = int(np.ceil(pulse_len / dt / 2) * 2)

    om = 2 * np.pi * 4257 * pbc  # modulation freq to center profile at pbc
    t = np.arange(0, n) * pulse_len / n - pulse_len / 2

    om1 = np.zeros((2 * n, g))
    dom = np.zeros((2 * n, g))
    for gind in range(1, g + 1):
        # design filter
        h = bsf * slr.dz_gslider_b(n, g, gind, tb, d1, d2, np.pi, n // 4)

        # modulate filter to center and add it to a time-reversed and modulated
        # copy, then take the imaginary part to get an odd filter
        h = np.imag(h * np.exp(1j * om * t) - h[n::-1] * np.exp(1j * -om * t))
        if split_and_reflect:
            # split and flip fm waveform to improve large-tip accuracy
            dom[:, gind - 1] = (
                np.concatenate((h[n // 2 :: -1], h, h[n : n // 2 : -1])) / 2
            )
        else:
            dom[:, gind - 1] = np.concatenate(
                (0 * h[n // 2 :: -1], h, 0 * h[n : n // 2 : -1])
            )
        # build am waveform
        om1[:, gind - 1] = np.concatenate(
            (-np.ones(n // 2), np.ones(n), -np.ones(n // 2))
        )

    # scale to target flip, convert to Hz
    dom = dom / (2 * np.pi * dt)

    return om1, dom


def dz_b1_hadamard_rf(
    dt=2e-6,
    g=8,
    tb=16,
    ptype="st",
    flip=np.pi / 6,
    pbw=2,
    pbc=2,
    d1=0.01,
    d2=0.01,
    split_and_reflect=True,
):
    """Design a :math:`B_1^{+}`-selective Hadamard-encoded pulse following \
     Grissom JMR 2014.
    Args:
        dt (float): hardware sampling dwell time in s.
        g (int): number of slabs to be acquired.
        tb (int): time-bandwidth product.
        ptype (string): pulse type, 'st' (small-tip excitation), 'ex' (pi/2 \
            excitation pulse), 'se' (spin-echo pulse), 'inv' (inversion), or \
            'sat' (pi/2 saturation pulse).
        flip (float): flip angle, in radians.
        pbw (float): width of passband in Gauss.
        pbc (float): center of passband in Gauss.
        d1 (float): passband ripple level in :math:`M_0^{-1}`.
        d2 (float): stopband ripple level in :math:`M_0^{-1}`.
        split_and_reflect (bool): option to split and reflect designed pulse.

    Split-and-reflect preserves pulse selectivity when scaled to excite large
    tip-angles.

    Returns:
        2-element tuple containing

        - **om1** (*array*): AM waveform.
        - **dom** (*array*): FM waveform (radians/s).

    References:
        Grissom, W., Cao, Z., & Does, M. (2014).
        :math:`B_1^{+}`-selective excitation pulse design using the Shinnar-Le
        Roux algorithm. Journal of Magnetic Resonance, 242, 189-196.
    """

    # calculate beta filter ripple
    [_, d1, d2] = slr.calc_ripples(ptype, d1, d2)
    bsf = flip

    # calculate pulse duration
    b = 4257 * pbw
    pulse_len = tb / b

    # calculate number of samples in pulse
    n = int(np.ceil(pulse_len / dt / 2) * 2)

    # modulation frequency to center profile at pbc gauss
    om = 2 * np.pi * 4257 * pbc
    t = np.arange(0, n) * pulse_len / n - pulse_len / 2

    om1 = np.zeros((2 * n, g))
    dom = np.zeros((2 * n, g))
    for gind in range(1, g + 1):
        # design filter
        h = bsf * slr.dz_hadamard_b(n, g, gind, tb, d1, d2, n // 4)

        # modulate filter to center and add it to a time-reversed and modulated
        # copy, then take the imaginary part to get an odd filter
        h = np.imag(h * np.exp(1j * om * t) - h[n::-1] * np.exp(1j * -om * t))
        if split_and_reflect:
            # split and flip fm waveform to improve large-tip accuracy
            dom[:, gind - 1] = (
                np.concatenate((h[n // 2 :: -1], h, h[n : n // 2 : -1])) / 2
            )
        else:
            dom[:, gind - 1] = np.concatenate(
                (0 * h[n // 2 :: -1], h, 0 * h[n : n // 2 : -1])
            )
        # build am waveform
        om1[:, gind - 1] = np.concatenate(
            (-np.ones(n // 2), np.ones(n), -np.ones(n // 2))
        )

    # scale to target flip, convert to Hz
    dom = dom / (2 * np.pi * dt)

    return om1, dom
