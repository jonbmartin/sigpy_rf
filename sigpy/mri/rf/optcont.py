# -*- coding: utf-8 -*-
"""Optimal Control Pulse Design functions.
"""
from sigpy import backend
from sigpy.mri.rf import slr
import numpy as np

__all__ = ['optcont1d', 'blochsim', 'deriv']


def optcont1d(dthick, N, os, tb, stepsize=0.001, max_iters=1000, d1=0.01,
              d2=0.01, dt=4e-6, conv_tolerance=1e-5):
    r"""1D optimal control pulse designer

    Args:
        dthick: thickness of the slice (cm)
        N: number of points in pulse
        os: matrix scaling factor
        tb: time bandwidth product, unitless
        stepsize: optimization step size
        max_iters: max number of iterations
        d1: ripple level in passband
        d2: ripple level in stopband
        dt: dwell time (s)
        conv_tolerance: max change between iterations, convergence tolerance

    Returns:
        gamgdt: scaled gradient
        pulse: pulse of interest, complex RF waveform

    """

    # set mag of gamgdt according to tb + dthick
    gambar = 4257  # gamma/2/pi, Hz/g
    gmag = tb / (N * dt) / dthick / gambar

    # get spatial locations + gradient
    x = np.arange(0, N * os, 1) / N / os - 1 / 2
    gamgdt = 2 * np.pi * gambar * gmag * dt * np.ones(N)

    # set up target beta pattern
    d1 = np.sqrt(d1 / 2)  # Mxy -> beta ripple for ex pulse
    d2 = d2 / np.sqrt(2)
    dib = slr.dinf(d1, d2)
    ftwb = dib / tb
    # freq edges, normalized to 2*nyquist
    fb = np.asarray([0, (1 - ftwb) * (tb / 2),
                     (1 + ftwb) * (tb / 2), N / 2]) / N

    dpass = np.abs(x) < fb[1]  # passband mask
    dstop = np.abs(x) > fb[2]  # stopband mask
    wb = [1, d1 / d2]
    w = dpass + wb[1] / wb[0] * dstop  # 'points we care about' mask

    # target beta pattern
    db = np.sqrt(1 / 2) * dpass * np.exp(-1j / 2 * x * 2 * np.pi)

    pulse = np.zeros(N, dtype=complex)

    a = np.exp(1j / 2 * x / (gambar * dt * gmag) * np.sum(gamgdt))
    b = np.zeros(a.shape, dtype=complex)

    eb = b - db
    cost = np.zeros(max_iters + 1)
    cost[0] = np.real(np.sum(w * np.abs(eb) ** 2))

    for ii in range(0, max_iters, 1):
        # calculate search direction
        auxb = w * (b - db)
        drf = deriv(pulse, x / (gambar * dt * gmag), gamgdt, None,
                    auxb, a, b)
        drf = 1j * np.imag(drf)

        # get test point
        pulse -= stepsize * drf

        # simulate test point
        [a, b] = blochsim(pulse, x / (gambar * dt * gmag), gamgdt)

        # calculate cost
        eb = b - db
        cost[ii + 1] = np.sum(w * np.abs(eb) ** 2)

        # check cost with tolerance
        if (cost[ii] - cost[ii + 1]) / cost[ii] < conv_tolerance:
            break

    return gamgdt, pulse


def blochsim(rf, x, g):
    r"""1D RF pulse simulation, with simultaneous RF + gradient rotations.
    Assume x has inverse spatial units of g, and g has gamma*dt applied and
    assume x = [...,Ndim], g = [Ndim,Nt].

     Args:
         rf (array): rf waveform input.
         x (array): spatial locations.
         g (array): gradient waveform.

     Returns:
         array: SLR alpha parameter
         array: SLR beta parameter
    """

    device = backend.get_device(rf)
    xp = device.xp
    with device:
        a = xp.ones(xp.shape(x)[0], dtype=complex)
        b = xp.zeros(xp.shape(x)[0], dtype=complex)
        for mm in range(0, xp.size(rf), 1):  # loop over time
            # apply RF
            c = xp.cos(xp.abs(rf[mm]) / 2)
            s = 1j * xp.exp(1j * xp.angle(rf[mm])) * xp.sin(xp.abs(rf[mm]) / 2)
            at = a * c - b * xp.conj(s)
            bt = a * s + b * c
            a = at
            b = bt

            # apply gradient
            if g.ndim > 1:
                z = xp.exp(-1j * x @ g[mm, :])
            else:
                z = xp.exp(-1j * x * g[mm])
            b = b * z

        # apply total phase accrual
        if g.ndim > 1:
            z = xp.exp(1j / 2 * x @ xp.sum(g, 0))
        else:
            z = xp.exp(1j / 2 * x * xp.sum(g))
        a = a * z
        b = b * z

        return a, b
    

def blochsim_amfm(am, om, x, g):
    r"""1D RF pulse simulation, with interleaved RF AM + (gradient + RF FM) rotations.
    Assume x has inverse spatial units of g, and g has gamma*dt applied and
    assume x = [...,Ndim], g = [Ndim,Nt]. 

     Args:
         am (array): am waveform input (radians).
         om (array): omega waveform input (radians).
         x (array): spatial locations.
         g (array): gradient waveform.

     Returns:
         array: SLR alpha parameter
         array: SLR beta parameter
    """

    device = backend.get_device(am)
    xp = device.xp
    with device:
        a = xp.ones(xp.shape(x)[0], dtype=complex)
        b = xp.zeros(xp.shape(x)[0], dtype=complex)
        for mm in range(0, xp.size(am), 1):  # loop over time
            # apply RF AM
            c = xp.cos(am[mm] / 2)
            s = 1j * xp.sin(am[mm] / 2)
            at = a * c - b * xp.conj(s)
            bt = a * s + b * c
            a = at
            b = bt

            # apply RF FM and gradient
            if g.ndim > 1:
                z = xp.exp(-1j * x @ g[mm, :] - 1j * om[mm])
            else:
                z = xp.exp(-1j * x * g[mm] - 1j * om[mm])
            b = b * z

        # apply total phase accrual
        if g.ndim > 1:
            z = xp.exp(1j / 2 * x @ xp.sum(g, 0) + 1j / 2 * om.sum())
        else:
            z = xp.exp(1j / 2 * x * xp.sum(g) + 1j / 2 * om.sum())
        a = a * z
        b = b * z

        return a, b


def deriv(rf, x, g, auxa, auxb, af, bf):
    r"""1D RF pulse simulation, with simultaneous RF + gradient rotations.

    'rf', 'g', and 'x' should have consistent units.

     Args:
         rf (array): rf waveform input.
         x (array): spatial locations.
         g (array): gradient waveform.
         auxa (None or array): auxa
         auxb (array): auxb
         af (array): forward sim a.
         bf( array): forward sim b.

     Returns:
         array: SLR alpha parameter
         array: SLR beta parameter
    """

    device = backend.get_device(rf)
    xp = device.xp
    with device:
        drf = xp.zeros(xp.shape(rf), dtype=complex)
        ar = xp.ones(xp.shape(af), dtype=complex)
        br = xp.zeros(xp.shape(bf), dtype=complex)

        for mm in range(xp.size(rf) - 1, -1, -1):
            # calculate gradient blip phase
            if g.ndim > 1:
                z = xp.exp(1j / 2 * x @ g[mm, :])
            else:
                z = xp.exp(1j / 2 * x * g[mm])

            # strip off gradient blip from forward sim
            af = af * xp.conj(z)
            bf = bf * z

            # add gradient blip to backward sim
            ar = ar * z
            br = br * z

            # strip off the curent rf rotation from forward sim
            c = xp.cos(xp.abs(rf[mm]) / 2)
            s = 1j * xp.exp(1j * xp.angle(rf[mm])) * xp.sin(xp.abs(rf[mm]) / 2)
            at = af * c + bf * xp.conj(s)
            bt = -af * s + bf * c
            af = at
            bf = bt

            # calculate derivatives wrt rf[mm]
            db1 = xp.conj(1j / 2 * br * bf) * auxb
            db2 = xp.conj(1j / 2 * af) * ar * auxb
            drf[mm] = xp.sum(db2 + xp.conj(db1))
            if auxa is not None:
                da1 = xp.conj(1j / 2 * bf * ar) * auxa
                da2 = 1j / 2 * xp.conj(af) * br * auxa
                drf[mm] += xp.sum(da2 + xp.conj(da1))

            # add current rf rotation to backward sim
            art = ar * c - xp.conj(br) * s
            brt = br * c + xp.conj(ar) * s
            ar = art
            br = brt

        return drf


def deriv_amfm(am, om, x, g, auxa, auxb, af, bf):
    r"""1D RF pulse simulation, with simultaneous RF + gradient rotations.

    'rf', 'g', and 'x' should have consistent units.

     Args:
         am (array): am waveform input (radians).
         om (array): omega waveform input (radians).
         x (array): spatial locations.
         g (array): gradient waveform.
         auxa (None or array): auxa
         auxb (array): auxb
         af (array): forward sim a.
         bf( array): forward sim b.

     Returns:
         array: SLR alpha parameter
         array: SLR beta parameter
    """

    device = backend.get_device(am)
    xp = device.xp
    with device:
        drf = xp.zeros(xp.shape(am), dtype=complex)
        ar = xp.ones(xp.shape(af), dtype=complex)
        br = xp.zeros(xp.shape(bf), dtype=complex)

        for mm in range(xp.size(am) - 1, -1, -1):
            # calculate gradient blip phase
            if g.ndim > 1:
                z = xp.exp(1j / 2 * x @ g[mm, :])
            else:
                z = xp.exp(1j / 2 * x * g[mm])

            # strip off gradient blip from forward sim
            af = af * xp.conj(z)
            bf = bf * z

            # add gradient blip to backward sim
            ar = ar * z
            br = br * z

            # strip off the curent rf rotation from forward sim
            c = xp.cos(am[mm] / 2)
            s = 1j * xp.sin(am[mm] / 2)
            at = af * c + bf * xp.conj(s)
            bt = -af * s + bf * c
            af = at
            bf = bt

            # calculate derivatives wrt rf[mm]
            db1 = xp.conj(1j / 2 * br * bf) * auxb
            db2 = xp.conj(1j / 2 * af) * ar * auxb
            drf[mm] = xp.sum(db2 + xp.conj(db1))
            if auxa is not None:
                da1 = xp.conj(1j / 2 * bf * ar) * auxa
                da2 = 1j / 2 * xp.conj(af) * br * auxa
                drf[mm] += xp.sum(da2 + xp.conj(da1))

            # add current rf rotation to backward sim
            art = ar * c - xp.conj(br) * s
            brt = br * c + xp.conj(ar) * s
            ar = art
            br = brt

        return drf