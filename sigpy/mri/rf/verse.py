# -*- coding: utf-8 -*-
"""RF pulse and gradient variable-rate selective excitation (VERSE)

"""
import numpy as np
from scipy.signal import firls, filtfilt


def ss_b1verse(g, rf, b1max, gmax, smax, ts, gamma, slew_penalty=0, dbg=0):
    if len(g) == 1:
        g = np.array(g) * np.ones_like(rf)
    if len(b1max) == 1:
        b1max = np.array(b1max) * np.ones_like(rf)
    
    dt = ts * np.ones_like(rf)
    t_unif = (np.arange(len(rf)) + 0.5) * ts
    dt_unif = dt
    S_rfg = 1 / (2 * np.pi * gamma * ts)
    
    if dbg >= 2:
        rfg = rf * S_rfg
        print(f'Initial Max RF fraction: {np.max(np.abs(rfg) / b1max)}')
    
    T = np.sum(dt)
    notdone = True
    niter = 0
    maxiter = 100
    
    while niter < maxiter and notdone:
        niter += 1
        
        Imaxed = np.zeros_like(rf)
        Islewed = np.zeros_like(rf)
        Islewed[np.where(g == 0)] = 1
        Islewed[0] = 1
        Islewed[-1] = 1
        
        b = firls(8, [0, 0.03, 0.06, 1], [1, 1, 0, 0])
        rffilt = filtfilt(b, [1.0], np.abs(rf)) / (np.sum(b) ** 2)
        rffilt = np.maximum(np.abs(rf), rffilt)
        filtmax = np.maximum(np.max(np.abs(rffilt)), np.max(b1max) / S_rfg)
        s_mod = (1 - 0.999 * rffilt / filtmax) ** slew_penalty
        smax_mod = smax * 1e3 * s_mod
        
        for k in range(1, len(rf)):
            slew = (g[k] - g[k-1]) / (dt[k] + dt[k-1]) / 0.5
            smaxk = smax * 1e3 * (1 - (np.abs(rffilt[k]) + np.abs(rffilt[k-1])) * S_rfg / (b1max[k] + b1max[k-1])) ** slew_penalty
            if slew > smaxk:
                gh, gl = g[k], g[k-1]
                dth, dtl = dt[k], dt[k-1]
                a = smax_mod[k] * dth
                b = smax_mod[k] * dtl + 2 * gl
                c = -2 * gh
                scale = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                rf[k] /= scale
                g[k] /= scale
                dt[k] *= scale
                Islewed[k] = 1
        
        for k in range(len(rf) - 2, -1, -1):
            slew = (g[k] - g[k+1]) / (dt[k] + dt[k+1]) / 0.5
            smaxk = smax * 1e3 * (1 - (np.abs(rffilt[k]) + np.abs(rffilt[k+1])) * S_rfg / (b1max[k] + b1max[k+1])) ** slew_penalty
            if slew > smaxk:
                gh, gl = g[k], g[k+1]
                dth, dtl = dt[k], dt[k+1]
                a = smax_mod[k] * dth
                b = smax_mod[k] * dtl + 2 * gl
                c = -2 * gh
                scale = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                rf[k] /= scale
                g[k] /= scale
                dt[k] *= scale
                Islewed[k] = 1
        
        for k in range(len(rf)):
            rfscale = np.abs(rf[k]) * S_rfg / (b1max[k] * 0.999)
            if rfscale > 1:
                rf[k] /= rfscale
                g[k] /= rfscale
                dt[k] *= rfscale
                Imaxed[k] = 1
        
        Ifixed = np.where(Imaxed | Islewed)[0]
        Ishrink = np.where(~(Imaxed | Islewed))[0]
        shrink_scale = (T - np.sum(dt[Ifixed])) / np.sum(dt[Ishrink])
        
        if shrink_scale < 0:
            rfv = np.array([])
            gv = np.array([])
            return rfv, gv
        elif np.abs(shrink_scale - 1) < 1e-14 or (len(Imaxed) == 0 and len(Islewed) == 0):
            notdone = False
        else:
            rf[Ishrink] /= shrink_scale
            g[Ishrink] /= shrink_scale
            dt[Ishrink] *= shrink_scale
            t = np.cumsum(np.concatenate(([0], dt[:-1]))) + dt / 2
            rf = np.interp(t_unif, t, rf, left=0, right=0)
            g = np.interp(t_unif, t, g, left=0, right=0)
            dt = dt_unif
        
    gscale = np.abs(g) / gmax
    if np.max(gscale) > 1:
        rfv = np.array([])
        gv = np.array([])
        return rfv, gv
    
    rfv = rf
    gv = g
    if dbg >= 2:
        rfvg = rfv * S_rfg
        print(f'Final Max RF fraction: {np.max(np.abs(rfvg) / b1max)}')
    
    return rfv, gv
