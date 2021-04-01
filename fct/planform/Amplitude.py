import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import j0
import click
from ..wavelet import wavelet as cwt
import pywt
import xarray as xr
from ..config import (
    DatasetParameter,
    LiteralParameter
)

class Parameters:

    planform = DatasetParameter(
        'Planform basic descriptors (shift, curvature, phi)',
        type='input')

    sample_distance = LiteralParameter(
        'distance between sampled talweg points')

    scale_min = LiteralParameter('minimum scale (longitudinal distance) used for scale averaging')
    scale_max = LiteralParameter('maximum scale (longitudinal distance) used for scale averaging')

    def __init__(self):

        self.planform = dict(key='planform_basic_metrics', refaxis='PLANAXIS')
        self.sample_distance = 20.0
        self.scale_min = 400.0
        self.scale_max = 10e3

def moving_average(data_set, periods=3):
    """
    https://gist.github.com/rday/5716218
    """
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='same')

def amplitude(signal, dt=20.0, scale_max=10e3):

    periods = np.int(scale_max / dt)

    return np.sqrt(
        # 2 * (
        moving_average(np.square(signal), periods)
        - np.square(moving_average(signal, periods))
        # )
    )

def scale_average_pywt(yt, scale_min=0.0, scale_max= 20e3, dt=20.0):

    N = yt.size

    variance = np.var(yt)
    mean = np.mean(yt)
    yt = (yt - mean) / np.sqrt(variance)

    wavelet = 'cmor1.5-1.0'
    Cdelta = 0.776 # this is for the MORLET wavelet
    scales = np.arange(2, 128)
    
    cfs, frequencies = pywt.cwt(yt, scales, wavelet, dt)
    power = (abs(cfs)) ** 2
    period = 1. / frequencies

    # global_ws = variance*power.sum(axis=1)/float(N)

    avg = (period >= scale_min) & (period < scale_max)
    scale_avg = np.dot(period.reshape(len(period), 1), np.ones((1, N)))
    scale_avg = power / scale_avg
    scale_avg = variance*dt/Cdelta*np.sum(scale_avg, axis=0)
    # scale_avg = dt/Cdelta*np.sum(scale_avg, axis=0)

    return scale_avg

def scale_average_wavelet(yt, scale_min=0.0, scale_max= 20e3, dt=20.0, dj=0.25):

    Cdelta = 0.776 #   % this is for the MORLET wavelet

    variance = np.var(yt)
    mean = np.mean(yt)
    yt = (yt - mean) / np.sqrt(variance)

    n = yt.size
    s0 = 2*dt
    wave, period, scale, coi = cwt(yt, dt, 1, dj, s0=s0, J1=-1, mother='MORLET')
    power = (np.abs(wave))**2

    avg = (scale >= scale_min) & (scale < scale_max)
    scale_avg = scale[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # expand scale --> (J+1)x(N) array
    scale_avg = power / scale_avg #   % [Eqn(24)]
    scale_avg = variance*dj*dt/Cdelta*sum(scale_avg[avg,:])

    return scale_avg

def Amplitude(source, params: Parameters) -> xr.Dataset:

    # planform = (
    #     xr.open_dataset(params.planform.filename(tileset=None))
    #     .set_index(sample=('axis', 'talweg_measure'))
    #     .load()
    # )

    planform = (
        xr.open_dataset(source.filename(tileset=None))
        .set_index(sample=('axis', 'talweg_measure'))
        .load()
    )

    dataset = list()

    with click.progressbar(np.unique(planform.axis)) as iterator:
        for axis in iterator:

            planformx = planform.sel(axis=axis).sortby('talweg_measure')
            size = planformx.talweg_measure.size
            
            shift = planformx.talweg_shift.values
            phi = planformx.talweg_direction_angle.values

            amp1 = amplitude(
                shift,
                dt=params.sample_distance,
                scale_max=0.5*params.scale_max)
            amp2 = np.sqrt(
                scale_average_wavelet(
                    shift,
                    dt=params.sample_distance,
                    scale_min=params.scale_min,
                    scale_max=params.scale_max))
            # amp3 = np.sqrt(scale_average_pywt(shift, dt=params.sample_distance))
            
            omega1 = amplitude(
                phi,
                dt=params.sample_distance,
                scale_max=0.5*params.scale_max)
            omega2 = np.sqrt(
                scale_average_wavelet(
                    phi,
                    dt=params.sample_distance,
                    scale_min=params.scale_min,
                    scale_max=params.scale_max))
            # omega3 = np.sqrt(scale_average_pywt(phi, dt=params.sample_distance))

            if amp1.size > size:

                amp1 = np.full(size, np.std(shift), dtype='float32')
                omega1 = np.full(size, np.std(phi), dtype='float32')

            values = xr.Dataset(
                {
                    'amplitude': ('sample', amp1),
                    'amplitude_scale_avg': ('sample', amp2),
                    # 'amplitude_scale_avg_pywt': ('sample', amp3),
                    'omega': ('sample', omega1),
                    'omega_scale_avg': ('sample', omega2),
                    # 'omega_scale_avg_pywt': ('sample', omega3)
                }, coords={
                    'axis': ('sample', np.full(size, axis, dtype='uint32')),
                    'talweg_measure': ('sample', planformx.talweg_measure.values)
                })

            dataset.append(values)

    return xr.concat(dataset, 'sample', 'all')

def InterpolateAlongMeasure(data, reference, dm=100.0):

    refaxis = (
        xr.open_dataset(reference.filename())
        .set_index(sample=('axis', 'talweg_measure'))
        .load()
    )

    data = (
        data
        .set_index(sample=('axis', 'talweg_measure'))
        .merge(refaxis)
    )

    dataset = list()

    with click.progressbar(np.unique(data.axis)) as iterator:
        for axis in iterator:

            datax = data.sel(axis=axis)
            measure_min = np.floor(np.min(datax.ref_measure) / dm) * dm
            measure_max = np.ceil(np.max(datax.ref_measure) / dm) * dm
            bins = np.arange(measure_min, measure_max + dm, dm)

            reindexed = (
                datax
                .set_index(talweg_measure='ref_measure')
                .rename(talweg_measure='measure')
            )

            grouped = (
                reindexed[['amplitude_scale_avg', 'omega_scale_avg']]
                .rename(
                    amplitude_scale_avg='amplitude',
                    omega_scale_avg='omega')
                .groupby_bins('measure', bins)
                .mean()
            )

            measures = np.array(
                [iv.mid for iv in grouped.measure_bins.values],
                dtype='float32')

            # missing = np.isnan(values)

            # if np.any(missing):

            #     values[missing] = np.interp(
            #         measures[missing],
            #         measures[~missing],
            #         values[~missing])

            size = grouped.measure_bins.size
            values = xr.Dataset(
                {
                    'amplitude': (('swath',), np.float32(grouped.amplitude.values)),
                    'omega': (('swath',), np.float32(grouped.omega.values))
                }, coords={
                    'axis': (('swath',), np.full(size, axis, dtype='uint32')),
                    'measure': (('swath',), measures)
                })

            dataset.append(values)

    return xr.concat(dataset, 'swath', 'all')

def InterpolateSwaths(data, reference, measures):

    refaxis = (
        xr.open_dataset(reference.filename())
        .set_index(sample=('axis', 'talweg_measure'))
        .load()
    )

    data = (
        data
        .set_index(sample=('axis', 'talweg_measure'))
        .merge(refaxis)
    )

    dataset = list()

    with click.progressbar(np.unique(data.axis)) as iterator:
        for axis in iterator:

            datax = data.sel(axis=axis).sortby('ref_measure')
            measx = measures.sel(axis=axis)

            size = measx.size
            ampl = np.interp(measx, datax.ref_measure, datax.amplitude_scale_avg)
            omega = np.interp(measx, datax.ref_measure, datax.omega_scale_avg)

            # missing = np.isnan(values)

            # if np.any(missing):

            #     values[missing] = np.interp(
            #         measures[missing],
            #         measures[~missing],
            #         values[~missing])

            values = xr.Dataset(
                {
                    'amplitude': (('swath',), np.float32(ampl)),
                    'omega': (('swath',), np.float32(omega))
                }, coords={
                    'axis': (('swath',), np.full(size, axis, dtype='uint32')),
                    'measure': (('swath',), np.float32(measx.values))
                })

            dataset.append(values)

    return xr.concat(dataset, 'swath', 'all')
