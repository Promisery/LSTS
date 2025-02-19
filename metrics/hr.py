import numpy as np
import torch
import heartpy as hp
from scipy.signal import detrend, butter, filtfilt, periodogram
from scipy.interpolate import UnivariateSpline
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')


def fillna(arr):
    mean = np.nanmean(arr, axis=0)
    mask = np.where(np.isnan(arr))
    arr[mask] = np.take(mean, mask[1])
    return arr

def format_check(signal):
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()
    elif not isinstance(signal, np.ndarray):
        signal = np.array(signal)
    if signal.ndim > 1:
        signal = signal.reshape(-1)
    return signal

def recover_from_diff(signal):
    return np.cumsum(signal)

def preprocess(signal, diff=False):
    signal = format_check(signal)
    if diff:
        signal = recover_from_diff(signal)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    signal = detrend(signal)
    return signal

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def calculate_hr_and_hrv(signal, diff=False, fs=30, bpmmin=40, bpmmax=180):
    signal = preprocess(signal, diff)
    freq_min = bpmmin / 60
    freq_max = bpmmax / 60
    
    [b, a] = butter(3, [freq_min / fs * 2, freq_max / fs * 2], btype='bandpass')
    filtered = filtfilt(b, a, signal)
    ppg_signal = np.expand_dims(filtered, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= freq_min) & (f_ppg <= freq_max))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    bpm = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    
    measures = np.full(13, np.nan)
    measures[0] = bpm
    
    try:    
        filtered = hp.filter_signal(signal, cutoff=[0.5, 8], sample_rate=fs, order=3, filtertype='bandpass')
        N = len(filtered)
        L = int(np.ceil(N / 2) - 1)

        # Step 1: calculate local maxima and local minima scalograms

        # - initialise LMS matrices
        m_max = np.full((L, N), False)

        # - populate LMS matrices
        for k in range(1, L):  # scalogram scales
            for i in range(k + 2, N - k + 1):
                if filtered[i - 1] > filtered[i - k - 1] and filtered[i - 1] > filtered[i + k - 1]:
                    m_max[k - 1, i - 1] = True

        # Step 2: find the scale with the most local maxima (or local minima)
        # - row-wise summation (i.e. sum each row)
        gamma_max = np.sum(m_max, axis=1)
        # the "axis=1" option makes it row-wise
        # - find scale with the most local maxima (or local minima)
        lambda_max = np.argmax(gamma_max)

        # Step 3: Use lambda to remove all elements of m for which k>lambda
        m_max = m_max[: (lambda_max + 1), :]

        # Step 4: Find peaks (and onsets)
        # - column-wise summation
        m_max_sum = np.sum(m_max == False, axis=0)
        peaks = np.asarray(np.where(m_max_sum == 0)).astype(int).reshape(-1)
        
        rri = np.diff(peaks) / fs * 1000
        rri_diff = np.abs(np.diff(rri))
        
        ibi = np.mean(rri)
        measures[1] = ibi
        sdnn = np.std(rri)
        measures[2] = sdnn
        sdsd = np.std(rri_diff)
        measures[3] = sdsd
        rmssd = np.sqrt(np.mean(rri_diff ** 2))
        measures[4] = rmssd
        nn20 = rri_diff[np.where(rri_diff > 20.0)]
        nn50 = rri_diff[np.where(rri_diff > 50.0)]
        try:
            pnn20 = float(len(nn20)) / float(len(rri_diff))
        except:
            pnn20 = np.nan
        measures[5] = pnn20
        try:
            pnn50 = float(len(nn50)) / float(len(rri_diff))
        except:
            pnn50 = np.nan
        measures[6] = pnn50
    except:
        pass
    
    try: 
        mean_rr = np.mean(rri)
        thirty_perc = 0.3 * mean_rr
        if thirty_perc <= 300:
            upper_threshold = mean_rr + 300
            lower_threshold = mean_rr - 300
        else:
            upper_threshold = mean_rr + thirty_perc
            lower_threshold = mean_rr - thirty_perc
        
        rem_idx = np.where((rri <= lower_threshold) | (rri >= upper_threshold))[0] + 1

        removed_beats = peaks[rem_idx]
        b_peaklist = np.asarray([0 if x in removed_beats else 1 for x in peaks])
        
        rr_mask = [0 if (b_peaklist[i] + b_peaklist[i+1] == 2) else 1 for i in range(len(rri))]
        
        x_plus = []
        x_minus = []

        for i in range(len(rr_mask) - 1):
            if rr_mask[i] + rr_mask[i + 1] == 0:
                #only add adjacent RR-intervals that are not rejected
                x_plus.append(rri[i])
                x_minus.append(rri[i + 1])
            else:
                pass

        #cast to arrays so we can do numerical work easily
        x_plus = np.asarray(x_plus)
        x_minus = np.asarray(x_minus)

        #compute parameters and append to dict
        x_one = (x_plus - x_minus) / np.sqrt(2)
        x_two = (x_plus + x_minus) / np.sqrt(2)
        sd1 = np.sqrt(np.var(x_one)) #compute stdev perpendicular to identity line
        sd2 = np.sqrt(np.var(x_two)) #compute stdev parallel to identity line
        s = np.pi * sd1 * sd2 #compute area of ellipse
        measures[7] = sd1
        measures[8] = sd2
        measures[9] = sd1/sd2
    except:
        pass
    
    try:
        degree_smoothing_spline = 3
        rr_x = np.cumsum(rri)
        resamp_factor = 4
        datalen = int((len(rr_x) - 1)*resamp_factor)
        rr_x_new = np.linspace(int(rr_x[0]), int(rr_x[-1]), datalen)
        interpolation_func = UnivariateSpline(rr_x, rri, k=degree_smoothing_spline)
        rr_interp = interpolation_func(rr_x_new)
        dt = np.mean(rri) / 1000  # in sec
        fs = 1 / dt  # about 1.1 Hz; 50 BPM would be 0.83 Hz, just enough to get the
        # max of the HF band at 0.4 Hz according to Nyquist
        fs_new = fs * resamp_factor
        frq = np.fft.fftfreq(datalen, d=(1 / fs_new))
        frq = frq[range(int(datalen / 2))]
        Y = np.fft.fft(rr_interp) / datalen
        Y = Y[range(int(datalen / 2))]
        psd = np.power(Y, 2)
        df = frq[1] - frq[0]
        lf = np.trapz(abs(psd[(frq >= 0.04) & (frq < 0.15)]), dx=df)
        hf = np.trapz(abs(psd[(frq >= 0.15) & (frq < 0.4)]), dx=df)
        lfhf = lf/hf

        nu_factor = 100 / (lf + hf)
        lf_nu = lf * nu_factor
        hf_nu = hf * nu_factor
        measures[10] = lf_nu
        measures[11] = hf_nu
        measures[12] = lfhf
    
    except:
        pass
    
    return measures


def calculate_hr_and_hrv_metrics(predictions, ground_truth, diff=False, fs=30, bpmmin=40, bpmmax=180):
    with Parallel(n_jobs=4) as parallel:
        hr_and_hrv_pred = parallel(delayed(calculate_hr_and_hrv)(pred, diff, fs, bpmmin, bpmmax) for pred in predictions)
        hr_and_hrv_gt = parallel(delayed(calculate_hr_and_hrv)(gt, False, fs, bpmmin, bpmmax) for gt in ground_truth)
        hr_and_hrv_pred, hr_and_hrv_gt = np.array(hr_and_hrv_pred), np.array(hr_and_hrv_gt)
    hr_and_hrv_pred = fillna(hr_and_hrv_pred)
    hr_and_hrv_gt = fillna(hr_and_hrv_gt)
    results = {}
    features = [
        'bpm',
        'ibi',
        'sdnn',
        'sdsd',
        'rmssd',
        'pnn20',
        'pnn50',
        'sd1',
        'sd2',
        'sd1sd2',
        'lf_nu',
        'hf_nu',
        'lfhf'
    ]
    for idx, feature in enumerate(features):
        pred = hr_and_hrv_pred[:, idx]
        gt = hr_and_hrv_gt[:, idx]
        mae = np.mean(np.abs(pred - gt))
        rmse = np.sqrt(np.mean(np.square(pred - gt)))
        mape = np.mean(np.abs((pred - gt) / (gt + 1e-8))) * 100
        pearson = np.corrcoef(pred, gt)
        results[feature + '/MAE'] = mae
        results[feature + '/MAPE'] = mape
        results[feature + '/RMSE'] = rmse
        results[feature + '/Pearson'] = pearson[0, 1]
        if feature == 'bpm':
            missrate = np.mean(np.abs(pred - gt) > 2)
            results[feature + '/MR'] = missrate
    return results