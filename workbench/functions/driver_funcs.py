import matplotlib
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import warnings
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import numpy.polynomial.polynomial as poly
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

def Fit_Sine(tt, yy):

    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    #guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_freq = 0.086
    guess_amp = np.std(yy) * 2.**0.5
    guess_phase = 0.
    guess = np.array([guess_amp,  2.*np.pi*guess_freq, guess_phase])

    def sinfunc(t, A, w, p):  return A *  np.sin(w*t + p)

    try:
        popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p = popt
        f = w/(2.*np.pi)
        fitfunc = lambda t: A  * np.sin(w*t + p)

        res = {"amp": A, "omega": w, "phase": p, "freq": f,
            "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

        x_fit = np.arange(0, len(tt))
        fit = res["fitfunc"](x_fit)
    except RuntimeError:
        fit = np.arange(0, len(tt))

    return fit

def Oscillation_STL(df_STL):
    df_STL = df_STL[df_STL.index.year <= 2021]
    years = np.unique(df_STL.index.year.values)
    df_STL_year = [df_STL[df_STL.index.year == year_] for year_ in years]

    maxima, minima = [], []
    for df_STL_y in df_STL_year:
        y = df_STL_y.values

        max_ind = np.argmax(y)
        min_ind = np.argmin(y)

        maxima_y = y[max_ind]; maxima.append(maxima_y)

        # for local minima
        minima_y = y[min_ind]; minima.append(minima_y)

    mean_ampl = np.mean(maxima) - np.mean(minima)

    df_STL = df_STL.resample('D').interpolate(method= 'linear')

    y2, T2 = df_STL.values, df_STL.index

    cross = y2[y2 != 0]
    zero_cross = np.where(np.diff(np.sign(cross)))[0]

    y2, T2 = y2[zero_cross], T2[zero_cross]

    periods = []
    for i in range(0, len(T2)-2, 2):
        p = (T2[i+2] - T2[i]).days
        periods.append(p)

    return np.mean(periods), mean_ampl

def convert_to_datetime(d):
    return datetime.strptime(np.datetime_as_string(d,unit='s'), '%Y-%m-%dT%H:%M:%S')

def identify_shorelinejump(stl_trend, df_filled, df_empty, driver, limit= 1.5):

    possibilities = ['reclamation', 'nourishment', 'storm']
    if driver not in possibilities:
        raise warnings.warn(f'The driver argument should be one of {possibilities}, not {driver}')
    
    lowess = sm.nonparametric.lowess


    stl_trend_diff = stl_trend.diff()

    cross = stl_trend_diff[stl_trend_diff != 0]
    zero_cross = list(np.where(np.diff(np.sign(cross)))[0])
    zero_cross.append(-1)

    indices = []
    change_lst = []
    change_x = []

    for i in range(1, len(zero_cross)-1):
        change = np.trapz(stl_trend_diff.iloc[zero_cross[i]: zero_cross[i+1]])
        v1, v2 = stl_trend_diff.index[zero_cross[i]], stl_trend_diff.index[zero_cross[i+1]]
        change_x.append(v1 + (v2 - v1) / 2)
        change_lst.append(change)

    meanchange, stdchange = np.mean(abs(np.array(change_lst))), np.std(abs(np.array(change_lst)))

    if driver == 'reclamation':
        indices = [i for i, ch in enumerate(change_lst) if abs(ch) >= meanchange + limit * stdchange]
    elif driver == 'nourishment':
        indices = [i for i, ch in enumerate(change_lst) if ch >= meanchange + limit * stdchange]
    elif driver == 'storm':
        indices = [i for i, ch in enumerate(change_lst) if ch <= -meanchange - limit * stdchange]

    change_vals = np.array(change_lst)[indices]
    indices = [[convert_to_datetime(stl_trend_diff.index.values[zero_cross[ind+1]]),\
                convert_to_datetime(stl_trend_diff.index.values[zero_cross[ind + 2]])] for ind in indices]
    indices.append([df_filled.index[-1]])
    dt = [(indices[i+1][0] - indices[i][-1]).days/365 for i in range(len(indices)-1)]

    trend1_lst, trend_03_lst, cr_lst, r2_lst = [], [], [], []
    for i in range(len(indices)-1):
        df_nmi = stl_trend[(stl_trend.index >= indices[i][1]) & (stl_trend.index <= indices[i+1][0])]
        z = lowess(endog= df_nmi, exog= np.arange(0, len(df_nmi.index)), frac= 1)
        trend_1 = pd.Series(z[:, 1], index= df_nmi.index)
        trend1_lst.append(trend_1)

        cr = (np.trapz(trend_1.diff().dropna())) / ((trend_1.index[-1] - trend_1.index[0]).days / 365)
        cr_lst.append(cr)

        z = lowess(endog= df_nmi, exog= np.arange(0, len(df_nmi.index)), frac=0.3)
        trend_03_lst.append(df_nmi)

        r2i = r2_score(df_nmi, trend_1)
        r2_lst.append(r2i)

    if driver == 'reclamation':
        indices = indices[:-1]
        if len(indices) > 0:
            right_date = indices[-1][-1]
        else:
            right_date = df_filled.index[-1]
        right_trend = stl_trend[stl_trend.index >= right_date]
        left_trend = stl_trend[stl_trend.index < right_date]

        change_smooth_after = abs(np.trapz(right_trend.diff().dropna()) / (len(right_trend) / 12))
        change_smooth_after_abs = abs(np.trapz(abs(right_trend.diff().dropna()))) / (len(right_trend) / 12)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize= (15, 8), sharex= True)

        filled_pos = df_filled[pd.isna(df_empty)]
        ax1.plot(df_empty.index, df_empty.values, color= 'k', marker= 'o', alpha= 0.4, linestyle= 'None', label= 'shoreline positions')
        ax1.plot(filled_pos.index, filled_pos.values, color= 'k', marker= 'x', markersize = 10, alpha= 0.6, linestyle= 'None', label= 'filled positions')
        #ax2.plot(trend_smooth.index, trend_smooth.values, color = 'r', linestyle = '--')
        if len(left_trend) > 0:
            left_trend.plot(label= 'left trend', ax= ax1, color= 'orange', linewidth= 3)
        if len(right_trend) > 0:
            right_trend.plot(label= 'right trend', ax= ax1, color= 'g', linewidth= 3)

        ax1.axvline(right_date, color= 'r', linestyle= '--', linewidth= 3, label= 'construction date')
        ax1.set_ylabel('Shoreline Position [m]')
        ax1.legend(loc= 'best')
        ax1.grid()


        ax2.stem(stl_trend_diff.index, stl_trend_diff, markerfmt= ' ',label= 'Difference')
        ax2.stem(change_x, change_lst, linefmt = 'r', markerfmt= ' ', label= 'dY$_i$')
        ax2.axhline(meanchange + limit * stdchange, color = 'r', linestyle = '--', label = f'$\mu_{{dY}}$ $\pm$ {limit} $\cdot \sigma_{{dY}}$')
        ax2.axhline(-meanchange - limit * stdchange, color = 'r', linestyle = '--')

        handles, labels = ax2.get_legend_handles_labels()
        order = [1, 2 , 0]
        ax2.set_ylabel('dY [m]')
        ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc= 'upper right')
        ax2.grid()
        ax2.set_xlabel('Time [years]')
        
        if len(indices) == 0:
            warnings.warn(f'No shorelinejump is found with a limit of {limit}. Perhaphs changing the limit might identify a reclamation.')
            params = {'T': None, 'Mcr': None, 'Tfixed': None, 'dYjump': None, 'dYtot': None,
                        'dYafter': None, 'absdYafter': None, 'dYdt_jump': None, 'dYdt_jumpcorrect': None}
            return params
        else:
            changeright = np.trapz(abs(right_trend.diff().dropna())) / len(right_trend)
            changeleft = np.trapz(abs(left_trend.diff().dropna())) / len(left_trend)

            if right_date == df_filled.index[-1] or right_date == df_filled.index[0]:
                warnings.warn(f'No shorelinejump is found with a limit of {limit}. Perhaphs changing the limit might identify a reclamation.')
                params = {'T': None, 'Mcr': None, 'Tfixed': None, 'dYjump': None, 
                            'dYafter': None, 'absdYafter': None, 'dYdt_jump': None, 'dYdt_jumpcorrect': None}
                return params
            else:
                dYdt_jump1 = change_vals[-1]
                if len(indices) > 1 and dYdt_jump1 >= 0:
                    if (indices[-1][0] - indices[-2][-1]).days / 365 <= 4:
                        dYdt_jump2 = change_vals[-2]
                    else:
                        dYdt_jump2 = 0
                else:
                    dYdt_jump2 = 0
                if dYdt_jump2 <= 0:
                    dYdt_jumpcorrected = (dYdt_jump1 + dYdt_jump2) / ((indices[-1][-1] - indices[-1][0]).days / 365)
                else:
                    dYdt_jumpcorrected = dYdt_jump1 / ((indices[-1][-1] - indices[-1][0]).days / 365)
                if dYdt_jumpcorrected >= 0:
                    params = {'T': int(round((df_filled.index[-1] - right_date).days/365, 0)),
                            'Mcr': changeright/changeleft,
                            'Tfixed': right_date.year,
                            'dYjump': abs(change_vals[-1]),
                            'dYafter': change_smooth_after,
                            'absdYafter': change_smooth_after_abs,
                            'dYdt_jump': dYdt_jump1 / ((indices[-1][-1] - indices[-1][0]).days / 365),
                            'dYdt_jumpcorrect': dYdt_jumpcorrected}
                else:
                    warnings.warn(f'No shorelinejump is found with a limit of {limit}. Perhaphs changing the limit might identify a reclamation.')
                    params = {'T': None, 'Mcr': None, 'Tfixed': None, 'dYjump': None, 
                            'dYafter': None, 'absdYafter': None, 'dYdt_jump': None, 'dYdt_jumpcorrect': None}
                return params
            
    elif driver == 'nourishment':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize= (20, 10), sharex= True)

        filled_pos = df_filled[pd.isna(df_empty)]
        ax1.plot(df_empty.index, df_empty, color= 'k', marker= 'o', alpha= 0.4, linestyle= 'None', label= 'shoreline positions')
        ax1.plot(filled_pos.index, filled_pos.values, color= 'k', marker= 'x', markersize = 10, alpha= 0.6, linestyle= 'None', label= 'filled positions')

        recover_index  = []
        if len(indices) > 0:
            for i,trend in enumerate(trend1_lst):
                #ax2.plot(trend.index, trend.values, color= 'r', linewidth= 3, label = 'Smoothened Erosion')
                recover_index.extend(trend.index)

        stl_trend_noerosion = stl_trend[~stl_trend.index.isin(recover_index)]
        freq = pd.Timedelta(weeks = 20)
        no_erosion_split = np.split(stl_trend_noerosion, np.flatnonzero(stl_trend_noerosion.index.to_series().diff() > freq))

        for i, tsplit in enumerate(no_erosion_split):
            ax1.plot(tsplit.index, tsplit.values, label= 'trend' if i == 0 else None, color= 'g', linewidth= 3)

        if len(indices) > 0:
            for i,trend in enumerate(trend1_lst):
                #ax2.plot(trend2.index, trend2.values, color= 'r', linewidth= 3, label = 'Trend erosion' if i ==0 else None)
                ax1.plot(trend.index, trend.values, color= 'r', linewidth= 3, label = 'erosion' if i ==0 else None)

        ax1.set_ylabel('Shoreline Position [m]')
        ax1.legend()
        ax1.grid()

        ax2.stem(stl_trend_diff.index, stl_trend_diff, markerfmt= ' ', label= 'Difference')
        ax2.stem(change_x, change_lst, linefmt = 'r', markerfmt= ' ', label= 'dY$_i$');
        ax2.axhline(meanchange + limit * stdchange, color = 'r', linestyle = '--', label = f'$\mu_{{diff}}$ + {limit} $\cdot \sigma_{{diff}}$')
        handles, labels = ax2.get_legend_handles_labels()
        order = [1, 2 , 0]
        ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
        ax2.grid()
        ax2.set_xlabel('Time [years]')

        if len(indices) == 0:
            warnings.warn(f'No shorelinejump is found with a limit of {limit}. Perhaphs changing the limit might identify a nourishment.')
            return [{'Date' : None,
                        'dY': None,
                        'LT': None,
                        'cr': None,
                        'r2': None}]
        else:
            years = []
            for i in range(len(indices)-1):
                start = (indices[i][0] + (indices[i][-1] - indices[i][0])/2).year
                end = (indices[i+1][0] + (indices[i+1][-1] - indices[i+1][0])/2).year
                years.append(f'{start}-{end}')
            start = (indices[-1][0] + (indices[-1][-1] - indices[-1][0])/2).year
            end = df_empty.index[-1].year
            years.append(f'{start}-{end}')
            return  [{'Date' : (date[0] + (date[1] - date[0]) / 2).strftime('%Y-%m-%d'),
                        'dY': change_vals[i],
                        'LT': dt[i] if not pd.isna(dt[i]) else 0,
                        'cr': cr_lst[i] if not pd.isna(cr_lst[i]) else 0,
                        'r2': r2_lst[i] if not pd.isna(r2_lst[i]) else 0 } for i, date in enumerate(indices[:-1])]

def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance in kilometers between two points
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
        return c * r
    
def identify_structure(transects, lons, lats,  dist_covered, plot):

    dist_transects = []
    for j in range(len(transects)-1):
        d= haversine(lons[0], lats[0], lons[1], lats[1])
        dist_transects.append(d)
    idx = 0
    x = np.cumsum([0] + dist_transects)
    coefs = poly.polyfit(x, dist_covered, 1)
    ys_sl = poly.polyval(x, coefs)
    r2 = [r2_score(dist_covered, ys_sl)]

    sameslope = False

    diffs = [abs((dist_covered[i+1] - dist_covered[i])) for i in range(len(dist_covered)-1)]
    idx_diff = np.argmax(diffs)

    idx_dist = np.argmax(dist_transects)
    state1, state2 = (diffs[idx_diff] > 2 * np.mean(diffs) and r2[0] < 0.5), (round(dist_transects[idx_dist], 1) >= 1.5 * np.mean(dist_transects) and r2[0] < 0.5)
    if state1 or state2 :
        if state1: idx = idx_diff
        else: idx = idx_dist
        idx1, idx2= idx+1, idx+1
        if abs(dist_covered[idx+1]) < 1:
            idx2 += 1
        x1, x2 = x[:idx1], x[idx2:]
        dist_covered1, dist_covered2 = dist_covered[:idx1], dist_covered[idx2:]
        coefs1 = poly.polyfit(x1, dist_covered1, 1)
        ys_sl1 = poly.polyval(x1, coefs1)
        r2_1 = r2_score(dist_covered1, ys_sl1)
        if len(x2) > 1:
            coefs2 = poly.polyfit(x2, dist_covered2, 1)
            ys_sl2 = poly.polyval(x2, coefs2)
            r2_2 = r2_score(dist_covered2, ys_sl2)
            a1, a2 = coefs1[1], coefs2[-1]
            if a1/a2 > 0:
                sameslope = True
        else:
            r2_2 = 0
        r2 = [r2_1, r2_2]
        idx = [idx1, idx2]
        if plot:
            #x = x[::-1]
            fig, ax = plt.subplots(figsize= (20, 7.5))
            ax.axvline((x[idx1-1]+x[idx1])/2, linestyle='--', color= 'r')
            ax.plot(x, dist_covered, marker= 'o', markersize = 5, color = 'k', linestyle= 'None')
            ax.plot(x1, ys_sl1, marker= 'o', markersize = 2, label= f' lin. fit 1 (r$_2$ = {round(r2_1, 2)})')
            if len(x2) > 1:
                ax.plot(x2, ys_sl2, marker= 'o', label= f' lin. fit 2 (r$_2$ = {round(r2_2, 2)})')
            ax.set_ylabel('Changerate [m/yr]')
            ax.set_xticks(x)
            ax.set_xticklabels([tr for tr in transects], rotation = 60)
            ax.legend()
            ax.grid()

    else:
        if plot:
            dist_covered = np.array(dist_covered)
            fig, ax = plt.subplots(figsize= (20, 7.5))
            ax.plot(x, dist_covered, marker= 'o', markersize = 5, color = 'k', linestyle= 'None')
            ax.plot(x, ys_sl,marker= 'o', markersize = 2, label= f' linf. fit (r$_2$ = {round(r2[0], 2)})')
            ax.set_ylabel('Changerate [m/yr]')
            ax.set_xticks(x)
            ax.set_xticklabels([tr for tr in transects], rotation = 60)
            ax.legend()
            ax.grid()
 
    return r2, idx, dist_transects, sameslope

def split_characteristics(trend, lim, max_seg= 5, plot= False):
    more_plot = False

    xs = np.arange(0, len(trend))
    ys = trend.values

    # Start with n_seg = 1
    coefs = poly.polyfit(xs, ys, 1)
    ys_sl = poly.polyval(xs, coefs)
    #ne = [norm_eucl(pd.Series(ys), pd.Series(ys_sl))]
    ne = [r2_score(ys, ys_sl)]

    #ENTER THE WHILE LOOP WITH N_SEG = 2
    split_years = trend.index[0]
    n_seg = 2
    msk_l = [[0, -1]]
    while any(i <= lim for i in ne) and n_seg <= max_seg:
        dys = np.gradient(ys, xs)

        rgr = DecisionTreeRegressor(max_leaf_nodes=n_seg)
        rgr.fit(xs.reshape(-1, 1), dys.reshape(-1, 1))
        dys_dt = rgr.predict(xs.reshape(-1, 1)).flatten()

        ys_sl = np.ones(len(xs)) * np.nan

        split_years = []
        ne = []
        msk_l = []
        for y in np.unique(dys_dt):
            more_plot = True
            msk = dys_dt == y
            lin_reg = LinearRegression()
            lin_reg.fit(xs[msk].reshape(-1, 1), ys[msk].reshape(-1, 1))
            ys_sl[msk] = lin_reg.predict(xs[msk].reshape(-1, 1)).flatten()
            msk_l.append(msk)
            split_years.append(trend.index[xs[msk][-1]] + pd.offsets.DateOffset(years=1))

            #n = norm_eucl(pd.Series(ys[msk]), pd.Series(ys_sl[msk]))
            n = r2_score(ys[msk], ys_sl[msk])
            ne.append(n)
        n_seg += 1

    split_years = pd.Series(split_years).sort_values().reset_index(drop= True)

    idx = [0]
    for i in range(len(split_years)-1):
        if int((split_years[i+1] - split_years[i]).days/365) != 1 and 1990 < split_years[i+1].year < 2018:
            idx.append(i+1)
    if len(split_years) > 1:
        split_years = split_years[idx]

    if plot:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize= (20, 7))
        if more_plot:
            ax0.set_title('Slope')
            ax0.scatter(xs, dys, label='data')
            ax0.scatter(xs, dys_dt, label='DecisionTree', s=2**2)
            ax0.legend()

            for msk in msk_l:
                ax1.plot([xs[msk][0], xs[msk][-1]],
                                 [ys_sl[msk][0], ys_sl[msk][-1]],
                                 color='r', zorder=1)
        else:
            ax1.plot(xs, ys_sl, color='r', zorder=1)

        ax1.set_title('Values')
        ax1.scatter(xs, ys, label='data')
        ax1.scatter(xs, ys_sl, s=3**2, label= f'seg lin reg', color='g', zorder=5)
        ax1.legend()

        plt.show()

    return sorted(split_years)

def merge_characteristics(trend, lim, stable_reg, plot= False):

    #Split on Characteristics
    split_years = split_characteristics(trend= trend, plot= plot, lim= lim)
    split_years.insert(0, trend.index[0])
    split_years.append(trend.index[-1] + pd.offsets.DateOffset(years= 1))
    split_years = np.unique(split_years)

    # #Fit on splits
    # fit_splits = fit_on_splits(df= trend, split_years= split_years)

    #Split to see changing changerates
    trend_split = [trend[(trend.index >= split_years[i]) & (trend.index < split_years[i+1])] for i in range(len(split_years)-1)]

    dates = []
    cr_values = []
    trend_split2 = []
    for ts in trend_split:
        ys = ts.values
        xs = np.arange(0, len(ts))
        coefs = poly.polyfit(xs, ys, 1)
        ys_sl = poly.polyval(xs, coefs)
        start, end = min(ts.index.strftime('%Y-%m')), max(ts.index.strftime('%Y-%m'))
        coastline_change = (ys_sl[-1] - ys_sl[0]) / ((ts.index[-1] - ts.index[0]).days / 365)
        dates.append(f'start = {start}, end = {end}')
        cr_values.append(coastline_change)

        ts_spl = pd.DataFrame(ys_sl, index = ts.index)
        trend_split2.append(ts_spl)

    nonstable_idx = [i for i, v in enumerate(cr_values) if abs(v) >= stable_reg]
    merged_dates = [dates[0]]
    k = 0
    if len(nonstable_idx) > 0:
        start_idx = nonstable_idx[0]
        for i in range(start_idx+1, len(cr_values)):
            start_cr = cr_values[start_idx]
            next_cr = cr_values[i]
            if abs(next_cr) >= stable_reg:
                if start_cr/next_cr > 0:
                    continue
                else:
                    start = merged_dates[k].split(',')[0]
                    end = dates[i].split(',')[0]
                    merged_dates[k] = f'{start}, {end}'
                    merged_dates.append(dates[i])
                    k +=1
                    start_idx = i
            else:
                continue

    start = merged_dates[-1].split(',')[0]
    end = dates[-1].split(',')[1]
    merged_dates[k] = f'{start}, {end}'

    merged_dates = [[pd.to_datetime(v.split(',')[0].split('=')[1]), pd.to_datetime(v.split(',')[1].split('=')[1])] for v in merged_dates]

    return merged_dates