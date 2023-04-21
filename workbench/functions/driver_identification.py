import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from statsmodels.tsa.seasonal import STL
from scipy.signal import argrelextrema
import warnings
import numpy.polynomial.polynomial as poly
import statsmodels.api as sm

from .driver_funcs import Fit_Sine, Oscillation_STL, identify_shorelinejump, identify_structure, merge_characteristics

class identify_drivers():

    def __init__(self, ds):
        """
        Parameters:
            - ds (xr.ArrayDataset): dataset containing monthly shoreline positions resolution (ds_hr).

        Returns:
            None.
        """
        self.ds = ds


    def raw_timeseries(self, transect):
        """
        This function extracts the timeseries of shoreline positions for a given transect. Outliers are also removed from this
        timeseries.

        Parameters:
            - transects (str): transect of a hotspot in the ds dataset.

        Returns:
            - timeseries (pd.DataFrame): timeseries of the transect.
        """
        station = list(self.ds['transect_id'].values).index(str.encode(transect)) # select station
        outl_idx = [i for i,x in enumerate(self.ds.isel(stations= station)['outliers'].values) if x == 1]
        timeseries = pd.DataFrame({transect: self.ds.isel(stations= station)['sp'].values}, index = self.ds.time)
        timeseries[outl_idx] = np.nan

        return timeseries

    def process_timeseries(self, timeseries, transect, remove_nans = True):
        """
        Fill missing values using several ML algorithm.

        Parameters:
            - timeseries (pd.DataFrame): indexes are monthly datetimes. Can be retrieved via the raw_timeseries function.
            - transect (str): transect of a hotspot in the ds dataset. Corresponds to the transect in the timeseries dataframe.
            - remove_nans (bool): remove the NaN-values at the start and the end of the timeseries.

        Returns:
            - tuple: A tuple containing:
                - timeseries (pd.DataFrame): timeseries of the transect containing NaN-values.
                - timeseries_filled (pd.DataFrame): timeseries of the transect with no NaN-values.
        """

        datetime_index = list(timeseries.index.values)
        timeseries.index = np.arange(0, len(timeseries))

        if remove_nans:
        # drop NaN rows at the start
            while timeseries.iloc[0].isnull().all():
                timeseries = timeseries.drop(0).reset_index(drop= True)
                del datetime_index[0]

            # drop NaN rows at the end
            while timeseries.iloc[-1].isnull().all():
                timeseries = timeseries.drop(len(timeseries)-1).reset_index(drop= True)
                del datetime_index[-1]

        # Create a list of candidate models
        models = [
            LinearRegression(),
            SVR(kernel='linear'),
            RandomForestRegressor(n_estimators=100),
            GradientBoostingRegressor()
        ]

        
        # Split the data into training and validation sets
        train_data = timeseries.dropna()
        valid_indices = timeseries.index[~timeseries[transect].isna()]

        # Initialize best model and R-squared value
        best_model = None
        best_r2 = -np.inf

        # Fit each candidate model and calculate R-squared on validation set
        for model in models:
            # Fit the model to the training data
            X_train = train_data.index.values.reshape(-1, 1)
            y_train = train_data[transect]

            model.fit(X_train, y_train)

            # Use the model to predict the missing values in the validation set
            X_valid = valid_indices.values.reshape(-1, 1)
            y_valid_pred = model.predict(X_valid)

            # Calculate the R-squared metric for the model's predictions on the validation set
            y_valid_true = timeseries.loc[valid_indices, transect]
            r2 = r2_score(y_valid_true, y_valid_pred)

            # Save the model and its R-squared value if it's the best so far
            if r2 > best_r2:
                best_model = model
                best_r2 = r2

        # Use the selected model to predict the missing values in the original timeseries data
        empty_indices =  timeseries.index[timeseries[transect].isna()].values
        X_fill = empty_indices.reshape(-1, 1)
        y_fill = best_model.predict(X_fill)

        timeseries_filled = timeseries.copy()
        timeseries_filled.iloc[empty_indices] = y_fill.reshape(-1, 1)
        # Replace the missing values with the predicted values

        timeseries.index = datetime_index
        timeseries_filled.index = datetime_index

        return timeseries, timeseries_filled

    def stl_decompositions(self, timeseries_filled, period = 12, seasonal = 61, robust = True, seasonal_deg = 0):
        """
        Decompose a timeseries into a trend, seasonal and remainder component.

        Parameters:
            - timeseries (pd.DataFrame): timeseries of the transect with no NaN-values.
            - period (int): Periodicity of the sequence. If None and endog is a pandas Series or DataFrame, attempts to determine 
                          from endog. If endog is a ndarray, period must be provided.
            - seasonal (int): Length of the seasonal smoother. Must be an odd integer, and should normally be >= 7 (default).
            - robust (bool): Flag indicating whether to use a weighted version that is robust to some forms of outliers.
            - seasonal_deg (int): Degree of seasonal LOESS. 0 (constant) or 1 (constant and trend).

        Returns:
            - pd.DataFrame: containting the STL decomposed signals.
        """
        return STL(timeseries_filled, period= period, seasonal = seasonal, robust = robust, seasonal_deg = seasonal_deg)
    
    def get_hotspot(self, transect, period = 12, seasonal = 61, robust = True, seasonal_deg = 0):
        """
        Retrieve the entire hotspot of a certain transect.

        Parameters:
            - transect (str): transect of a hotspot in the ds dataset.
            - period (int): Periodicity of the sequence. If None and endog is a pandas Series or DataFrame, attempts to determine 
                          from endog. If endog is a ndarray, period must be provided.
            - seasonal (int): Length of the seasonal smoother. Must be an odd integer, and should normally be >= 7 (default).
            - robust (bool): Flag indicating whether to use a weighted version that is robust to some forms of outliers.
            - seasonal_deg (int): Degree of seasonal LOESS. 0 (constant) or 1 (constant and trend).

        Returns:
            tuple: containing four items:
                - df_hotspot (pd.DataFrame): timeseries dataframe with all the transects containing no NaN-values.
                - trend05_dict (dict): dictionary with the keys the transects and the values the smoothed timeseries in a pd.Series.
                - lons (lst): the longitutes of the transects.
                - lats (lst): the latitudes of the transects.
        """
        station = list(self.ds['transect_id'].values).index(str.encode(transect)) # select station
        hotspot_id = self.ds.isel(stations=station)['hotspot_id'].values
        ds_hotspot_idx = np.where(self.ds['hotspot_id'] == hotspot_id)[0]
        ds_hotspot = self.ds.isel(stations=ds_hotspot_idx)
        transects = [x.decode("utf-8") for x in ds_hotspot['transect_id'].values]
        sp = ds_hotspot['sp'].values
        dict_sp = dict(zip(transects, sp))

        lons = ds_hotspot['lon'].values
        lats = ds_hotspot['lat'].values

        df_hotspot = pd.DataFrame(dict_sp, index= self.ds.time)

        datetime_index = list(df_hotspot.index.values)
        df_hotspot.index = np.arange(0, len(df_hotspot))

        # drop NaN rows at the start
        while df_hotspot.iloc[0].isnull().all():
            df_hotspot = df_hotspot.drop(0).reset_index(drop= True)
            del datetime_index[0]

        # drop NaN rows at the end
        while df_hotspot.iloc[-1].isnull().all():
            df_hotspot = df_hotspot.drop(len(df_hotspot)-1).reset_index(drop= True)
            del datetime_index[-1]

        df_hotspot.index = datetime_index

        trend05_dict = {}
        for transect in df_hotspot.columns:
            timeseries, timeseries_filled = self.process_timeseries(timeseries = df_hotspot[transect].to_frame(), transect= transect, remove_nans= False)
            df_hotspot[transect] = timeseries_filled.values

            lowess = sm.nonparametric.lowess
            stl = self.stl_decompositions(df_hotspot[transect], period = period, seasonal = seasonal, robust = robust, seasonal_deg = seasonal_deg).fit()
            z = lowess(endog= stl.trend + stl.resid, exog= np.arange(0, len(stl.trend.index)), frac= 0.5)
            trend_05 = pd.Series(z[:, 1], index= df_hotspot[transect].index)
            trend05_dict[transect] = trend_05

        return df_hotspot, trend05_dict, lons, lats

    def seasonality(self, transect, period = 12, seasonal = 61, robust = True, seasonal_deg = 0):
        """
        Generate the parameters needed to identify seasonality at a certain transect. Also a plot is created to visualize the identification.
        ---
        Parameters:
            - transect (str): transect of a hotspot in the ds dataset.
            - period (int): Periodicity of the sequence. If None and endog is a pandas Series or DataFrame, attempts to determine 
                          from endog. If endog is a ndarray, period must be provided.
            - seasonal (int): Length of the seasonal smoother. Must be an odd integer, and should normally be >= 7 (default).
            - robust (bool): Flag indicating whether to use a weighted version that is robust to some forms of outliers.
            - seasonal_deg (int): Degree of seasonal LOESS. 0 (constant) or 1 (constant and trend).
        ---
        returns:
            - params (dict): dictionary containing the identification parameters.
        """

        raw_timeseries = self.raw_timeseries(transect = transect)
        timeseries, timeseries_filled = self.process_timeseries(timeseries= raw_timeseries, transect= transect)
        final_decade_empt = timeseries[timeseries.index >= pd.to_datetime('2013-03-01')]

        pmis_2013 = ((final_decade_empt.isna().sum() / len(final_decade_empt)) * 100).values[0]
        df_transectf = timeseries_filled
        # Decompose
        final_decade = df_transectf[df_transectf.index >= pd.to_datetime('2013-03-01')]
        years = np.unique(final_decade.index.year)
        
        if pmis_2013 <= 40 and len(years) >= 3:
            stl = self.stl_decompositions(final_decade, period= period, seasonal= seasonal, robust= robust, seasonal_deg = seasonal_deg).fit()
            df_stl = pd.concat([stl.seasonal, stl.trend, stl.resid], axis= 1)
            stl_seasonal = df_stl.season
            stl_trend = df_stl.trend


            # Metrics
            Ps_10, Ds_10 = Oscillation_STL(df_STL= stl_seasonal)
            FS_seas_10 = Fit_Sine(tt= np.arange(0, len(stl_seasonal)), yy= stl_seasonal)
            r2_sine_10 = round(r2_score(stl_seasonal, FS_seas_10), 2)
            Fit_series = pd.Series(FS_seas_10, index= stl_seasonal.index)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize= (15, 8), sharex= True)

            transect_10_f = df_transectf[df_transectf.index >=pd.to_datetime('2013-03-01')]
            transect_10_e = timeseries[timeseries.index >= pd.to_datetime('2013-03-01')]
            filled_pos = transect_10_f[pd.isna(transect_10_e)]

            #plot1
            ax1.plot(transect_10_e.index, transect_10_e.values, label= 'shoreline positions', color= 'k', marker = 'o', linestyle = 'None', alpha = 0.4)
            ax1.plot(filled_pos.index, filled_pos.values, label= 'filled positions', color= 'k', marker = 'x', markersize = 10, linestyle = 'None', alpha = 0.6)
            ax1.plot(stl_trend.index, (stl_trend + stl_seasonal).values, label= 'seasonal', color= 'g')
            ax1.plot(stl_trend.index, stl_trend.values, label= 'trend', color= 'r')
            handles, labels = ax1.get_legend_handles_labels()
            order = [0, 1, 2, 3]
            ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
            ax1.set_ylabel('Shoreline Position [m]')
            ax1.grid()

            #plot2
            ax2.plot(stl_seasonal.index, stl_seasonal.values, label= 'seasonal', color= 'g')
            ax2.plot(stl_seasonal.index, FS_seas_10,  label= 'sinusoidal fit', color= 'r', linestyle = '--')
            ax2.set_ylabel('Seasonal Shoreline Position [m]')
            ax2.set_xlabel('Time [years]')
            ax2.grid()

            #period of erosion vs accretion
            seas_10_daily = Fit_series.resample('D').interpolate()
            yrs = np.unique(seas_10_daily.index.year)

            seas_10_daily_2 = stl_seasonal.resample('D').interpolate()

            maxi, mini = [], []
            maxi2, mini2 = [], []
            for i, yr in enumerate(yrs):
                seas_yr = seas_10_daily[seas_10_daily.index.year == yr]
                maxi_y, mini_y = np.argmax(seas_yr), np.argmin(seas_yr)

                seas_yr2 = seas_10_daily_2[seas_10_daily_2.index.year == yr]
                maxi_y2, mini_y2 = argrelextrema(seas_yr2.values, np.greater)[0], argrelextrema(seas_yr2.values, np.less)[0]
                if len(maxi_y2) == 0: maxi_y2 = [0]
                if len(mini_y2) == 0: mini_y2 = [0]
                maxi_y2, mini_y2 = maxi_y2[np.argmax(seas_yr2.values[maxi_y2])], mini_y2[np.argmin(seas_yr2.values[mini_y2])]
                if seas_yr2[maxi_y2] < seas_yr2[0]: maxi_y2 = 0
                if seas_yr2[mini_y2] > seas_yr2[0]: mini_y2 = 0
    
                ax2.plot(seas_yr2.index[maxi_y2], seas_yr2.iloc[maxi_y2], color= 'k', marker= 'o', linestyle= 'None', label= 'seasonal extrema' if i == 0 else None)
                ax2.plot(seas_yr2.index[mini_y2], seas_yr2.iloc[mini_y2], color= 'k', marker= 'o', linestyle= 'None')

                maxi.append(seas_yr.index[maxi_y].month)
                mini.append(seas_yr.index[mini_y].month)

                maxi2.append(seas_yr2.index[maxi_y2].month)
                mini2.append(seas_yr2.index[mini_y2].month)
              
                ax2.axhline(0, color = 'grey', linestyle = '--', alpha = 0.5)
                ax2.legend()
                fig.tight_layout()

            maxi, mini = maxi[len(maxi)//2] , mini[len(mini)//2]
            maxi2, mini2 = maxi2[len(maxi2)//2] , mini2[len(mini2)//2]

            min_width_fit, max_width_fit = mini, maxi
            min_width_data, max_width_data = mini2, maxi2
            if mini2 > maxi2:
                Taccr = (pd.to_datetime(f'2017-{maxi2}-01') - pd.to_datetime(f'2016-{mini}-01')).days
            else:
                Taccr = (pd.to_datetime(f'2016-{maxi2}-01') - pd.to_datetime(f'2016-{mini}-01')).days


            params = {
                        'Ds': round(Ds_10, 0), 'r2': r2_sine_10, 't_max_seasonal_sp': max_width_data,
                        't_min_seasonal_sp': min_width_data, "t_accr": Taccr, 'perc_mis2013': round(pmis_2013, 2)
                        }
        else:
            warnings.warn("There are more than 40% missing values after 2013. Seasonality can not accurately be determined and identification is classified as unknown.")

            params = {
                        'Ds': None, 'r2': None, 't_max_seasonal_sp': None,
                        't_min_seasonal_sp': None, "t_accr": None, 'perc_mis2013': round(pmis_2013, 2)
                        }

        return params
    
    def reclamation(self, transect, period = 12, seasonal = 61, robust = True, seasonal_deg = 0, limit = 2):
        """
        Generate the parameters needed to identify a reclamation at a certain transect. Also a plot is created to visualize the identification.
        ---
        Parameters:
            - transect (str): transect of a hotspot in the ds dataset.
            - period (int): Periodicity of the sequence. If None and endog is a pandas Series or DataFrame, attempts to determine 
                          from endog. If endog is a ndarray, period must be provided.
            - seasonal (int): Length of the seasonal smoother. Must be an odd integer, and should normally be >= 7 (default).
            - robust (bool): Flag indicating whether to use a weighted version that is robust to some forms of outliers.
            - seasonal_deg (int): Degree of seasonal LOESS. 0 (constant) or 1 (constant and trend).
        ---
        Returns:
            - params (dict): dictionary containing the identification parameters.
        """
        raw_timeseries = self.raw_timeseries(transect = transect)
        timeseries, timeseries_filled = self.process_timeseries(timeseries= raw_timeseries, transect= transect)

        stl = self.stl_decompositions(timeseries_filled, period= period, seasonal= seasonal, robust= robust, seasonal_deg = seasonal_deg).fit()

        params = identify_shorelinejump(stl_trend = stl.trend, df_filled= timeseries_filled, df_empty= timeseries, driver= 'reclamation', limit= limit)

        return params
    
    def nourishment(self, transect, period = 12, seasonal = 61, robust = True, seasonal_deg = 0, limit = 2):

        """
        Generate the parameters needed to identify a nourishment at a certain transect. Also a plot is created to visualize the identification.
        ---
        Parameters:
            - transect (str): transect of a hotspot in the ds dataset.
            - period (int): Periodicity of the sequence. If None and endog is a pandas Series or DataFrame, attempts to determine 
                        from endog. If endog is a ndarray, period must be provided.
            - seasonal (int): Length of the seasonal smoother. Must be an odd integer, and should normally be >= 7 (default).
            - robust (bool): Flag indicating whether to use a weighted version that is robust to some forms of outliers.
            - seasonal_deg (int): Degree of seasonal LOESS. 0 (constant) or 1 (constant and trend).
        ---
        Returns:
            - params (dict): dictionary containing the identification parameters.
        """

        raw_timeseries = self.raw_timeseries(transect = transect)
        timeseries, timeseries_filled = self.process_timeseries(timeseries= raw_timeseries, transect= transect)

        stl = self.stl_decompositions(timeseries_filled, period= period, seasonal= seasonal, robust= robust, seasonal_deg = seasonal_deg).fit()

        params = identify_shorelinejump(stl_trend = stl.trend, df_filled= timeseries_filled, df_empty= timeseries, driver= 'nourishment', limit= limit)

        return params
    
    def littoral_driftbarrier(self, transect, n, min_transects=4, lim=0.9, stable_reg = 1, smooth_frac= 0.5):

        """
        Generate the parameters needed to identify a littoral drift barrier at a certain transect. Also a plot is created to 
        visualize the identification.
        ---
        Parameters:
            - transect (str): transect of a hotspot in the ds dataset.
            - n (int): the 1/n proportion of the transects at the start and end of the hotspot are evaluated against each other.
                        Suppose there are 10 transects and n equals 5. Then the 1/5 * 10 = 2 transects at the start and the end are 
                        evaluated regarding there shoreline changes.
            - min_transects (int): minimum amount of transects considered in a hotspot at which this function is executed.
                        Littoral drift barriers are expected to have a large spatial extent.
            - lim (float): limit of r2 in the merge_characteristics function. A characteristic is defined as shoreline positions in time
                        that can be fit by a linear line with a r2 below the lim compared to the shoreline positions.
            - stable_reg (int): the maximum changerate when the changerate of a characteristic is classified in the stable regime (m/yr).
            - smooth_frac (float): smoothening of the timeseries. Should be between 0 and 1.
        ---
        Returns:
            - params (dict): dictionary containing the identification parameters.
        """

        #check if there is a port in the hotspot and transects are missing
        dist_covered = []
        timeseries_filled, trend05_dict, lons, lats = self.get_hotspot(transect= transect)
        transects = timeseries_filled.columns
        ldb_type = 'Unknown'
        ncharc, list_cr = [[None, None]], [[None], [None]]
        end_trend_lst = [None]

        if len(transects) >= min_transects:
            for transect in transects:
                dist_covered.append(np.trapz(trend05_dict[transect].diff().dropna())/ (len(timeseries_filled)/12))

            r2, idx, dist_transects, sameslope = identify_structure(transects= transects, lons= lons, lats= lats, 
                                                                    dist_covered= dist_covered, plot= True)
            r2 = [r if not np.isnan(r) else 0 for r in r2 ]
            if len(r2) > 1:
                dfs = [timeseries_filled[transects[:idx[0]]], timeseries_filled[transects[idx[1]:]]]
                dist_perhs = [np.sum(dist_transects[:idx[0]]), np.sum(dist_transects[idx[1]:])]

            else:
                dfs = [timeseries_filled]
                dist_perhs = [np.sum(dist_transects)]

            max_dist = max(dist_transects)
            Metrics = {}
            num_transects = []
            ncharc, list_cr, end_trend_lst = [], [], []
            for i, df in enumerate(dfs):
                num_transects.append(len(df.columns))
                if len(df.columns) >= min_transects:
                    transects, transecte = df.columns[0], df.columns[-1]

                    ts = trend05_dict[transects]
                    te = trend05_dict[transecte]

                    ends, ende = np.trapz(abs(ts.diff().dropna())), np.trapz(abs(te.diff().dropna()))

                    N = int(len(df.columns) * (1/n))
                    if ends >= ende:
                        closest_transects = df.columns[0:1+N]
                        furthest_transects = df.columns[-1-N:]
                    else:
                        closest_transects = df.columns[-1-N:]
                        furthest_transects = df.columns[0:1+N]

                    # First Metric
                    closest_df = df[closest_transects]
                    closest_df_mean = closest_df.mean(axis= 1)
                    stl = STL(closest_df_mean, period= 12, seasonal= 37, robust= True).fit()
                    lowess = sm.nonparametric.lowess
                    z = lowess(endog= stl.trend + stl.resid, exog= np.arange(0, len(stl.trend.index)), frac= smooth_frac)
                    trend_closest = pd.Series(z[:, 1], index= df.index)
                    dist_closest = np.trapz(trend_closest.diff().dropna())

                    furthest_df = df[furthest_transects]
                    furthest_df_mean = furthest_df.mean(axis= 1)
                    stl = STL(furthest_df_mean, period= 12, seasonal= 37, robust= True).fit()
                    lowess = sm.nonparametric.lowess
                    z = lowess(endog= stl.trend + stl.resid, exog= np.arange(0, len(stl.trend.index)), frac= smooth_frac)
                    trend_furthest = pd.Series(z[:, 1], index= df.index)
                    dist_furthest = np.trapz(trend_furthest.diff().dropna())

                    last_10_trend = trend_closest[(trend_closest.index <= trend_closest.index[-1]) & (trend_closest.index >= trend_closest.index[-1]-pd.DateOffset(years = 10))]
                    last_20_trend = trend_closest[(trend_closest.index <= trend_closest.index[-1]-pd.DateOffset(years = 10)) & (trend_closest.index >= trend_closest.index[-1]-pd.DateOffset(years = 20))]
                    dY_10 = np.trapz(last_10_trend.diff().dropna())
                    dY_20 = np.trapz(last_20_trend.diff().dropna())
                    dYdiff = dY_10 / dY_20
                    end_trend_lst.append(dYdiff)
                    close_dates = merge_characteristics(trend = trend_closest, lim= lim, stable_reg= stable_reg)
                    furth_dates = merge_characteristics(trend= trend_furthest, lim= lim, stable_reg= stable_reg)

                    ncharc.append([len(close_dates), len(furth_dates)])

                    M1 = abs(round((dist_closest-dist_furthest) / dist_furthest, 1))
                    ldb_type = 'Updrift'
                    if dist_closest < 0:
                        ldb_type = 'Downdrift'
                    cr_clst, cr_flst = [], []

                    for dcl in close_dates:
                        ts_c = trend_closest[(trend_closest.index >= dcl[0]) & (trend_closest.index <= dcl[1])]
                        ys = ts_c.values
                        xs = np.arange(0, len(ts_c))
                        coefs = poly.polyfit(xs, ys, 1)
                        ys_sl = poly.polyval(xs, coefs)
                        cr_cl = (ys_sl[-1] - ys_sl[0]) / ((ts_c.index[-1] - ts_c.index[0]).days / 365)
                        cr_clst.append(cr_cl)
                    for dfu in furth_dates:
                        ts_f = trend_furthest[(trend_furthest.index >= dfu[0]) & (trend_furthest.index <= dfu[1])]
                        ys = ts_f.values
                        xs = np.arange(0, len(ts_f))
                        coefs = poly.polyfit(xs, ys, 1)
                        ys_sl = poly.polyval(xs, coefs)
                        cr_f = (ys_sl[-1] - ys_sl[0]) / ((ts_f.index[-1] - ts_f.index[0]).days / 365)
                        cr_flst.append(cr_f)

                    list_cr.append([cr_clst, cr_flst])

                    if dist_furthest > 0: label1, label2, c1, c2 = 'Trend$_{close}$', 'Trend$_{far}$', 'k', 'r'
                    else: label1, label2, c1, c2 = 'Trend$_{close}$', 'Trend$_{far}$',  'r', 'k'

                    fig, ax = plt.subplots(figsize= (20, 7.5))

                    trend_closest.plot(ax = ax, label = label1, color= c1)
                    trend_furthest.plot(ax= ax, label = label2, color= c2)

                    for i, dcl in enumerate(close_dates):
                        ts_c = trend_closest[(trend_closest.index >= dcl[0]) & (trend_closest.index <= dcl[1])]
                        ys = ts_c.values
                        xs = np.arange(0, len(ts_c))
                        coefs = poly.polyfit(xs, ys, 1)
                        ys_sl = poly.polyval(xs, coefs)
                        cr_cl = (ys_sl[-1] - ys_sl[0]) / ((ts_c.index[-1] - ts_c.index[0]).days / 365)
                        tc_spl = pd.Series(ys_sl, index = ts_c.index)
                        tc_spl.plot(ax=ax, color = 'k', linestyle = '--', linewidth = 4, label= "")

                    for dfu in furth_dates:
                        ts_f = trend_furthest[(trend_furthest.index >= dfu[0]) & (trend_furthest.index <= dfu[1])]
                        ys = ts_f.values
                        xs = np.arange(0, len(ts_f))
                        coefs = poly.polyfit(xs, ys, 1)
                        ys_sl = poly.polyval(xs, coefs)
                        cr_f = (ys_sl[-1] - ys_sl[0]) / ((ts_f.index[-1] - ts_f.index[0]).days / 365)
                        tf_spl = pd.Series(ys_sl, index = ts_f.index)
                        tf_spl.plot(ax=ax, color = 'r', linestyle = '--', linewidth = 4, label= "")
                        
                        ax.legend()
                        ax.set_ylabel('Shoreline Position [m]')
                        ax.set_xlabel('Time [years]')
                        ax.grid()

                    Metrics[f'{ldb_type}_{i+1}'] = M1

                else:
                    Metrics[f'Unknown_{i+1}'] = 0
                    ncharc.append([None, None])
                    list_cr.append([[None], [None]])
                    end_trend_lst.append(None)
            if len(Metrics) > 1 and not sameslope and not any(['Unknown' in m or 'Downdrift' in m for m in Metrics.keys()]):
                ldb_type = 'Double Updrift'
            elif len(Metrics) > 1 and sameslope:
                ldb_type = f'Splitted'
            return {'metric': Metrics, 'r2': r2, 'ldb_type': ldb_type, 'num_transects': num_transects, 'num_chars': ncharc}
        else:
            return {'metric': {'Unknown': 0}, 'r2': [0], 'ldb_type': ldb_type, 'num_transects': [len(transects)], 'num_chars': ncharc}