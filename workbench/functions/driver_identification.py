import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from statsmodels.tsa.seasonal import STL
from scipy.signal import argrelextrema
import scipy

class identify_drivers():

    def __init__(self, ds, transect):
        
        station = list(ds['transect_id'].values).index(str.encode(transect)) # select station
        self.ds = ds
        self.sp = ds.isel(stations=station)["sp"].values
        self.time = ds.time
        self.transect = transect

        outl_idx = [i for i,x in enumerate(self.ds.isel(stations= station)['outliers'].values) if x == 1]
        self.df_empty = pd.DataFrame({transect: self.sp}, index = self.time)
        self.df_empty.iloc[outl_idx] = np.nan

        self.df_filled = None

    def fill_timeseries(self):
        """
        Fill missing values using several ML algorithm
        """

        # Create a list of candidate models
        models = [
            LinearRegression(),
            SVR(kernel='linear'),
            RandomForestRegressor(n_estimators=100),
            GradientBoostingRegressor()
        ]

        datetime_index = self.df_empty.index
        self.df_empty.index = np.arange(0, len(self.df_empty))
        # Split the data into training and validation sets
        train_data = self.df_empty.dropna()
        valid_indices = self.df_empty.index[~self.df_empty[self.transect].isna()]

        # Initialize best model and R-squared value
        best_model = None
        best_r2 = -np.inf

        # Fit each candidate model and calculate R-squared on validation set
        for model in models:
            # Fit the model to the training data
            X_train = train_data.index.values.reshape(-1, 1)
            y_train = train_data[self.transect]

            model.fit(X_train, y_train)

            # Use the model to predict the missing values in the validation set
            X_valid = valid_indices.values.reshape(-1, 1)
            y_valid_pred = model.predict(X_valid)

            # Calculate the R-squared metric for the model's predictions on the validation set
            y_valid_true = self.df_empty.loc[valid_indices, self.transect]
            r2 = r2_score(y_valid_true, y_valid_pred)

            # Save the model and its R-squared value if it's the best so far
            if r2 > best_r2:
                best_model = model
                best_r2 = r2

        # Use the selected model to predict the missing values in the original timeseries data
        empty_indices =  self.df_empty.index[self.df_empty[self.transect].isna()]
        X_fill = empty_indices.values.reshape(-1, 1)
        y_fill = best_model.predict(X_fill)

        self.df_filled = self.df_empty.copy()
        # Replace the missing values with the predicted values
        self.df_filled.loc[empty_indices, self.transect] = y_fill

        self.df_empty.index = datetime_index
        self.df_filled.index = datetime_index

        return self.df_filled

    @staticmethod
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
    
    @staticmethod
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


    def seasonality(self, period = 12):
        pass
        """
        Generate the parameters needed to identify seasonality at a certain transect
        ---
        arguments:
            period (int): period of the seasonal oscillaiton
        ---
        returns:
            dict
        """
        matplotlib.rcParams.update({'font.size': 20})

        
        final_decade_empt = self.df_empty[self.df_empty.index >= pd.to_datetime('2013-03-01')]

        pmis_2013 = (final_decade_empt.isna().sum() / len(final_decade_empt) * 100).values
        df_transectf = self.df_filled
        # Decompose
        final_decade = df_transectf[df_transectf.index >= pd.to_datetime('2013-03-01')]
        years = np.unique(final_decade.index.year)
        
        if pmis_2013 <= 40 and len(years) >= 3:
            stl = STL(final_decade, period= period, seasonal= 61, robust= True, seasonal_deg = 0).fit()
            df_stl = pd.concat([stl.seasonal, stl.trend, stl.resid], axis= 1)
            stl_seasonal = df_stl.season
            stl_trend = df_stl.trend


            # Metrics
            Ps_10, Rs_10 = self.Oscillation_STL(df_STL= stl_seasonal)
            FS_seas_10 = self.Fit_Sine(tt= np.arange(0, len(stl_seasonal)), yy= stl_seasonal)
            r2_sine_10 = round(r2_score(stl_seasonal, FS_seas_10), 2)
            Fit_series = pd.Series(FS_seas_10, index= stl_seasonal.index)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize= (20, 15), sharex= True)

            transect_10_f = df_transectf[df_transectf.index >=pd.to_datetime('2013-03-01')]
            transect_10_e = self.df_empty[self.df_empty.index >= pd.to_datetime('2013-03-01')]
            filled_pos = transect_10_f[pd.isna(transect_10_e)]

            #plot1
            ax1.plot(transect_10_e.index, transect_10_e.values, label= 'SDS Positions', color= 'k', marker = 'o', linestyle = 'None', alpha = 0.4)
            ax1.plot(filled_pos.index, filled_pos.values, label= 'Filled Positions', color= 'k', marker = 'x', markersize = 10, linestyle = 'None', alpha = 0.6)
            ax1.plot(stl_trend.index, (stl_trend + stl_seasonal).values, label= 'Seasonal Component', color= 'g')
            ax1.plot(stl_trend.index, stl_trend.values, label= 'Trend', color= 'r')
            handles, labels = ax1.get_legend_handles_labels()
            order = [0, 1, 2, 3]
            ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
            ax1.set_ylabel('Distance w.r.t Origin [m]')

            #plot2
            ax2.plot(stl_seasonal.index, stl_seasonal.values, label= 'Seasonal Component', color= 'g')
            ax2.plot(stl_seasonal.index, FS_seas_10,  label= 'Sinusoidal fit', color= 'r', linestyle = '--')
            ax2.set_ylabel('Seasonal Shoreline Position [m]')
            ax2.set_xlabel('Time [years]')

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
    
                ax2.plot(seas_yr2.index[maxi_y2], seas_yr2.iloc[maxi_y2], color= 'k', marker= 'o', linestyle= 'None', label= 'Seasonal Extrema' if i == 0 else None)
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
                        'Ds': round(Rs_10, 0), 'Ps': round(Ps_10, 0),'r2': r2_sine_10, 
                        'Width_fit': [[min_width_fit, max_width_fit]], 'Width_data': [[min_width_data, max_width_data]], 
                        "Taccr": Taccr, 'Pmis2013': pmis_2013, 'Y2013': len(years)
                        }
        else:
            params = {
                        'Ds': 0, 'Ps': 0, 'r2': 0, 'Width_fit': [[0, 0]],
                        'Width_data': [[0, 0]], "Taccr": 0, 'Pmis2013': pmis_2013, 
                        'Y2013': len(years)
            }
        return params