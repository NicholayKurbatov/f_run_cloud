import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()


sales = pd.read_csv('d_history_sales.csv', parse_dates=['date', ])

start_tests = ['2021-08-31', '2021-09-30', '2021-10-31', '2021-11-30', '2021-12-31',
               '2022-01-31', '2022-02-28', '2022-03-31', '2022-04-30', '2022-05-31', 
               '2022-06-30', '2022-07-31',]
stop_tests = ['2021-09-30', '2021-10-31', '2021-11-30', '2021-12-31', '2022-01-31', 
              '2022-02-28', '2022-03-31', '2022-04-30', '2022-05-31', '2022-06-30', 
              '2022-07-31', '2022-08-31',]


def pred_sarima(data):
    # prepering
    store = data.store.iloc[0]
    item = data.item.iloc[0]
    
    tot = []
    for i in range(len(start_tests)):
        i = 0
        train = data[data.date < start_tests[i]].sort_values('date').copy().reset_index(drop=True)
        train.index = train.date
        train = train[['qnty']]

        # find additive seasonal component
        result_mul = seasonal_decompose(train['qnty'][-int(12*2):],   # 2 years
                                        model='additive', # 'multiplicative', # 
                                        extrapolate_trend='freq')

        seasonal_index = result_mul.seasonal[-12:].to_frame()
        seasonal_index['month'] = pd.to_datetime(seasonal_index.index).month

        # merge with the base data
        train['month'] = train.index.month
        train = pd.merge(train, seasonal_index, how='left', on='month')
        train.columns = ['qnty', 'month', 'seasonal_index']
        # Seasonal - fit stepwise auto-ARIMA
        sxmodel = pm.auto_arima(train[['qnty']].iloc[-int(12*4):], exogenous=train[['seasonal_index']],
                               start_p=1, start_q=1,
                               test='adf',
                               max_p=3, max_q=4, m=12,
                               start_P=1, seasonal=True,
                               d=1, D=1, trace=False,
                               error_action='ignore',  
                               suppress_warnings=True, 
                               stepwise=True)

        n_periods = 2
        pred = sxmodel.predict(n_periods=n_periods, return_conf_int=False)
        pred = pd.DataFrame({'date': [start_tests[i], ], 
                             'qnty': [data[data.date == start_tests[i]].qnty.iloc[0], ],
                             'qnty_1': [data[data.date == stop_tests[i]].qnty.iloc[0], ],
                             'sarima': [pred.iloc[0], ], 'sarima_1': [pred.iloc[1], ],})
        del sxmodel
        tot += [pred]
    
    tot = pd.concat(tot)
    tot['store'] = store
    tot['item'] = item
    return tot


sarima_m_preds = sales.groupby(['store', 'item'], as_index=False).progress_apply(pred_sarima)
sarima_m_preds.loc[sarima_m_preds['sarima'] < 0, 'sarima'] = 0
sarima_m_preds.loc[sarima_m_preds['sarima_1'] < 0, 'sarima_1'] = 0
sarima_m_preds.to_csv('sarima_m_preds.csv', index=False)