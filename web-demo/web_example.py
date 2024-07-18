import io
import base64
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from mip import Model, xsum, BINARY, INTEGER, CONTINUOUS, minimize, maximize, OptimizationStatus

import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

description = {}
description['status'] = '最佳化運算結果，可能為"FEASIBLE", "INFEASIBLE", "OPTIMAL", "NO_SOLUTION_FOUND", "ERROR"......'
description['data_freq'] = '系統操作週期'
description['max_sec'] = '求解上限時間(秒)，若時限內無法求得最佳解則返回"NO_SOLUTION_FOUND"或"FEASIBLE"'
description['c_cap'] = '契約容量(kWh)'
description['summer'] = '是否使用夏季流動電價（高壓用戶三段式時間電價/固定尖峰時間）'
description['basic_tariff_per_kwh'] = '基本時間電價，目前使用111年7月公布之版本，參考值：夏季=223.6/非夏季=166.9（https://www.taipower.com.tw/upload/6614/2022070417371173396.pdf#page=13）'
description['e_cap'] = '機組容量(kWh)'
description['soc_init'] = '初始SOC(%)，數值需介於0-100之間'
description['opt_soc_init'] = '是否由系統建議最佳初始SOC；若勾選此項，則目前設定值不參與計算，改由演算法推薦'
description['soc_end'] = '最後一個時段的SOC下限(%)，數值需介於0-100之間'
description['opt_soc_end'] = '是否由系統建議最佳結束SOC；若勾選此項，則目前設定值不參與計算，改由演算法推薦'
description['lb'] = 'SOC下限(%)，數值需介於0-100之間'
description['ub'] = 'SOC上限(%)，數值需介於0-100之間'
description['ess_degradation_cost_per_kwh_discharged'] = '儲能系統每放一度電所產生的成本與設備折舊。Ex: 1e+7/(1000*1000) = ＄10/kWh'
description['factory_profit_per_kwh'] = '工廠每使用一度電進行生產可收穫之利潤'
description['tendered_cap'] = '參與即時備轉服務之投標容量，以kWh為單位'
description['clearing_price_per_mwh'] = '日前即時備轉容量結清價格(每mWh)'

description['exec_rate'] = '調度執行率(%)，影響服務品質指標（https://atenergy.com.tw/wp-content/uploads/2020/11/%E8%BC%94%E5%8A%A9%E6%9C%8D%E5%8B%99%E5%8F%8A%E5%82%99%E7%94%A8%E5%AE%B9%E9%87%8F%E4%BA%A4%E6%98%93%E8%A9%A6%E8%A1%8C%E5%B9%B3%E5%8F%B0%E7%AC%AC%E4%BA%8C%E6%AC%A1%E5%85%AC%E9%96%8B%E8%AA%AA%E6%98%8E%E6%9C%832020.11.11.pdf#page=30）'

description['opt_tendered_cap'] = '是否由系統建議最佳投標容量，若勾選此項，下表中的"tendered_cap"一項不會作為模型輸入，改由演算法推薦。'
description['relax_tendered_step'] = '放鬆投標容量間格限制，若勾選，則最小投標單位不限於 0.1 mWh'
description['tendered_ub'] = '投標容量上限(mWh)，預設不可超過儲能機組SOC上限。'
description['tendered_lb'] = '投標容量下限(mWh)'

description['effectiveness_level'] = '效能級數，影響效能價格。 1: ＄100/mWh, 2: ＄60/mWh, 3:  ＄40/mWh, other: ＄0/mWh'
description['DA_margin_price_per_mwh'] = '日前電能邊際價格(每mWh)'
description['dispatch_ratio'] = '預估調度比例(%)，數值需介於0-100之間'

description['txt_ed_bid'] = '可自由設定投標情境。"bid"：該時段投標與否, "win"：該時段若投標是否會得標, "dispatch"：該時段若得標是否會被調度, "tendered_cap"：投標容量, "dispatch_ratio"：調度比例，以投標容量計算, "clearing_price"：該時段日前結清價, "marginal_price"：該時段日前電能報價。'
description['opt_bid'] = '是否由系統建議最佳投標時段；若勾選此項，下表中的"bid"一項不會作為模型輸入，改由演算法推薦；但仍需要設定"win"、"dispatch"兩項。'

description['limit_g_es_p'] = '儲能機組充電功率上限(kW)'
description['limit_es_p'] = '儲能機組放電功率上限(kW)'
description['limit_g_p'] = '電網輸入功率上限(kW)'
description['limit_pv_p'] = '太陽能機組輸送功率上限(kW)'
description['loss_coef'] = '電能輸送損失係數，數值需介於0-1之間'
description['bulk_tariff_per_kwh'] = '太陽能躉售電價(每kWh)。Ex: (3.8680 + 5.8952)/2 = 4.8816'

# inititalize session
if 'data_load' not in st.session_state:
    st.session_state['data_load'] = None
if 'data_pv' not in st.session_state:
    st.session_state['data_pv'] = None
if 'optimization_status' not in st.session_state:
    st.session_state['optimization_status'] = None    
if 'optimization_count' not in st.session_state:
    st.session_state['optimization_count'] = 0
    
def make_data_plot(df, title='data', x='time', y='value'):
    df = df.reset_index()
    fig = px.line(df, x=x, y=y)
    fig.update_layout(
        title=title,
        xaxis_title="time",
        yaxis_title="value(kWh)",
        # width=1800, 
        # height=800, 
        font=dict(
            family="Arial",
            # size=20,
            color="black"
        ))
    return fig

def make_result_plot(df, title='data', secondary_y_limit=None):
    x_index = df.index
    dash_line = dict(dash = 'dash')
    opacity = 0.4
    if not secondary_y_limit:
        secondary_y_limit = [df['total_power_flow_of_ESS'].min() - 100, max(df['power_from_ESS_to_factory'].max(), df['power_from_PV_to_factory'].max())]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=x_index, y=[7000]*len(df), name='Contract Capacity', marker_color='#DD6251', line=dash_line), secondary_y=True)
    fig.add_trace(go.Scatter(x=x_index, y=df['load'], name='Load', marker_color='#26272C'), secondary_y=True)
    fig.add_trace(go.Scatter(x=x_index, y=df['total_power_from_grid_to_factory'], name='Total power from grid to factory', marker_color='#6A90AB'), secondary_y=True)
    fig.add_trace(go.Scatter(x=x_index, y=df['safe_range'], name='safe loading range', marker_color='green', line=dash_line), secondary_y=True)
    # fig.add_trace(go.Bar(x=x_index, y=df['power_from_ESS_to_factory'], name='Power from ESS to factory', marker_color='#F7A64F', opacity=opacity), secondary_y=False)
    fig.add_trace(go.Bar(x=x_index, y=df['total_power_flow_of_ESS'], name='total power flow of ESS', marker_color='#AF5F08', opacity=opacity), secondary_y=False)
    fig.add_trace(go.Bar(x=x_index, y=df['power_from_PV_to_factory'], name='Power from PV to factory', marker_color='#42AC6D', opacity=opacity), secondary_y=False)
    # fig.add_trace(go.Scatter(x=x_index, y=df['pv'], name='PV'), secondary_y=False)

    # update layout
    fig.update_layout(dict(yaxis2={'anchor': 'x', 'overlaying': 'y', 'side': 'left'}, 
                           yaxis={'anchor': 'x', 'domain': [0.0, 1.0], 'side':'right'}))
    fig.update_layout(title_text=title, xaxis_title="time", yaxis_title="Power(kWh)", margin=dict(t=28),
                      font=dict(size=32, family="Arial", color="black"))
    # fig.update_yaxes(range=[0,df['total_power_from_grid_to_factory'].min()-100], secondary_y=True)
    # fig.update_yaxes(range=[0, secondary_y_limit], secondary_y=False)
    # fig.update_yaxes(range=secondary_y_limit, secondary_y=False)
    
    fig.update_yaxes(rangemode='nonnegative', scaleanchor='y', scaleratio=1, constraintoward='bottom', secondary_y=True)
    fig.update_yaxes(rangemode='normal', scaleanchor='y2', scaleratio=0.5, constraintoward='bottom', secondary_y=False)
    
    return fig
    
def get_service_quality_index(exec_rate=95):
    if exec_rate >= 95:
        return 1.0
    elif exec_rate >= 85:
        return 0.7
    elif exec_rate >= 70:
        return 0
    else:
        return -240

def get_effectiveness_price(level=0):
    if level == 1:
        return 100 # per mwh
    elif level == 2:
        return 60
    elif level == 3:
        return 40
    else:
        return 0
       
def optimize(df_load, df_pv, max_sec, data_freq, 
             c_cap, basic_tariff_per_kwh, summer, e_cap, soc_init, opt_soc_init, soc_end, opt_soc_end, lb, ub, 
             ess_degradation_cost_per_kwh_discharged, factory_profit_per_kwh, 
             tendered_cap, clearing_price_per_mwh, exec_rate, effectiveness_level, DA_margin_price_per_mwh, dispatch_ratio, 
             opt_bid, opt_tendered_cap, relax_tendered_step, tendered_lb, tendered_ub, bid, bid_win, dispatch, 
             limit_g_es_p, limit_es_p, limit_g_p, limit_pv_p, loss_coef, bulk_tariff_per_kwh, **kwargs):
    # try:
        report = []
        result = {}
        ################################################## parameter setting ##################################################
        #### auxiliary var.
        freq = data_freq
        consecutive_n = int(60/freq)
        n = int(len(df_load)/(freq/5))
        #### flatten data
        # index
        index = df_load['time'].iloc[::int(freq/5)].values.flatten()
        # Load
        load = df_load['value'].iloc[::int(freq/5)].values.flatten()
        # load[20] += 2000
        # PV
        pv = df_pv['value'].iloc[::int(freq/5)].values.flatten()
    
        #### energy charge(111/7)
        if summer:
            # summer
            p1 = np.array([1.58]*int(n*(15/48))) # 0000-0730
            p2 = np.array([3.54]*int(n*(5/48))) # 0730-1000
            p3 = np.array([5.31]*int(n*(4/48))) # 1000-1200
            p4 = np.array([3.54]*int(n*(2/48))) # 1200-1300
            p5 = np.array([5.31]*int(n*(8/48))) # 1300-1700
            p6 = np.array([3.54]*int(n*(11/48))) # 1700-2230
            p7 = np.array([1.58]*int(n*(3/48))) # 2230-0000
            price = np.hstack([p1, p2, p3, p4, p5, p6, p7])

            if freq == 60:
                price = np.array([1.58]*7 + # 0000-0700
                                 [2.56]*1 + # 0700-0800(mixed)
                                 [3.54]*2 + # 0800-1000
                                 [5.31]*2 + # 1000-1200
                                 [3.54]*1 + # 1200-1300
                                 [5.31]*4 + # 1300-1700
                                 [3.54]*5 + # 1700-2200
                                 [2.56]*1 + # 2200-2300(mixed)
                                 [1.58]*1 ) # 2300-0000
        else:
            # other
            p1 = np.array([1.50]*int(n*(15/48))) # 0000-0730
            p2 = np.array([3.44]*int(n*(30/48))) # 0730-2230
            p3 = np.array([1.50]*int(n*(3/48))) # 2230-0000
            price = np.hstack([p1, p2, p3])

            if freq == 60:
                price = np.array([1.50]*7 + # 0000-0700
                                 [2.32]*1 + # 0700-0800(mixed)
                                 [3.44]*14 + # 0800-2200
                                 [2.32]*1 + # 2200-2300(mixed)
                                 [1.50]*1 ) # 2300-0000
        #####################################
        # #### energy charge(112/4)
        # if summer:
        #     # summer
        #     p1 = np.array([1.91]*int(n*(9/24))) # 0000-0900
        #     p2 = np.array([4.39]*int(n*(7/24))) # 0900-1600
        #     p3 = np.array([7.03]*int(n*(6/24))) # 1600-2200
        #     p4 = np.array([4.39]*int(n*(2/24))) # 2200-0000
        #     price = np.hstack([p1, p2, p3, p4])
        # else:
        #     # other
        #     p1 = np.array([1.75]*int(n*(6/24))) # 0000-0600
        #     p1 = np.array([4.11]*int(n*(5/24))) # 0600-1100
        #     p1 = np.array([1.75]*int(n*(3/24))) # 1100-1400
        #     p1 = np.array([4.11]*int(n*(10/24))) # 1400-0000
        #     price = np.hstack([p1, p2, p3, p4])
        #####################################
        
        # Multiplication factor for penalty charge
        dummy_penalty_coef_1 = 2
        dummy_penalty_coef_2 = 3
        
        #### ESS
        # unit
        soc_init = soc_init/100
        soc_end = soc_end/100
        lb = lb/100
        ub = ub/100
        
        # init.
        e_init = e_cap*soc_init
        e_end = e_cap*soc_end
        # ESS boundary
        soc_lb = np.array([e_cap*lb for i in range(n)])
        soc_ub = np.array([e_cap*ub for i in range(n)])
        
        #### Trading
        tendered_cap = [v*10 for v in tendered_cap] # temporarily converted to integer level for usage of INTEGER variable type. Ex: 1.2 mWh --> 12.0
        service_quality_index = get_service_quality_index(exec_rate)
        effectiveness_price_per_kwh = get_effectiveness_price(effectiveness_level)/1000
        clearing_price_per_kwh = [v/1000 for v in clearing_price_per_mwh]
        DA_margin_price_per_kwh = [v/1000 for v in DA_margin_price_per_mwh]
        dispatch_ratio = [v/100 for v in dispatch_ratio]
        
        #### other
        # big M for modeling
        M = 1e+15
        ################################################## parameter setting ##################################################
        ################################################## model ##################################################
        
        #### initialize a model
        ################################################################################
        ## init.
        model = Model(solver_name='CBC')
        
        ## decision variables
        # obj.
        revenue = model.add_var(name='revenue', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        income = model.add_var(name='income', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        cost = model.add_var(name='cost', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        # power during the time interval
        p_g_f = [model.add_var(name=f"power_from_grid_to_factory_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        p_es_f = [model.add_var(name=f"power_from_ESS_to_factory_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        p_pv_f = [model.add_var(name=f"power_from_PV_to_factory_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        p_pv_es = [model.add_var(name=f"power_from_PV_to_ESS_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)] # the power from PV will be served to the factory first
        p_pv_g = [model.add_var(name=f"power_from_PV_to_grid_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)] # (躉售)the power from PV will be served to the factory first 
        p_g_es = [model.add_var(name=f"power_from_grid_to_ESS_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        # total power from grid (aux for visulization)
        total_g = [model.add_var(name=f'total_power_from_grid_t{i}', var_type=CONTINUOUS) for i in range(n)]
        total_g_f = [model.add_var(name=f'total_power_from_grid_to_factory_t{i}', var_type=CONTINUOUS) for i in range(n)]
        total_g_es = [model.add_var(name=f'total_power_from_grid_to_ESS_t{i}', var_type=CONTINUOUS) for i in range(n)]
        
        # ESS SOC "at the beginning" of the time interval
        es = [model.add_var(name=f"ESS_SOC_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        # excessive load
        b_exceed = [model.add_var(name=f"if_exceed_110%_cap_at_t{i}", var_type=BINARY) for i in range(n)]
        dummy_g_1 = [model.add_var(name=f"dummy_power_1_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        dummy_g_2 = [model.add_var(name=f"dummy_power_2_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        dummy_g_f = [model.add_var(name=f"dummy_power_from_grid_to_factory_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        dummy_g_es = [model.add_var(name=f"dummy_power_from_grid_to_ESS_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        # ESS charging/discharging status (or)
        b_chg = [model.add_var(name=f"ESS_is_charging_at_t{i}", var_type=BINARY) for i in range(n)]
        b_dch = [model.add_var(name=f"ESS_is_discharging_at_t{i}", var_type=BINARY) for i in range(n)]
        aux_p_g_es = [model.add_var(name=f"aux_power_from_grid_to_ESS_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        aux_p_es_f = [model.add_var(name=f"aux_power_from_ESS_to_factory_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        aux_dummy_g_es = [model.add_var(name=f"aux_dummy_power_from_grid_to_ESS_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        # total excessive power (aux for visulization)
        total_dummy = [model.add_var(name=f'total_excessive_power_t{i}', var_type=CONTINUOUS) for i in range(n)]# total excessive power (aux for visulization)
        total_flow_es = [model.add_var(name=f'total_power_flow_of_ESS_t{i}', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf')) for i in range(n)]
        # dummies for penalty calculation
        q_1 = model.add_var(name=f"max_dummy_power_1", var_type=CONTINUOUS)
        q_2 = model.add_var(name=f"max_dummy_power_2", var_type=CONTINUOUS)
        b_max_aux_1 = [model.add_var(name=f"max_func_aux_1_t{i}", var_type=BINARY) for i in range(n)]
        b_max_aux_2 = [model.add_var(name=f"max_func_aux_2_t{i}", var_type=BINARY) for i in range(n)]
        # bidding decision
        bid = [1 if v else 0 for v in bid]
        bid_win = [1 if v else 0 for v in bid_win]
        dispatch = [1 if v else 0 for v in dispatch]
        if opt_bid:
            bid = [model.add_var(name=f"if_bid_at_t{i}", var_type=BINARY) for i in range(n)]
        # tendered capacity
        if opt_tendered_cap:
            if relax_tendered_step:
                tendered_cap = [model.add_var(name=f"tendered_cap_at_t{i}", var_type=CONTINUOUS) for i in range(n)]
            else:
                tendered_cap = [model.add_var(name=f"tendered_cap_at_t{i}", var_type=INTEGER) for i in range(n)]
        # for multiplication of tendered capacity and bidding decision
        # aux_tendered_cap = [model.add_var(name=f"aux_tendered_cap_at_t{i}", lb=float('-Inf'), ub=float('Inf'), var_type=CONTINUOUS) for i in range(n)]
        if relax_tendered_step:
            aux_tendered_cap = [model.add_var(name=f"aux_tendered_cap_at_t{i}", lb=float('-Inf'), ub=float('Inf'), var_type=CONTINUOUS) for i in range(n)]
        else:
            aux_tendered_cap = [model.add_var(name=f"aux_tendered_cap_at_t{i}", lb=float('-Inf'), ub=float('Inf'), var_type=INTEGER) for i in range(n)]
            
        #### add constraints
        ################################################################################
        ## basic constraints
        # linearization of multiplication of decision variables
        for i in range(n):
            # either charging or discharging
            model.add_constr(b_chg[i] + b_dch[i] <= 1)
            model.add_constr(aux_p_g_es[i] <= p_g_es[i])
            model.add_constr(aux_p_g_es[i] <= M*b_chg[i])
            model.add_constr(aux_p_g_es[i] >= p_g_es[i] + M*(b_chg[i]-1))
            model.add_constr(aux_dummy_g_es[i] <= dummy_g_es[i])
            model.add_constr(aux_dummy_g_es[i] <= M*b_chg[i])
            model.add_constr(aux_dummy_g_es[i] >= dummy_g_es[i] + M*(b_chg[i]-1))
            model.add_constr(aux_p_es_f[i] <= p_es_f[i])
            model.add_constr(aux_p_es_f[i] <= M*b_dch[i])
            model.add_constr(aux_p_es_f[i] >= p_es_f[i] + M*(b_dch[i]-1))
            # tendered capacity and bidding decision
            if opt_tendered_cap:
                model.add_constr(aux_tendered_cap[i] >= 0) # just ensure
                model.add_constr(aux_tendered_cap[i] >= tendered_cap[i] - M*(1-bid[i]))
                model.add_constr(aux_tendered_cap[i] <= M*bid[i])
                model.add_constr(aux_tendered_cap[i] <= tendered_cap[i])
            else:
                model.add_constr(aux_tendered_cap[i] == tendered_cap[i]*bid[i])
                
        # non-negative
        for i in range(n):
            model.add_constr(p_g_f[i] >= 0.0)
            model.add_constr(p_es_f[i] >= 100*aux_tendered_cap[i]*bid_win[i]*dispatch[i]*dispatch_ratio[i])
            model.add_constr(aux_p_es_f[i] >= 100*aux_tendered_cap[i]*bid_win[i]*dispatch[i]*dispatch_ratio[i])
            model.add_constr(p_pv_f[i] >= 0.0)
            model.add_constr(p_pv_es[i] >= 0.0)
            model.add_constr(p_pv_g[i] >= 0.0)
            model.add_constr(p_g_es[i] >= 0.0)
            model.add_constr(aux_p_g_es[i] >= 0.0)
            model.add_constr(dummy_g_f[i] >= 0.0)
            model.add_constr(dummy_g_es[i] >= 0.0)
            model.add_constr(aux_dummy_g_es[i] >= 0.0)
            
        ## maximum function of dummy variables, for panelty calculation
        for i in range(n):
            model.add_constr(q_1 >= dummy_g_1[i])
            model.add_constr(q_1 <= dummy_g_1[i] + M*b_max_aux_1[i])
            model.add_constr(q_2 >= dummy_g_2[i])
            model.add_constr(q_2 <= dummy_g_2[i] + M*b_max_aux_2[i])
        model.add_constr( xsum( b_max_aux_1[i] for i in range(n) ) <= n-1 )
        model.add_constr( xsum( b_max_aux_2[i] for i in range(n) ) <= n-1 )

        ## factory
        # load
        for i in range(n):
            model.add_constr(dummy_g_f[i] + p_g_f[i] + loss_coef*(aux_p_es_f[i] + p_pv_f[i]) == load[i])
        # grid contract boundary (panelty for excessive capacity are added later with dummy vars.)
        for i in range(n):
            model.add_constr(p_g_f[i] + p_pv_f[i] - p_pv_g[i] <= c_cap - 100*aux_tendered_cap[i]*bid_win[i]*dispatch[i]*dispatch_ratio[i]) ############################ 全額躉售計費修改

        ## dispatch
        # 1. sum of dispatch_start <= 1 in any arbitrary 3 consecutive hours
        if opt_bid:
            for i in range(n-3*consecutive_n):
                model.add_constr(xsum(bid[j]*bid_win[j]*dispatch[j] for j in range(i, i+3*consecutive_n)) <= 1)
        
        ## bidding
        # bounds
        for i in range(n):
            model.add_constr(aux_tendered_cap[i] >= 10*tendered_lb*bid[i])
            model.add_constr(aux_tendered_cap[i] <= 10*tendered_ub*bid[i])
        
            
        ## ESS
        # init.
        if not opt_soc_init:
            model.add_constr(es[0] == e_init)
        # ending SOC lb
        if not opt_soc_end:
            model.add_constr(es[-1] >= e_end)
        
        # output capacity limitation
        for i in range(n):
            model.add_constr(aux_p_es_f[i] <= es[i])
            model.add_constr(p_es_f[i] <= es[i])
            # model.add_constr(p_es_f[i] <= es[i])
        # update
        for i in range(1,n):
            model.add_constr(es[i] == es[i-1] + (aux_dummy_g_es[i-1] + aux_p_g_es[i-1] + p_pv_es[i-1] - aux_p_es_f[i-1])/consecutive_n)
        # SOC boundary
        for i in range(n):
            model.add_constr(es[i] >= soc_lb[i])
            model.add_constr(es[i] <= soc_ub[i])
            
        # print(e_init)
        # print(soc_lb[i], soc_ub[i])
        ## PV
        # flow balance
        for i in range(n):
            model.add_constr((p_pv_f[i] + p_pv_es[i] + p_pv_g[i]) == pv[i])
        # serving priority
        for i in range(n):
            model.add_constr(p_pv_f[i] >= p_pv_g[i])

        ## split excessive power for additional tariff calculation
        for i in range(n):
            model.add_constr(0.1*c_cap*b_exceed[i] <= dummy_g_1[i])
            model.add_constr(dummy_g_1[i] <= 0.1*c_cap)
            model.add_constr(dummy_g_2[i] >= 0)
            model.add_constr(dummy_g_2[i] <= b_exceed[i]*M)
            model.add_constr(dummy_g_1[i] + dummy_g_2[i] == dummy_g_f[i] + aux_dummy_g_es[i])

        ## transfer limitation
        for i in range(n):
            model.add_constr(p_g_f[i] <= limit_g_p)
            model.add_constr(p_es_f[i] <= limit_es_p)
            model.add_constr(aux_p_es_f[i] <= limit_es_p)
            model.add_constr(p_pv_f[i] <= limit_pv_p)
            
            model.add_constr(p_pv_es[i] <= limit_pv_p)
            # model.add_constr(p_pv_es[i] <= limit_g_es_p)
            
            model.add_constr(p_pv_g[i] <= limit_pv_p)
            
            model.add_constr(p_g_es[i] <= limit_g_es_p)
            model.add_constr(aux_p_g_es[i] <= limit_g_es_p)
            
            model.add_constr(dummy_g_f[i] <= limit_g_p)
            model.add_constr(dummy_g_es[i] <= limit_g_es_p)
            model.add_constr(dummy_g_es[i] <= limit_g_p)
            model.add_constr(aux_dummy_g_es[i] <= limit_g_es_p)
            # model.add_constr(aux_dummy_g_es[i] <= limit_g_p)
            
        #### obj. ensemble
        ################################################################################
        # dispatch income, 即時備轉收益 = (容量費 + 效能費) × 服務品質指標 ＋ 電能費
        dispatch_income = model.add_var(name='dispatch_income', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        capacity_reserve_income = model.add_var(name='capacity_reserve_income', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        effectiveness_income = model.add_var(name='effectiveness_income', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        energy_income = model.add_var(name='dispatch_income', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        # 容量費
        model.add_constr(capacity_reserve_income == xsum( (clearing_price_per_kwh[i]*100*aux_tendered_cap[i]*bid_win[i]) for i in range(n) ))
        # 效能費
        model.add_constr(effectiveness_income == xsum( (effectiveness_price_per_kwh*100*aux_tendered_cap[i]*bid_win[i]*dispatch[i]) for i in range(n) ))
        # 電能費
        model.add_constr(energy_income == xsum( (DA_margin_price_per_kwh[i]*100*aux_tendered_cap[i]*bid_win[i]*dispatch[i]*dispatch_ratio[i]) for i in range(n) ))
        # total
        model.add_constr(dispatch_income == ((capacity_reserve_income+effectiveness_income)*service_quality_index + energy_income)/consecutive_n)

        # factory income
        factory_income = model.add_var(name='factory_income', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        model.add_constr(factory_income == xsum( factory_profit_per_kwh*load[i]/consecutive_n for i in range(n) ))

        # PV income
        pv_income = model.add_var(name='PV_income', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        model.add_constr(pv_income == xsum( bulk_tariff_per_kwh*(p_pv_g[i]+p_pv_f[i])/consecutive_n for i in range(n) )) ############################ 全額躉售計費修改

        # total income
        model.add_constr(income == (dispatch_income + factory_income + pv_income))

        # fixed eletricity tariff
        fixed_e_cost = model.add_var(name='fixed_eletricity_tariff', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        model.add_constr(fixed_e_cost == basic_tariff_per_kwh*(1*c_cap + dummy_penalty_coef_1*q_1 + dummy_penalty_coef_2*q_2)/30) ############################ 全額躉售計費修改

        # usage eletricity tariff
        usage_e_cost = model.add_var(name='usage_eletricity_tariff', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        model.add_constr(usage_e_cost == xsum( price[i]*(p_g_f[i] + p_pv_f[i] - p_pv_g[i] + aux_p_g_es[i] + dummy_g_1[i] + dummy_g_2[i])/consecutive_n for i in range(n) )) ############################ 全額躉售計費修改

        # ESS degradation
        ess_dis_cost = model.add_var(name='ess_discharging_degradation_cost', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        model.add_constr(ess_dis_cost == xsum( ess_degradation_cost_per_kwh_discharged*aux_p_es_f[i]/consecutive_n for i in range(n) ))

        # total cost
        model.add_constr(cost == (fixed_e_cost + usage_e_cost + ess_dis_cost))

        # total revenue
        model.add_constr(revenue == (income - cost))
        model.objective = maximize(revenue)
        
        #### Other given setting and aux var
        ################################################################################
        # no power from PV to Grid/ESS directly
        for i in range(n):
            model.add_constr(p_pv_es[i] == 0) #### 無饋線
            model.add_constr(p_pv_g[i] == 0) #### 目前為全額躉售
        
        # total power from grid (aux for visulization)
        for i in range(n):
            model.add_constr(total_g_f[i] == p_g_f[i] + dummy_g_f[i])
            model.add_constr(total_g_es[i] == aux_p_g_es[i] + aux_dummy_g_es[i])
            model.add_constr(total_g[i] == total_g_f[i] + total_g_es[i])
            
        # total excessive power (aux for visulization)
        for i in range(n):
            model.add_constr(total_dummy[i] == dummy_g_1[i] + dummy_g_2[i])
        # total power flow of ESS (aux for visulization)
        for i in range(n):
            model.add_constr(total_flow_es[i] == total_g_es[i] + p_pv_es[i] - aux_p_es_f[i])
        ################################################################################
        ################################################## model ##################################################
        ################################################## optimize ##################################################
        status = model.optimize(max_seconds=max_sec)
        ################################################## result ##################################################
        if status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
            # report
            report.append(('總收益', revenue.x))
            report.append(('輔助服務總收入', dispatch_income.x))

            report.append(('容量費', capacity_reserve_income.x))
            if capacity_reserve_income.x is None:
                report.append(('容量費(服務品質指標)', None))
            else:
                report.append(('容量費(服務品質指標)', capacity_reserve_income.x * service_quality_index))

            report.append(('效能費', effectiveness_income.x))
            if effectiveness_income.x is None:
                report.append(('效能費(服務品質指標)', None))            
            else:
                report.append(('效能費(服務品質指標)', effectiveness_income.x * service_quality_index))

            report.append(('電能費', energy_income.x))
            report.append(('工廠生產收入', factory_income.x))
            report.append(('太陽能躉售收入', pv_income.x))
            report.append(('總收入', income.x))
            report.append(('基本電費', fixed_e_cost.x))
            report.append(('流動電費', usage_e_cost.x))
            report.append(('儲能設備耗損成本', ess_dis_cost.x))
            report.append(('總成本', cost.x))
            if status == 0:
                report = [round(r, 4) for r in report]
            df_report = pd.DataFrame(report, columns=['項目', '金額'])

            # optimized schedule
            result['time'] = index
            result['load'] = [val for val in load]
            result['pv'] = [val for val in pv]

            result['safe_range'] = [c_cap-100*aux_tendered_cap[i].x*bid_win[i]*dispatch[i]*dispatch_ratio[i] for i in range(n)]

            result['power_from_grid_to_factory'] = [v.x for v in p_g_f]
            result['power_from_ESS_to_factory'] = [v.x for v in aux_p_es_f]
            result['power_from_PV_to_factory'] = [v.x for v in p_pv_f]
            result['power_from_PV_to_ESS'] = [v.x for v in p_pv_es]
            result['power_from_PV_to_grid'] = [v.x for v in p_pv_g]
            result['power_from_grid_to_ESS'] = [v.x for v in aux_p_g_es]

            result['ESS_SOC'] = [v.x for v in es]
            result['ESS_is_charging'] = [v.x for v in b_chg]
            result['ESS_is_discharging'] = [v.x for v in b_dch]

            result['exceed_contract_capacity'] = [v.x for v in b_exceed]
            result['excessive_power_below_110%'] = [v.x for v in dummy_g_1]
            result['excessive_power_over_110%'] = [v.x for v in dummy_g_2]
            result['excessive_power_from_grid_to_factory'] = [v.x for v in dummy_g_f]
            result['excessive_power_from_grid_to_ESS'] = [v.x for v in aux_dummy_g_es]

            if opt_bid:
                result['bid'] = [v.x for v in bid]
            else:
                result['bid'] = [v for v in bid]
            result['bid_win'] = [v for v in bid_win]
            result['dispatch'] = [v for v in dispatch]
            result['aux_tendered_cap(mWh)'] = [v.x/10 for v in aux_tendered_cap] # [v.x/10 if v.x else None for v in aux_tendered_cap]

            result['total_power_from_grid'] = [v.x for v in total_g]
            result['total_power_from_grid_to_factory'] = [v.x for v in total_g_f]
            result['total_power_from_grid_to_ESS'] = [v.x for v in total_g_es]
            result['total_excessive_power'] = [v.x for v in total_dummy]
            result['total_power_flow_of_ESS'] = [v.x for v in total_flow_es]
            if status == 0:
                for k, l in result.items():
                    if k == 'time':
                        continue
                    else:
                        if k in ['ESS_is_charging', 'ESS_is_discharging', 'bid', 'bid_win', 'dispatch']:
                            result[k] = [int(val) for val in l]
                        else:
                            result[k] = [round(val, 4) for val in l]
            df_result = pd.DataFrame(result)
            
            return status.name, df_report, df_result
        else:
            return status.name, pd.DataFrame(), pd.DataFrame()

def sidebar_bg(filepath):
    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

        st.sidebar.markdown(
            f"""
            <div style="display:table; margin-top:-28%; margin-left:-2%; 
                        font-family:'Source Sans Pro', sans-serif; margin-bottom: -1rem; color: rgb(163, 168, 184); font-size: 14px;">
                <img src="data:image/png;base64,{data}" width="100" height="100">
                <p style="font-family:'Source Sans Pro', sans-serif; color: rgb(163, 168, 184); font-size: 14px;">
                    A dog saying "THIS IS FINE".
                </p>
            </div>
            """,
            # f"""
            # <div style="display:table; margin-top:-32%; margin-left:-2%; 
            # font-family:'Source Sans Pro', sans-serif; margin-bottom: -1rem; color: rgb(163, 168, 184); font-size: 14px;">
            #     <img src="data:image/png;base64,{data}" width="60" height="60">
            #     <p style="font-family:'Source Sans Pro', sans-serif; margin-bottom: -1rem; color: rgb(163, 168, 184); font-size: 14px;">
            #         {caption}
            #     </p>
            # </div>
            # """,
            
            # f"""
            # <div style="display:table; margin-top:-32%; margin-left:-2%;">
            #     <img src="data:image/png;base64,{data}" width="60" height="60">
            #     <p style="font-family:'Source Sans Pro', sans-serif; margin-bottom: -1rem; color: rgb(163, 168, 184); font-size: 14px;">
            #         {caption}
            #     </p>
            # </div>
            # """,
            unsafe_allow_html=True,
        )

def verify_bd_rule(df, opt_bid=1):
    if opt_bid:
        return all([bw >= d for bw, d in zip(df['win'], df['dispatch'])])
    else:
        row = all([bw >= d for bw, d in zip(df['win'], df['dispatch'])])
        tmp = df['bid']*df['win']*df['dispatch']
        dispatch = all([tmp[i:i+3].sum() <= 1 for i in range(len(df)-2)])
        return all([row, dispatch])

# def verify_bd_rule(bid=None, bid_win=None, dispatch=None, opt_bid=1):
#     if opt_bid:
#         return all([bw >= d for bw, d in zip(bid_win, dispatch)])
#     else:
#         row = all([b >= bw >= d for b, bw, d in zip(bid, bid_win, dispatch)])
#         dispatch = all([bid[i]*dispatch[i]+bid[i+1]*dispatch[i+1]+bid[i+2]*dispatch[i+2] <= 1 for i in range(len(bid)-2)])
#         return all([row, dispatch])

# def verify_bd_rule(df, opt_bid=1):
#     if opt_bid:
#         return all([bw >= d for bw, d in zip(df['win'], df['dispatch'])])
#     else:
#         row = all([b >= bw >= d for b, bw, d in zip(df['bid'], df['win'], df['dispatch'])])
#         # dispatch = all([df['bid'][i]*df['dispatch'][i]+df['bid'][i+1]*df['dispatch'][i+1]+df['bid'][i+2]*df['dispatch'][i+2] <= 1 for i in range(len(df)-2)])
#         dispatch = all([df['dispatch'][i:i+3].sum() <= 1 for i in range(len(df)-2)])
#         print([(df['dispatch'][i:i+2], df['dispatch'][i:i+2].sum() <= 1) for i in range(len(df)-2)])
#         return all([row, dispatch])


def verify_tendered_capacity_integrity(ed_bid, relax=0):
    if not relax:
        return all([(10*i).is_integer() for i in ed_bid['tendered_cap(mWh)']])
    return True

def verify_tendered_capacity_in_bound(ed_bid, lb=0, ub=float('Inf')):
    return all([lb <= i <= ub for i in ed_bid['tendered_cap(mWh)']])

def verify_tendered_capacity_non_negative(ed_bid):
    return all([i >= 0 for i in ed_bid['tendered_cap(mWh)']])


######## page body
# set layout
st.set_page_config(page_title='Power Optimizer test(Cht)', layout="wide", page_icon='./fine_n.png')
# =None, layout="centered", initial_sidebar_state="auto", menu_items=None


# style
# button alignment
# style = "<style>.row-widget.stButton {text-align: center;}</style>"
st.markdown("<style>.row-widget.stButton {text-align: center;}</style>", unsafe_allow_html=True)

# button color
# st.markdown("<style>div.stButton > button:first-child {background-color: #00754a;}</style>", unsafe_allow_html=True)
# m = st.markdown("<style>div.stButton > button:first-child {background-color: #f2f0eb;}</style>", unsafe_allow_html=True)



# ; 
# ;color:white;font-size:20px;height:3em;width:30em;border-radius:10px 10px 10px 10px;

 # ("visible" or "hidden" or "collapsed")

# Header
st.title('最佳化工具 Demo')

# init. layout
### upload field
exp_upload = st.expander('資料上傳區域', expanded=True)
exp_upload.markdown('<p style="font-family:Source Sans Pro, sans-serif; font-size: 1.5rem; font-weight: bold;">於此處上傳資料</p>', unsafe_allow_html=True)
exp_upload.text('目前使用採樣週期為五分鐘的單日資料。')

col_upload = exp_upload.columns(2)

# Create data sections
upload_load = col_upload[0].file_uploader('工廠負載資料')
upload_pv = col_upload[1].file_uploader('太陽能發電資料')

######## sidebar
sidebar_bg('./phrase_clean.png')

#################################################################################################
form = st.sidebar.form(key='Optimize', clear_on_submit=False)
# button
placeholder_btn = form.empty()
btn_opt = placeholder_btn.form_submit_button(label='Optimize')
# status
placeholder_status = form.empty()
text_opt_status = placeholder_status.text("Status : ", help=description['status'])
form.divider()

# Parameter setting
# Header
form.header('參數設定')

## Data and Solver setting
exp_param_1 = form.expander('資料與求解參數')#, expanded=True
# sb_data_freq = exp_param_1.selectbox(label='操作週期', options=[5, 10, 15, 30, 60], index=4, help=description['data_freq'])
sb_data_freq = exp_param_1.selectbox(label='操作週期', options=[60], index=0, help=description['data_freq'])
input_max_sec = exp_param_1.number_input(label='求解時間上限(seconds)', value=5, step=1, help=description['max_sec'])

## Price-related setting
exp_param_2 = form.expander('電力價格相關')
input_c_cap = exp_param_2.number_input(label='契約容量(kWh)', value=7000, step=100, help=description['c_cap'])
input_basic_tariff_per_kwh = exp_param_2.number_input(label='基本時間電價(kWh)', value=223.60, step=0.1, format="%.2f", help=description['basic_tariff_per_kwh'])
cb_summer = exp_param_2.checkbox(label='是否使用夏季電價', value=True, help=description['summer'])

## ESS-related setting
exp_param_3 = form.expander('儲能系統相關')
input_e_cap = exp_param_3.number_input(label='機組容量(kWh)', value=1300, step=100, help=description['e_cap'])
input_ub = exp_param_3.number_input(label='SOC上限(%)', value=95, step=1, min_value=0, max_value=100, help=description['ub'])
input_lb= exp_param_3.number_input(label='SOC下限(%)', value=5, step=1, min_value=0, max_value=100, help=description['lb'])
input_soc_init = exp_param_3.number_input(label='初始SOC(%)', value=95, step=1, min_value=0, max_value=100, help=description['soc_init'])
cb_opt_soc_init = exp_param_3.checkbox(label='系統建議初始SOC', value=False, help=description['opt_soc_init'])
input_soc_end = exp_param_3.number_input(label='結束SOC(%)', value=20, step=1, min_value=0, max_value=100, help=description['soc_end'])
cb_opt_soc_end = exp_param_3.checkbox(label='系統建議結束SOC', value=False, help=description['opt_soc_end'])
input_ess_degradation_cost_per_kwh_discharged = exp_param_3.number_input(label='放電耗損成本(每kWh)', value=10.00, step=1.0, format="%.2f", 
                                                                         help=description['ess_degradation_cost_per_kwh_discharged'])

## Production-related setting
exp_param_4 = form.expander('生產相關')
input_factory_profit_per_kwh = exp_param_4.number_input(label='工廠生產利潤(每使用一度電)', value=5.00, step=1.0, format="%.2f", 
                                                        help=description['factory_profit_per_kwh'])

## Trading-related setting
exp_param_5 = form.expander('輔助服務投標相關')
# input_tendered_cap = exp_param_5.number_input(label='投標容量(kWh)', value=1200, step=100, help=description['tendered_cap'])
# input_clearing_price_per_mwh = exp_param_5.number_input(label='日前即時備轉容量結清價格(每mWh)', value=350.00, step=5.0, format="%.2f", 
                                                        # help=description['clearing_price_per_mwh'])
input_exec_rate = exp_param_5.number_input(label='執行率(%)', value=95, step=1, min_value=0, max_value=100, help=description['exec_rate'])
sb_effectiveness_level = exp_param_5.selectbox(label='效能級數', options=[0, 1, 2, 3], index=1, help=description['effectiveness_level'])
# input_DA_margin_price_per_mwh = exp_param_5.number_input(label='日前電能邊際價格(每mWh)', value=4757.123, step=0.25, format="%.3f", 
                                                         # help=description['DA_margin_price_per_mwh'])
# input_dispatch_ratio = exp_param_5.number_input(label='預估調度比例(%)', value=60, step=1, min_value=0, max_value=100, help=description['dispatch_ratio'])
## Scenario setting
exp_param_6 = form.expander('投標情境設定')
cb_opt_bid = exp_param_6.checkbox(label='系統建議投標時段', value=True, help=description['opt_bid'])
cb_opt_tendered_cap = exp_param_6.checkbox(label='系統建議投標容量', value=True, help=description['opt_tendered_cap'])
cb_relax_tendered_step = exp_param_6.checkbox(label='放鬆投標單位限制', value=False, help=description['relax_tendered_step'])
input_tendered_ub = exp_param_6.number_input(label='投標容量上限(mWh)', value=1.200, step=0.100, min_value=0.000, max_value=100.000, format="%.3f", 
                                             help=description['tendered_ub'])
input_tendered_lb = exp_param_6.number_input(label='投標容量下限(mWh)', value=1.000, step=0.100, min_value=1.000, max_value=100.000, format="%.3f", 
                                             help=description['tendered_lb'])
ed_bid_init = pd.DataFrame({'bid': [False]*24, 'win': [True]*24, 'dispatch': [True]*24, 
                            'tendered_cap(mWh)': [1.2]*24, 'dispatch_ratio(%)': [60]*24, 
                            'clearing_price(mWh)': [350.00]*24, 'marginal_price(mWh)': [4757.123]*24}) #, 'time':range(24)
# ed_bid_init = ed_bid_init.set_index('time')
txt_ed_bid = exp_param_6.text('情境設定表格', help=description['txt_ed_bid'])
ed_bid = exp_param_6.data_editor(ed_bid_init, use_container_width=True)

## Transmission-related setting
exp_param_7 = form.expander('電力輸送相關')
input_limit_g_es_p = exp_param_7.number_input(label='儲能機組充電功率上限(kW)', value=1500, step=100, help=description['limit_g_es_p'])
input_limit_es_p = exp_param_7.number_input(label='儲能機組放電功率上限(kW)', value=1500, step=100, help=description['limit_es_p'])
input_limit_g_p = exp_param_7.number_input(label='電網輸入功率上限(kW)', value=10000, step=100, help=description['limit_g_p'])
input_limit_pv_p = exp_param_7.number_input(label='太陽能機組輸送功率上限(kW)', value=10000, step=100, help=description['limit_pv_p'])
input_loss_coef = exp_param_7.number_input(label='電能輸送損失係數', value=0.946, step=0.05, min_value=0.0, max_value=1.0, format="%.3f", help=description['loss_coef'])

## PV-related setting
exp_param_8 = form.expander('太陽能發電機組相關')
input_bulk_tariff_per_kwh = exp_param_8.number_input(label='太陽能躉售電價(每kWh)', value=4.8816, step=0.25, format="%.4f", help=description['bulk_tariff_per_kwh'])

######################################################################
######################################################################

# load file upload
if upload_load is not None:
    col_upload[0].subheader(upload_load.name)
    try:
        # read data
        df = pd.read_csv(upload_load)
        df['value'] = df['value']
        # save data to session state
        st.session_state['data_load'] = df.copy()
        df = df.set_index('time')
        
        # # create expander container
        # expander = exp_load.expander("See Table and Figure")
        # col_upload[0].markdown("檢視圖表")
        
        # fig
        fig_load = make_data_plot(df, title='')
        col_upload[0].plotly_chart(fig_load, use_container_width=True)
        
        # translation
        df = df.rename(columns={'time':'時間', 'value':'負載量(kWh)'})
        # table
        col_upload[0].dataframe(df, use_container_width=True)
        
    except Exception as e:
        col_upload[0].write(f'Error while reading file: {upload_load.name}, {e}')
else:
    st.session_state['data_load'] = None
    
# PV file upload
if upload_pv is not None:
    col_upload[1].subheader(upload_pv.name)
    try:
        # read data
        df = pd.read_csv(upload_pv)
        # save data to session state
        st.session_state['data_pv'] = df.copy()
        df = df.set_index('time')
        
        # # create expander container
        # expander = exp_pv.expander("See Table and Figure")
        # col_upload[1].markdown("檢視圖表")
        
        # fig
        fig_pv = make_data_plot(df, title='')
        col_upload[1].plotly_chart(fig_pv, use_container_width=True)
        
        # translation
        df = df.rename(columns={'time':'時間', 'value':'發電量(kWh)'})
        # table
        col_upload[1].dataframe(df, use_container_width=True)
        
    except Exception as e:
        col_upload[1].write(f'Error while reading file: {upload_pv.name}, {e}')
else:
    st.session_state['data_pv'] = None

if btn_opt or st.session_state['optimization_count'] > 0:
    with st.spinner('ZzZZzzz...'):
        # get input data
        try:
            # df_load = pd.read_csv(st.session_state['data_load'], engine = "python")
            # df_pv = pd.read_csv(st.session_state['data_pv'], engine = "python")
            df_load = st.session_state['data_load']
            df_pv = st.session_state['data_pv']

        except Exception as e:
            st.write(f'Error while loading file: {e}')

        ## verify settings
        placeholder_warning = st.empty()
        
        # SOC setting
        if not (input_lb < input_ub and input_lb <= input_soc_init <= input_ub and input_lb <= input_soc_end <= input_ub):
            placeholder_warning.warning('Check SOC boundary setting.', icon="⚠️")
            st.stop()
        
        # trading-related rule
        if not verify_bd_rule(ed_bid, opt_bid=cb_opt_bid):
            placeholder_warning.warning('Check trading senario setting is correct.', icon="⚠️")
            st.stop()
        # tendered capacity
        if not cb_opt_tendered_cap:
            if not all([verify_tendered_capacity_integrity(ed_bid, relax=cb_relax_tendered_step), 
                        verify_tendered_capacity_in_bound(ed_bid, lb=input_tendered_lb, ub=input_tendered_ub), 
                        verify_tendered_capacity_non_negative(ed_bid)]):
                placeholder_warning.warning('Check tendered capacity setting is correct.(non-negativity / integrity / not in bound)', icon="⚠️")
                st.stop()
        
        # optimize
        # try:
        # update status on web interface
        btn_opt = placeholder_btn.form_submit_button(label='Optimizing...')
        # optimize
        status, df_report, df_result = optimize(df_load=df_load, df_pv=df_pv, max_sec=input_max_sec, data_freq=sb_data_freq, 
                                                c_cap=input_c_cap, basic_tariff_per_kwh=input_basic_tariff_per_kwh, summer=cb_summer, e_cap=input_e_cap, 
                                                soc_init=input_soc_init, opt_soc_init=cb_opt_soc_init, soc_end=input_soc_end, opt_soc_end=cb_opt_soc_end, 
                                                lb=input_lb, ub=input_ub, 
                                                ess_degradation_cost_per_kwh_discharged=input_ess_degradation_cost_per_kwh_discharged, 
                                                factory_profit_per_kwh=input_factory_profit_per_kwh, 
                                                tendered_cap=ed_bid['tendered_cap(mWh)'], clearing_price_per_mwh=ed_bid['clearing_price(mWh)'], 
                                                exec_rate=input_exec_rate, effectiveness_level=sb_effectiveness_level, 
                                                DA_margin_price_per_mwh=ed_bid['marginal_price(mWh)'], 
                                                dispatch_ratio=ed_bid['dispatch_ratio(%)'], 
                                                opt_bid=cb_opt_bid, opt_tendered_cap=cb_opt_tendered_cap, relax_tendered_step=cb_relax_tendered_step, 
                                                tendered_ub=input_tendered_ub, tendered_lb=input_tendered_lb, 
                                                bid=ed_bid['bid'].tolist(), bid_win=ed_bid['win'].tolist(), dispatch=ed_bid['dispatch'].tolist(), 
                                                limit_g_es_p=input_limit_g_es_p, 
                                                limit_es_p=input_limit_es_p, 
                                                limit_g_p=input_limit_g_p, 
                                                limit_pv_p=input_limit_pv_p, 
                                                loss_coef=input_loss_coef, bulk_tariff_per_kwh=input_bulk_tariff_per_kwh)

        # set dataframe index
        df_report = df_report.set_index('項目')
        df_result = df_result.set_index('time')

        # update session states
        st.session_state['optimization_status'] = status
        st.session_state['optimization_count'] += 1

        # create container
        exp_opt = st.expander("檢視最佳化結果與報表", expanded=True)

        # grpah
        exp_opt.subheader('最佳化排程圖表')
        fig_result = make_result_plot(df_result, title='')# , secondary_y_limit=[0,input_tendered_cap]
        exp_opt.plotly_chart(fig_result, use_container_width=True)

        # create column
        col_opt = exp_opt.columns((2,4))

        # report
        col_opt[0].markdown('<p style="font-family:Source Sans Pro, sans-serif; font-size: 1.5rem; font-weight: bold;">收益報表</p>', unsafe_allow_html=True)
        col_opt[0].dataframe(df_report, use_container_width=True)

        # operational result
        col_opt[1].markdown('<p style="font-family:Source Sans Pro, sans-serif; font-size: 1.5rem; font-weight: bold;">最佳化排程</p>', unsafe_allow_html=True)
        col_opt[1].dataframe(df_result, use_container_width=True)

        # update status
        # btn
        # placeholder_btn.empty()
        btn_opt = placeholder_btn.form_submit_button(label=' Optimize ')

        # status text
        # st.write(123)
        placeholder_status.empty()
        text_opt_status = placeholder_status.text(f'Status : {status}', help=description['status'])
        # developing text
        st.caption(f"Opimization count : {st.session_state['optimization_count']}")
        
        # except Exception as e:
        #     btn_opt = placeholder_btn.form_submit_button(label=' Optimize ')
        #     st.write(f'Error while optimizing: {e}')