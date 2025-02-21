import os
import io
import json
import base64
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from mip import Model, xsum, BINARY, INTEGER, CONTINUOUS, minimize, maximize, OptimizationStatus
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# set layout
st.set_page_config(page_title='Power Optimizer test(Cht)', layout="wide", page_icon='./img/favicon.png')
st.markdown("<style>.row-widget.stButton {text-align: center;}</style>", unsafe_allow_html=True)

class DataService:
    """Handles data loading and caching operations"""

    @staticmethod
    @st.cache_resource
    def load_sample_data(data_dir: str = "./data") -> Dict[str, pd.DataFrame]:
        """Load and cache data from CSV files"""
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        sample_data_ref = {
            f.replace(".csv", ""): pd.read_csv(os.path.join(data_dir, f)) for f in files
            }
        return sample_data_ref

    @staticmethod
    @st.cache_resource
    def load_default_params(data_dir: str = "./data") -> Dict[str, str]:
        """Load and cache data from json files"""
        with open(os.path.join(data_dir, 'params.json'), 'r') as file:
            params = json.load(file)

        # read json data as dataframe
        params["ed_bid"]["data"] = pd.DataFrame.from_dict(params["ed_bid"]["data"]).copy()

        return params


class ModelBuilder:
    def __init__(self):
        self.model = Model()
        pass

    def apply_config(self, data, params: dict):
        pass

    def add_vars(self):
        pass

    def add_constraints(self):
        pass

    def add_objectives(self):
        pass

    def build(self):
        # self.add_vars()
        # self.add_objectives()
        # self.add_constraints()
        return self.model

class ESSModelBuilder(ModelBuilder):
    def __init__(self, data, params: dict):
        super().__init__()

        ### load default data
        self.df_load = data["load"]
        self.df_pv = data["pv"]

        ### load default params
        self.params = params

    def apply_config(self, data, params: dict):
        # apply parameter and data
        self.data = data
        self.params = params

        df_load = data["load"]
        df_pv = data["pv"]

        self.data_freq = params["sb_data_freq"]["options"][params["sb_data_freq"]["index"]]
        self.max_sec = params["input_max_sec"]["value"]
        self.c_cap = params["input_c_cap"]["value"]
        self.basic_tariff_per_kwh = params["input_basic_tariff_per_kwh"]["value"]
        self.summer = params["cb_summer"]["value"]
        self.e_cap = params["input_e_cap"]["value"]
        self.soc_init = params["input_soc_init"]["value"]
        self.opt_soc_init = params["cb_opt_soc_init"]["value"]
        self.soc_end = params["input_soc_end"]["value"]
        self.opt_soc_end = params["cb_opt_soc_end"]["value"]
        self.lb = params["input_lb"]["value"]
        self.ub = params["input_ub"]["value"]
        self.ess_degradation_cost_per_kwh_discharged = params["input_ess_degradation_cost_per_kwh_discharged"]["value"]
        self.factory_profit_per_kwh = params["input_factory_profit_per_kwh"]["value"]
        self.tendered_cap = params["ed_bid"]["data"]["tendered_cap(mWh)"]
        self.clearing_price_per_mwh = params["ed_bid"]["data"]["clearing_price(mWh)"]
        self.exec_rate = params["input_exec_rate"]["value"]
        self.effectiveness_level = params["sb_effectiveness_level"]["options"][params["sb_effectiveness_level"]["index"]]
        self.DA_margin_price_per_mwh = params["ed_bid"]["data"]["marginal_price(mWh)"]
        self.dispatch_ratio = params["ed_bid"]["data"]["dispatch_ratio(%)"]
        self.opt_bid = params["cb_opt_bid"]["value"]
        self.opt_tendered_cap = params["cb_opt_tendered_cap"]["value"]
        self.relax_tendered_step = params["cb_relax_tendered_step"]["value"]
        self.tendered_lb = params["input_tendered_lb"]["value"]
        self.tendered_ub = params["input_tendered_ub"]["value"]
        self.bid = params["ed_bid"]["data"]["bid"].tolist()
        self.bid_win = params["ed_bid"]["data"]["win"].tolist()
        self.dispatch = params["ed_bid"]["data"]["dispatch"].tolist()
        self.limit_g_es_p = params["input_limit_g_es_p"]["value"]
        self.limit_es_p = params["input_limit_es_p"]["value"]
        self.limit_g_p = params["input_limit_g_p"]["value"]
        self.limit_pv_p = params["input_limit_pv_p"]["value"]
        self.loss_coef = params["input_loss_coef"]["value"]
        self.bulk_tariff_per_kwh = params["input_bulk_tariff_per_kwh"]["value"]

        ### retrieve info from input data
        self.consecutive_n = int(60/self.data_freq)
        self.n = int(len(df_load)/(self.data_freq/5)) # number of time window
        # index
        self.index = df_load['time'].iloc[::int(self.data_freq/5)].values.flatten()
        # Load
        self.load = df_load['value'].iloc[::int(self.data_freq/5)].values.flatten()
        # PV
        self.pv = df_pv['value'].iloc[::int(self.data_freq/5)].values.flatten()

        ### energy charging rate (111/7)
        if self.summer:
            # summer charging rate
            p1 = np.array([1.58]*int(n*(15/48))) # 0000-0730
            p2 = np.array([3.54]*int(n*(5/48))) # 0730-1000
            p3 = np.array([5.31]*int(n*(4/48))) # 1000-1200
            p4 = np.array([3.54]*int(n*(2/48))) # 1200-1300
            p5 = np.array([5.31]*int(n*(8/48))) # 1300-1700
            p6 = np.array([3.54]*int(n*(11/48))) # 1700-2230
            p7 = np.array([1.58]*int(n*(3/48))) # 2230-0000
            self.price = np.hstack([p1, p2, p3, p4, p5, p6, p7])

            if self.data_freq == 60:
                self.price = np.array([1.58]*7 + # 0000-0700
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
            self.price = np.hstack([p1, p2, p3])

            if self.data_freq == 60:
                self.price = np.array([1.50]*7 + # 0000-0700
                                      [2.32]*1 + # 0700-0800(mixed)
                                      [3.44]*14 + # 0800-2200
                                      [2.32]*1 + # 2200-2300(mixed)
                                      [1.50]*1 ) # 2300-0000

        #####################################
        # ### energy charging rate (112/4)
        # if summer:
        #     # summer charging rate
        #     p1 = np.array([1.91]*int(n*(9/24))) # 0000-0900
        #     p2 = np.array([4.39]*int(n*(7/24))) # 0900-1600
        #     p3 = np.array([7.03]*int(n*(6/24))) # 1600-2200
        #     p4 = np.array([4.39]*int(n*(2/24))) # 2200-0000
        #     self.price = np.hstack([p1, p2, p3, p4])
        # else:
        #     # other
        #     p1 = np.array([1.75]*int(n*(6/24))) # 0000-0600
        #     p1 = np.array([4.11]*int(n*(5/24))) # 0600-1100
        #     p1 = np.array([1.75]*int(n*(3/24))) # 1100-1400
        #     p1 = np.array([4.11]*int(n*(10/24))) # 1400-0000
        #     self.price = np.hstack([p1, p2, p3, p4])
        #####################################

        ### prepare initial values for parameters and auxiliary variables
        # Multiplication factor for penalty charge
        self.dummy_penalty_coef_1 = 2
        self.dummy_penalty_coef_2 = 3

        ### ESS
        # unit conversion
        self.soc_init /= 100
        self.soc_end /= 100
        self.lb /= 100
        self.ub /= 100

        # init/end SOC
        self.e_init = self.e_cap*self.soc_init
        self.e_end = self.e_cap*self.soc_end

        # ESS boundary
        self.soc_lb = np.array([self.e_cap*self.lb for i in range(self.n)])
        self.soc_ub = np.array([self.e_cap*self.ub for i in range(self.n)])

        ### Trading params
        self.tendered_cap = [v*10 for v in self.tendered_cap] # temporarily converted to integer level for usage of INTEGER variable type. Ex: 1.2 mWh --> 12.0
        self.service_quality_index = get_service_quality_index(self.exec_rate)
        self.effectiveness_price_per_kwh = get_effectiveness_price(self.effectiveness_level)/1000
        self.clearing_price_per_kwh = [v/1000 for v in self.clearing_price_per_mwh]
        self.DA_margin_price_per_kwh = [v/1000 for v in self.DA_margin_price_per_mwh]
        self.dispatch_ratio = [v/100 for v in self.dispatch_ratio]

        ### other
        # big M for penalty
        self.M = 1e+15

    def _add_var(self):
        ### retrieve constants and data length
        n = self.n

        ### set decision variables
        # set objectives
        self.revenue = self.model.add_var(name='revenue', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        self.income = self.model.add_var(name='income', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        self.cost = self.model.add_var(name='cost', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        # power during the time interval
        self.p_g_f = [self.model.add_var(name=f"power_from_grid_to_factory_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        self.p_es_f = [self.model.add_var(name=f"power_from_ESS_to_factory_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        self.p_pv_f = [self.model.add_var(name=f"power_from_PV_to_factory_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        self.p_pv_es = [self.model.add_var(name=f"power_from_PV_to_ESS_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)] # the power from PV will be served to the factory first
        self.p_pv_g = [self.model.add_var(name=f"power_from_PV_to_grid_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)] # (躉售)the power from PV will be served to the factory first
        self.p_g_es = [self.model.add_var(name=f"power_from_grid_to_ESS_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]

        # total power from grid (aux for visualization)
        self.total_g = [self.model.add_var(name=f'total_power_from_grid_t{i}', var_type=CONTINUOUS) for i in range(n)]
        self.total_g_f = [self.model.add_var(name=f'total_power_from_grid_to_factory_t{i}', var_type=CONTINUOUS) for i in range(n)]
        self.total_g_es = [self.model.add_var(name=f'total_power_from_grid_to_ESS_t{i}', var_type=CONTINUOUS) for i in range(n)]

        # ESS SOC "at the beginning" of the time interval
        self.es = [self.model.add_var(name=f"ESS_SOC_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]

        # excessive load
        self.b_exceed = [self.model.add_var(name=f"if_exceed_110%_cap_at_t{i}", var_type=BINARY) for i in range(n)]
        self.dummy_g_1 = [self.model.add_var(name=f"dummy_power_1_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        self.dummy_g_2 = [self.model.add_var(name=f"dummy_power_2_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        self.dummy_g_f = [self.model.add_var(name=f"dummy_power_from_grid_to_factory_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        self.dummy_g_es = [self.model.add_var(name=f"dummy_power_from_grid_to_ESS_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]

        # ESS charging/discharging status (or)
        self.b_chg = [self.model.add_var(name=f"ESS_is_charging_at_t{i}", var_type=BINARY) for i in range(n)]
        self.b_dch = [self.model.add_var(name=f"ESS_is_discharging_at_t{i}", var_type=BINARY) for i in range(n)]
        self.aux_p_g_es = [self.model.add_var(name=f"aux_power_from_grid_to_ESS_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        self.aux_p_es_f = [self.model.add_var(name=f"aux_power_from_ESS_to_factory_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        self.aux_dummy_g_es = [self.model.add_var(name=f"aux_dummy_power_from_grid_to_ESS_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]

        # total excessive power (aux for visualization)
        self.total_dummy = [self.model.add_var(name=f'total_excessive_power_t{i}', var_type=CONTINUOUS) for i in range(n)]
        self.total_flow_es = [self.model.add_var(name=f'total_power_flow_of_ESS_t{i}', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf')) for i in range(n)]

        # dummies for penalty calculation
        self.q_1 = self.model.add_var(name=f"max_dummy_power_1", var_type=CONTINUOUS)
        self.q_2 = self.model.add_var(name=f"max_dummy_power_2", var_type=CONTINUOUS)
        self.b_max_aux_1 = [self.model.add_var(name=f"max_func_aux_1_t{i}", var_type=BINARY) for i in range(n)]
        self.b_max_aux_2 = [self.model.add_var(name=f"max_func_aux_2_t{i}", var_type=BINARY) for i in range(n)]

        # bidding decision
        self.bid = [1 if v else 0 for v in self.bid]
        self.bid_win = [1 if v else 0 for v in self.bid_win]
        self.dispatch = [1 if v else 0 for v in self.dispatch]
        if self.opt_bid:
            self.bid = [self.model.add_var(name=f"if_bid_at_t{i}", var_type=BINARY) for i in range(n)]

        # tendered capacity
        if self.opt_tendered_cap:
            if self.relax_tendered_step:
                self.tendered_cap = [self.model.add_var(name=f"tendered_cap_at_t{i}", var_type=CONTINUOUS) for i in range(n)]
            else:
                self.tendered_cap = [self.model.add_var(name=f"tendered_cap_at_t{i}", var_type=INTEGER) for i in range(n)]

        # for multiplication of tendered capacity and bidding decision
        # aux_tendered_cap = [self.model.add_var(name=f"aux_tendered_cap_at_t{i}", lb=float('-Inf'), ub=float('Inf'), var_type=CONTINUOUS) for i in range(n)]
        if self.relax_tendered_step:
            self.aux_tendered_cap = [self.model.add_var(name=f"aux_tendered_cap_at_t{i}", lb=float('-Inf'), ub=float('Inf'), var_type=CONTINUOUS) for i in range(n)]
        else:
            self.aux_tendered_cap = [self.model.add_var(name=f"aux_tendered_cap_at_t{i}", lb=float('-Inf'), ub=float('Inf'), var_type=INTEGER) for i in range(n)]

        return self

    def _set_constraints(self):
        ### retrieve constants and data length
        n = self.n
        M = self.M

        ### add constraints
        ## basic constraints
        # set linear constraints for multiplication of decision variables
        for i in range(n):
            # either charging or discharging
            self.model.add_constr(self.b_chg[i] + self.b_dch[i] <= 1)
            self.model.add_constr(self.aux_p_g_es[i] <= self.p_g_es[i])
            self.model.add_constr(self.aux_p_g_es[i] <= M * self.b_chg[i])
            self.model.add_constr(self.aux_p_g_es[i] >= self.p_g_es[i] + M * (self.b_chg[i]-1))
            self.model.add_constr(self.aux_dummy_g_es[i] <= self.dummy_g_es[i])
            self.model.add_constr(self.aux_dummy_g_es[i] <= M * self.b_chg[i])
            self.model.add_constr(self.aux_dummy_g_es[i] >= self.dummy_g_es[i] + M * (self.b_chg[i]-1))
            self.model.add_constr(self.aux_p_es_f[i] <= self.p_es_f[i])
            self.model.add_constr(self.aux_p_es_f[i] <= M * self.b_dch[i])
            self.model.add_constr(self.aux_p_es_f[i] >= self.p_es_f[i] + M * (self.b_dch[i]-1))
            # tendered capacity and bidding decision
            if self.opt_tendered_cap:
                self.model.add_constr(self.aux_tendered_cap[i] >= 0) # just ensure
                self.model.add_constr(self.aux_tendered_cap[i] >= self.tendered_cap[i] - M * (1-self.bid[i]))
                self.model.add_constr(self.aux_tendered_cap[i] <= M * self.bid[i])
                self.model.add_constr(self.aux_tendered_cap[i] <= self.tendered_cap[i])
            else:
                self.model.add_constr(self.aux_tendered_cap[i] == self.tendered_cap[i]*self.bid[i])

        # non-negative
        for i in range(n):
            self.model.add_constr(self.p_g_f[i] >= 0.0)
            self.model.add_constr(self.p_es_f[i] >= 100*self.aux_tendered_cap[i]*self.bid_win[i]*self.dispatch[i]*self.dispatch_ratio[i])
            self.model.add_constr(self.aux_p_es_f[i] >= 100*self.aux_tendered_cap[i]*self.bid_win[i]*self.dispatch[i]*self.dispatch_ratio[i])
            self.model.add_constr(self.p_pv_f[i] >= 0.0)
            self.model.add_constr(self.p_pv_es[i] >= 0.0)
            self.model.add_constr(self.p_pv_g[i] >= 0.0)
            self.model.add_constr(self.p_g_es[i] >= 0.0)
            self.model.add_constr(self.aux_p_g_es[i] >= 0.0)
            self.model.add_constr(self.dummy_g_f[i] >= 0.0)
            self.model.add_constr(self.dummy_g_es[i] >= 0.0)
            self.model.add_constr(self.aux_dummy_g_es[i] >= 0.0)

        ## maximum function of dummy variables, for panelty calculation
        for i in range(n):
            self.model.add_constr(self.q_1 >= self.dummy_g_1[i])
            self.model.add_constr(self.q_1 <= self.dummy_g_1[i] + M * self.b_max_aux_1[i])
            self.model.add_constr(self.q_2 >= self.dummy_g_2[i])
            self.model.add_constr(self.q_2 <= self.dummy_g_2[i] + M * self.b_max_aux_2[i])
        self.model.add_constr( xsum( self.b_max_aux_1[i] for i in range(n) ) <= n-1 )
        self.model.add_constr( xsum( self.b_max_aux_2[i] for i in range(n) ) <= n-1 )

        ## factory
        # load
        for i in range(n):
            self.model.add_constr(self.dummy_g_f[i] + self.p_g_f[i] + self.loss_coef*(self.aux_p_es_f[i] + self.p_pv_f[i]) == self.load[i])
        # grid contract boundary (penalty for excessive capacity are added later with dummy vars.)
        for i in range(n):
            self.model.add_constr(self.p_g_f[i] + self.p_pv_f[i] - self.p_pv_g[i] <= self.c_cap - 100*self.aux_tendered_cap[i]*self.bid_win[i]*self.dispatch[i]*self.dispatch_ratio[i]) ############################ 全額躉售計費修改

        ## dispatch
        # 1. sum of dispatch_start <= 1 in any arbitrary 3 consecutive hours
        if self.opt_bid:
            for i in range(n-3*self.consecutive_n):
                self.model.add_constr( xsum(self.bid[j]*self.bid_win[j]*self.dispatch[j] for j in range(i, i+3*self.consecutive_n)) <= 1 )

        ## bidding
        # bounds
        for i in range(n):
            self.model.add_constr(self.aux_tendered_cap[i] >= 10*self.tendered_lb*self.bid[i])
            self.model.add_constr(self.aux_tendered_cap[i] <= 10*self.tendered_ub*self.bid[i])


        ## ESS
        # init.
        if not self.opt_soc_init:
            self.model.add_constr(self.es[0] == self.e_init)
        # ending SOC lb
        if not self.opt_soc_end:
            self.model.add_constr(self.es[-1] >= self.e_end)

        # output capacity limitation
        for i in range(n):
            self.model.add_constr(self.aux_p_es_f[i] <= self.es[i])
            self.model.add_constr(self.p_es_f[i] <= self.es[i])
            # self.model.add_constr(p_es_f[i] <= es[i])
        # update
        for i in range(1,n):
            self.model.add_constr(self.es[i] == self.es[i-1] + (self.aux_dummy_g_es[i-1] + self.aux_p_g_es[i-1] + self.p_pv_es[i-1] - self.aux_p_es_f[i-1])/self.consecutive_n)
        # SOC boundary
        for i in range(n):
            self.model.add_constr(self.es[i] >= self.soc_lb[i])
            self.model.add_constr(self.es[i] <= self.soc_ub[i])

        # print(e_init)
        # print(soc_lb[i], soc_ub[i])
        ## PV
        # flow balance
        for i in range(n):
            self.model.add_constr((self.p_pv_f[i] + self.p_pv_es[i] + self.p_pv_g[i]) == self.pv[i])
        # serving priority
        for i in range(n):
            self.model.add_constr(self.p_pv_f[i] >= self.p_pv_g[i])

        ## split excessive power for additional tariff calculation
        for i in range(n):
            self.model.add_constr(0.1*self.c_cap*self.b_exceed[i] <= self.dummy_g_1[i])
            self.model.add_constr(self.dummy_g_1[i] <= 0.1*self.c_cap)
            self.model.add_constr(self.dummy_g_2[i] >= 0)
            self.model.add_constr(self.dummy_g_2[i] <= self.b_exceed[i]*M)
            self.model.add_constr(self.dummy_g_1[i] + self.dummy_g_2[i] == self.dummy_g_f[i] + self.aux_dummy_g_es[i])

        ## transfer limitation
        for i in range(n):
            self.model.add_constr(self.p_g_f[i] <= self.limit_g_p)
            self.model.add_constr(self.p_es_f[i] <= self.limit_es_p)
            self.model.add_constr(self.aux_p_es_f[i] <= self.limit_es_p)
            self.model.add_constr(self.p_pv_f[i] <= self.limit_pv_p)

            self.model.add_constr(self.p_pv_es[i] <= self.limit_pv_p)
            # self.model.add_constr(p_pv_es[i] <= limit_g_es_p)

            self.model.add_constr(self.p_pv_g[i] <= self.limit_pv_p)

            self.model.add_constr(self.p_g_es[i] <= self.limit_g_es_p)
            self.model.add_constr(self.aux_p_g_es[i] <= self.limit_g_es_p)

            self.model.add_constr(self.dummy_g_f[i] <= self.limit_g_p)
            self.model.add_constr(self.dummy_g_es[i] <= self.limit_g_es_p)
            self.model.add_constr(self.dummy_g_es[i] <= self.limit_g_p)
            self.model.add_constr(self.aux_dummy_g_es[i] <= self.limit_g_es_p)
            # self.model.add_constr(aux_dummy_g_es[i] <= limit_g_p)

            ### Other given condition and constraints for aux var.
            # no power from PV to Grid/ESS directly
            for i in range(n):
                self.model.add_constr(self.p_pv_es[i] == 0) #### 無饋線
                self.model.add_constr(self.p_pv_g[i] == 0) #### 目前為全額躉售

            # total power from grid (aux for visualization)
            for i in range(n):
                self.model.add_constr(self.total_g_f[i] == self.p_g_f[i] + self.dummy_g_f[i])
                self.model.add_constr(self.total_g_es[i] == self.aux_p_g_es[i] + self.aux_dummy_g_es[i])
                self.model.add_constr(self.total_g[i] == self.total_g_f[i] + self.total_g_es[i])

            # total excessive power (aux for visualization)
            for i in range(n):
                self.model.add_constr(self.total_dummy[i] == self.dummy_g_1[i] + self.dummy_g_2[i])
            # total power flow of ESS (aux for visualization)
            for i in range(n):
                self.model.add_constr(self.total_flow_es[i] == self.total_g_es[i] + self.p_pv_es[i] - self.aux_p_es_f[i])

        return self

    def _set_objectives(self):
        ### retrieve constants and data length
        n = self.n
        M = self.M

        #### ensemble objective variables
        ################################################################################
        # dispatch income, 即時備轉收益 = (容量費 + 效能費) × 服務品質指標 ＋ 電能費
        self.dispatch_income = self.model.add_var(name='dispatch_income', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        self.capacity_reserve_income = self.model.add_var(name='capacity_reserve_income', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        self.effectiveness_income = self.model.add_var(name='effectiveness_income', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        self.energy_income = self.model.add_var(name='dispatch_income', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        # 容量費
        self.model.add_constr(self.capacity_reserve_income == xsum( (self.clearing_price_per_kwh[i]*100*self.aux_tendered_cap[i]*self.bid_win[i]) for i in range(n) ))
        # 效能費
        self.model.add_constr(self.effectiveness_income == xsum( (self.effectiveness_price_per_kwh*100*self.aux_tendered_cap[i]*self.bid_win[i]*self.dispatch[i]) for i in range(n) ))
        # 電能費
        self.model.add_constr(self.energy_income == xsum( (self.DA_margin_price_per_kwh[i]*100*self.aux_tendered_cap[i]*self.bid_win[i]*self.dispatch[i]*self.dispatch_ratio[i]) for i in range(n) ))
        # total
        self.model.add_constr(self.dispatch_income == ((self.capacity_reserve_income+self.effectiveness_income)*self.service_quality_index + self.energy_income)/self.consecutive_n)

        # factory income
        factory_income = self.model.add_var(name='factory_income', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        self.model.add_constr(factory_income == xsum( self.factory_profit_per_kwh*load[i]/self.consecutive_n for i in range(n) ))

        # PV income
        pv_income = self.model.add_var(name='PV_income', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        self.model.add_constr(pv_income == xsum( self.bulk_tariff_per_kwh*(self.p_pv_g[i]+self.p_pv_f[i])/self.consecutive_n for i in range(n) )) ############################ 全額躉售計費修改

        # total income
        self.model.add_constr(self.income == (self.dispatch_income + factory_income + pv_income))

        # fixed eletricity tariff
        fixed_e_cost = self.model.add_var(name='fixed_eletricity_tariff', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        self.model.add_constr(fixed_e_cost == self.basic_tariff_per_kwh*(1*self.c_cap + self.dummy_penalty_coef_1*self.q_1 + self.dummy_penalty_coef_2*self.q_2)/30) ############################ 全額躉售計費修改

        # usage eletricity tariff
        usage_e_cost = self.model.add_var(name='usage_eletricity_tariff', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        self.model.add_constr(usage_e_cost == xsum( self.price[i]*(self.p_g_f[i] + self.p_pv_f[i] - self.p_pv_g[i] + self.aux_p_g_es[i] + self.dummy_g_1[i] + self.dummy_g_2[i])/self.consecutive_n for i in range(n) )) ############################ 全額躉售計費修改

        # ESS degradation
        ess_dis_cost = self.model.add_var(name='ess_discharging_degradation_cost', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        self.model.add_constr(ess_dis_cost == xsum( self.ess_degradation_cost_per_kwh_discharged*self.aux_p_es_f[i]/self.consecutive_n for i in range(n) ))

        # total cost
        self.model.add_constr(self.cost == (fixed_e_cost + usage_e_cost + ess_dis_cost))

        # total revenue
        self.model.add_constr(self.revenue == (self.income - self.cost))
        self.model.objective = maximize(self.revenue)
        return self

    def build(self):
        self._add_var()
        self._set_constraints()
        self._set_objectives()
        return self.model

class Optimizer:
    def __init__(self, data_service: DataService, model_builder: ModelBuilder):
        self.data_service = data_service
        self.model_builder = model_builder

    def add(self):
        pass

    def optimize(self):
        pass
