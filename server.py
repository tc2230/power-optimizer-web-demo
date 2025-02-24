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

def verify_tendered_capacity_integrity(ed_bid, relax=0):
    if not relax:
        return all([(10*i).is_integer() for i in ed_bid['tendered_cap(mWh)']])
    return True

def verify_tendered_capacity_in_bound(ed_bid, lb=0, ub=float('Inf')):
    return all([lb <= i <= ub for i in ed_bid['tendered_cap(mWh)']])

def verify_tendered_capacity_non_negative(ed_bid):
    return all([i >= 0 for i in ed_bid['tendered_cap(mWh)']])

def verify_bid_rule(df, opt_bid=1):
    if opt_bid:
        return all([bw >= d for bw, d in zip(df['win'], df['dispatch'])])
    else:
        row = all([bw >= d for bw, d in zip(df['win'], df['dispatch'])])
        tmp = df['bid']*df['win']*df['dispatch']
        dispatch = all([tmp[i:i+3].sum() <= 1 for i in range(len(df)-2)])
        return all([row, dispatch])

class DataService:
    """Handles data loading and caching operations"""
    def __init__(self, data_dir: str = "./data", params_filename: str = "default_params.json"):
        self.data_dir = data_dir
        self.params_filename = params_filename

    @staticmethod
    @st.cache_data
    def load_sample_data() -> Dict[str, pd.DataFrame]:
        """Load sample data from CSV files"""

        data_dir = "./data"
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        sample_data_ref = {
            f.replace(".csv", "").replace("sample_", ""): pd.read_csv(os.path.join(data_dir, f)) for f in files
            }
        return sample_data_ref

    @staticmethod
    @st.cache_data
    def load_sample_params() -> Dict[str, str]:
        """Load sample params from json files"""

        data_dir = "./data"
        params_filename = "default_params.json"
        with open(os.path.join(data_dir, params_filename), 'r') as file:
            params = json.load(file)

        # read json data as dataframe
        params["ed_bid"]["data"] = pd.DataFrame.from_dict(params["ed_bid"]["data"]).copy()

        return params

    def load_data(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        """Load data from CSV files"""

        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        sample_data_ref = {
            f.replace(".csv", "").replace("sample_", ""): pd.read_csv(os.path.join(data_dir, f)) for f in files
            }
        return sample_data_ref

    def load_params(self, data_dir: str, params_filename: str) -> Dict[str, str]:
        """Load params from json files"""

        with open(os.path.join(data_dir, params_filename), 'r') as file:
            params = json.load(file)

        # read json data as dataframe
        params["ed_bid"]["data"] = pd.DataFrame.from_dict(params["ed_bid"]["data"]).copy()

        return params

class MIPModelBuilder:
    def __init__(self):
        pass

    def update(self, data, params: dict):
        # update params and aux from session state
        pass

    def _add_vars(self):
        pass

    def _add_constraints(self):
        pass

    def _add_objectives(self):
        pass

    def build(self):
        # init model
        # update params
        # self._add_vars()
        # self._add_objectives()
        # self._add_constraints()
        # return self.model
        pass

class ESSModelBuilder(MIPModelBuilder):
    def __init__(self):
        super().__init__()

    def update(self):
        # apply parameter and data
        df_load = st.session_state["data"]["load"]
        df_pv = st.session_state["data"]["pv"]
        params = st.session_state["params"]

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
            p1 = np.array([1.58]*int(self.n*(15/48))) # 0000-0730
            p2 = np.array([3.54]*int(self.n*(5/48))) # 0730-1000
            p3 = np.array([5.31]*int(self.n*(4/48))) # 1000-1200
            p4 = np.array([3.54]*int(self.n*(2/48))) # 1200-1300
            p5 = np.array([5.31]*int(self.n*(8/48))) # 1300-1700
            p6 = np.array([3.54]*int(self.n*(11/48))) # 1700-2230
            p7 = np.array([1.58]*int(self.n*(3/48))) # 2230-0000
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
            p1 = np.array([1.50]*int(self.n*(15/48))) # 0000-0730
            p2 = np.array([3.44]*int(self.n*(30/48))) # 0730-2230
            p3 = np.array([1.50]*int(self.n*(3/48))) # 2230-0000
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
        self.soc_lb = np.array([self.e_cap*self.lb]*self.n)
        self.soc_ub = np.array([self.e_cap*self.ub]*self.n)

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
        self.model.add_constr(factory_income == xsum( self.factory_profit_per_kwh*self.load[i]/self.consecutive_n for i in range(n) ))

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
        self.model = Model()
        self.update()
        self._add_var()
        self._set_constraints()
        self._set_objectives()
        # return self
        return self

    def optimize(*args, **kwrgs):
        ## return results
        pass

# User --[data, params]--> Optimizer --[data, params]--> Model Builder: build(), optimize() --[result]--> Optimizer --[result, plots]--> User
class UIHandler:
    """Base class for UI styling and components"""
    def __init__(self):
        pass

    def _update_session_state(self, component_name, key, value):
        item = {k: v for k, v in st.session_state[component_name]}
        item[key] = value
        return item

    # set background picture and caption at the top of sidebar
    def set_sidebar_markdown(self, img_path, caption=None):
        with open(img_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")

            st.sidebar.markdown(
                f"""
                <div style="display:table; margin-top:-28%; margin-left:-2%; font-family:'Source Sans Pro', sans-serif; margin-bottom: -1rem; color: rgb(163, 168, 184); font-size: 14px;">
                    <img src="data:image/png;base64,{data}" width="100" height="100">
                    <p style="font-family:'Source Sans Pro', sans-serif; color: rgb(163, 168, 184); font-size: 14px;">
                        A dog saying "THIS IS FINE".
                    </p>
                </div>
                """,
                # f"""
                # <div style="display:table; margin-top:-32%; margin-left:-2%; font-family:'Source Sans Pro', sans-serif; margin-bottom: -1rem; color: rgb(163, 168, 184); font-size: 14px;">
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

    def create_layout(self):
        params = st.session_state["params"]

        ### global streamlit config
        st.set_page_config(page_title='Power Optimizer test(Cht)', layout="wide", page_icon='./img/favicon.png')
        st.markdown("<style>.row-widget.stButton {text-align: center;}</style>", unsafe_allow_html=True) # for button css

        ### Dashboard section
        # page title
        st.title('最佳化工具 Demo')

        # expander for upload field
        exp_upload = st.expander('資料上傳區域', expanded=True)
        exp_upload.markdown('<p style="font-family:Source Sans Pro, sans-serif; font-size: 1.5rem; font-weight: bold;">於此處上傳資料</p>', unsafe_allow_html=True)
        exp_upload.text('目前使用採樣週期為五分鐘的單日資料。')

        col_upload = exp_upload.columns(2)

        # Create data sections
        upload_load = col_upload[0].file_uploader('工廠負載資料')
        upload_power = col_upload[1].file_uploader('太陽能發電資料')

        ### Sidebar section
        # set sidebar picture
        self.set_sidebar_markdown(img_path="./img/thisisfine.png")

        ## form general setting
        form = st.sidebar.form(key='Optimize', clear_on_submit=False)

        # button
        placeholder_btn = form.empty()
        btn_opt = placeholder_btn.form_submit_button(
            label='Optimize',
            on_click=lambda: setattr(st.session_state, "zzzzzz", self._update_session_state("zzzzzz", "zzzzzz", "zzzzzz"))
            )

        # status
        placeholder_status = form.empty()
        text_opt_status = placeholder_status.text(
            body=f"Status: {params["text_opt_status"]["body"]}",
            help=params["text_opt_status"]["help"]
            )
        form.divider()

        ## Parameter setting
        # header
        form.header('參數設定')

        # optimization setting
        exp_param_1 = form.expander('資料與求解參數')#, expanded=True
        sb_data_freq = exp_param_1.selectbox(
            label=params["sb_data_freq"]["label"],
            options=params["sb_data_freq"]["options"],
            index=params["sb_data_freq"]["index"],
            help=params["sb_data_freq"]["help"],
            on_change=None
            )
        input_max_sec = exp_param_1.number_input(
            label=params["input_max_sec"]["label"],
            value=params["input_max_sec"]["value"],
            step=params["input_max_sec"]["step"],
            help=params["input_max_sec"]["help"],
            on_change=None
            )

        # Price-related setting
        exp_param_2 = form.expander('電力價格相關')
        input_c_cap = exp_param_2.number_input(
            label=params["input_c_cap"]["label"],
            value=params["input_c_cap"]["value"],
            step=params["input_c_cap"]["step"],
            help=params["input_c_cap"]["help"],
            on_change=None
            )
        input_basic_tariff_per_kwh = exp_param_2.number_input(
            label=params["input_basic_tariff_per_kwh"]["label"],
            value=params["input_basic_tariff_per_kwh"]["value"],
            step=params["input_basic_tariff_per_kwh"]["step"],
            format=params["input_basic_tariff_per_kwh"]["format"],
            help=params["input_basic_tariff_per_kwh"]["help"],
            on_change=None
            )
        cb_summer = exp_param_2.checkbox(
            label=params["cb_summer"]["label"],
            value=params["cb_summer"]["value"],
            help=params["cb_summer"]["help"],
            on_change=None
            )

        # ESS-related setting
        exp_param_3 = form.expander('儲能系統相關')
        input_e_cap = exp_param_3.number_input(
            label=params["input_e_cap"]["label"],
            value=params["input_e_cap"]["value"],
            step=params["input_e_cap"]["step"],
            help=params["input_e_cap"]["help"],
            on_change=None
            )
        input_ub = exp_param_3.number_input(
            label=params["input_ub"]["label"],
            value=params["input_ub"]["value"],
            step=params["input_ub"]["step"],
            min_value=params["input_ub"]["min_value"],
            max_value=params["input_ub"]["max_value"],
            help=params["input_ub"]["help"],
            on_change=None
            )
        input_lb= exp_param_3.number_input(
            label=params["input_lb"]["label"],
            value=params["input_lb"]["value"],
            step=params["input_lb"]["step"],
            min_value=params["input_lb"]["min_value"],
            max_value=params["input_lb"]["max_value"],
            help=params["input_lb"]["help"],
            on_change=None
            )
        input_soc_init = exp_param_3.number_input(
            label=params["input_soc_init"]["label"],
            value=params["input_soc_init"]["value"],
            step=params["input_soc_init"]["step"],
            min_value=params["input_soc_init"]["min_value"],
            max_value=params["input_soc_init"]["max_value"],
            help=params["input_soc_init"]["help"],
            on_change=None
            )
        cb_opt_soc_init = exp_param_3.checkbox(
            label=params["cb_opt_soc_init"]["label"],
            value=params["cb_opt_soc_init"]["value"],
            help=params["cb_opt_soc_init"]["help"],
            on_change=None
            )
        input_soc_end = exp_param_3.number_input(
            label=params["input_soc_end"]["label"],
            value=params["input_soc_end"]["value"],
            step=params["input_soc_end"]["step"],
            min_value=params["input_soc_end"]["min_value"],
            max_value=params["input_soc_end"]["max_value"],
            help=params["input_soc_end"]["help"],
            on_change=None
            )
        cb_opt_soc_end = exp_param_3.checkbox(
            label=params["cb_opt_soc_end"]["label"],
            value=params["cb_opt_soc_end"]["value"],
            help=params["cb_opt_soc_end"]["help"],
            on_change=None
            )
        input_ess_degradation_cost_per_kwh_discharged = exp_param_3.number_input(
            label=params["input_ess_degradation_cost_per_kwh_discharged"]["label"],
            value=params["input_ess_degradation_cost_per_kwh_discharged"]["value"],
            step=params["input_ess_degradation_cost_per_kwh_discharged"]["step"],
            format=params["input_ess_degradation_cost_per_kwh_discharged"]["format"],
            help=params["input_ess_degradation_cost_per_kwh_discharged"]["help"],
            on_change=None
            )

        # Production-related setting
        exp_param_4 = form.expander('生產相關')
        input_factory_profit_per_kwh = exp_param_4.number_input(
            label=params["input_factory_profit_per_kwh"]["label"],
            value=params["input_factory_profit_per_kwh"]["value"],
            step=params["input_factory_profit_per_kwh"]["step"],
            format=params["input_factory_profit_per_kwh"]["format"],
            help=params["input_factory_profit_per_kwh"]["help"],
            on_change=None
            )

        # Trading-related setting
        exp_param_5 = form.expander('輔助服務投標相關')
        # input_tendered_cap = exp_param_5.number_input(label='投標容量(kWh)', value=1200, step=100, help=description['tendered_cap'])
        # input_clearing_price_per_mwh = exp_param_5.number_input(label='日前即時備轉容量結清價格(每mWh)', value=350.00, step=5.0, format="%.2f", help=description['clearing_price_per_mwh'])
        input_exec_rate = exp_param_5.number_input(
            label=params["input_exec_rate"]["label"],
            value=params["input_exec_rate"]["value"],
            step=params["input_exec_rate"]["step"],
            min_value=params["input_exec_rate"]["min_value"],
            max_value=params["input_exec_rate"]["max_value"],
            help=params["input_exec_rate"]["help"],
            on_change=None
            )
        sb_effectiveness_level = exp_param_5.selectbox(
            label=params["sb_effectiveness_level"]["label"],
            options=params["sb_effectiveness_level"]["options"],
            index=params["sb_effectiveness_level"]["index"],
            help=params["sb_effectiveness_level"]["help"],
            on_change=None
            )
        # input_DA_margin_price_per_mwh = exp_param_5.number_input(label='日前電能邊際價格(每mWh)', value=4757.123, step=0.25, format="%.3f", help=description['DA_margin_price_per_mwh'])
        # input_dispatch_ratio = exp_param_5.number_input(label='預估調度比例(%)', value=60, step=1, min_value=0, max_value=100, help=description['dispatch_ratio'])

        # Scenario setting
        exp_param_6 = form.expander('投標情境設定')
        cb_opt_bid = exp_param_6.checkbox(
            label=params["cb_opt_bid"]["label"],
            value=params["cb_opt_bid"]["value"],
            help=params["cb_opt_bid"]["help"],
            on_change=None
            )
        cb_opt_tendered_cap = exp_param_6.checkbox(
            label=params["cb_opt_tendered_cap"]["label"],
            value=params["cb_opt_tendered_cap"]["value"],
            help=params["cb_opt_tendered_cap"]["help"],
            on_change=None
            )
        cb_relax_tendered_step = exp_param_6.checkbox(
            label=params["cb_relax_tendered_step"]["label"],
            value=params["cb_relax_tendered_step"]["value"],
            help=params["cb_relax_tendered_step"]["help"],
            on_change=None
            )
        input_tendered_ub = exp_param_6.number_input(
            label=params["input_tendered_ub"]["label"],
            value=params["input_tendered_ub"]["value"],
            step=params["input_tendered_ub"]["step"],
            min_value=params["input_tendered_ub"]["min_value"],
            max_value=params["input_tendered_ub"]["max_value"],
            format=params["input_tendered_ub"]["format"],
            help=params["input_tendered_ub"]["help"],
            on_change=None
            )
        input_tendered_lb = exp_param_6.number_input(
            label=params["input_tendered_lb"]["label"],
            value=params["input_tendered_lb"]["value"],
            step=params["input_tendered_lb"]["step"],
            min_value=params["input_tendered_lb"]["min_value"],
            max_value=params["input_tendered_lb"]["max_value"],
            format=params["input_tendered_lb"]["format"],
            help=params["input_tendered_lb"]["help"],
            on_change=None
            )
        txt_ed_bid = exp_param_6.text(
            body=params["txt_ed_bid"]["body"],
            help=params["txt_ed_bid"]["help"]
            )
        ed_bid = exp_param_6.data_editor(
            data=params["ed_bid"]["data"],
            use_container_width=params["ed_bid"]["use_container_width"]
            )

        # Transmission-related setting
        exp_param_7 = form.expander('電力輸送相關')
        input_limit_g_es_p = exp_param_7.number_input(
            label=params["input_limit_g_es_p"]["label"],
            value=params["input_limit_g_es_p"]["value"],
            step=params["input_limit_g_es_p"]["step"],
            help=params["input_limit_g_es_p"]["help"],
            on_change=None
            )
        input_limit_es_p = exp_param_7.number_input(
            label=params["input_limit_es_p"]["label"],
            value=params["input_limit_es_p"]["value"],
            step=params["input_limit_es_p"]["step"],
            help=params["input_limit_es_p"]["help"],
            on_change=None
            )
        input_limit_g_p = exp_param_7.number_input(
            label=params["input_limit_g_p"]["label"],
            value=params["input_limit_g_p"]["value"],
            step=params["input_limit_g_p"]["step"],
            help=params["input_limit_g_p"]["help"],
            on_change=None
            )
        input_limit_pv_p = exp_param_7.number_input(
            label=params["input_limit_pv_p"]["label"],
            value=params["input_limit_pv_p"]["value"],
            step=params["input_limit_pv_p"]["step"],
            help=params["input_limit_pv_p"]["help"],
            on_change=None
            )
        input_loss_coef = exp_param_7.number_input(
            label=params["input_loss_coef"]["label"],
            value=params["input_loss_coef"]["value"],
            step=params["input_loss_coef"]["step"],
            min_value=params["input_loss_coef"]["min_value"],
            max_value=params["input_loss_coef"]["max_value"],
            format=params["input_loss_coef"]["format"],
            help=params["input_loss_coef"]["help"],
            on_change=None
            )

        # PV-related setting
        exp_param_8 = form.expander('太陽能發電機組相關')
        input_bulk_tariff_per_kwh = exp_param_8.number_input(
            label=params["input_bulk_tariff_per_kwh"]["label"],
            value=params["input_bulk_tariff_per_kwh"]["value"],
            step=params["input_bulk_tariff_per_kwh"]["step"],
            format=params["input_bulk_tariff_per_kwh"]["format"],
            help=params["input_bulk_tariff_per_kwh"]["help"],
            on_change=None
            )


class PlotRenderer:
    pass

class Optimizer:
    def __init__(self, data_service: DataService, model_builder: MIPModelBuilder, ui_handler: UIHandler):
        # init data service and data/params
        self.data_service = data_service
        if "sample_data" not in st.session_state:
            st.session_state["sample_data"] = data_service.load_sample_data()
        if "sample_params" not in st.session_state:
            st.session_state["sample_params"] = data_service.load_sample_params()
        if "data" not in st.session_state:
            st.session_state["data"] = st.session_state["sample_data"]
        if "params" not in st.session_state:
            st.session_state["params"] = st.session_state["sample_params"]

        # init model builder
        self.model_builder = model_builder
        self.model_builder.update()

        # UI handler
        self.ui_handler = ui_handler

    def set_state(self, key, value):
        st.session_state[key] = value

    def _initialize_session_state(self):
        # init streamlit session state
        # st.session_state["ui"] = {"input":{}, "select_box":{}, "check_box":{}, "editor":{}, "button":{}}
        pass

    def add(self):
        # for customize constraints / vars / objs
        pass

    def optimize(self):
        # 1. apply config/params updated from UI through set functions to model_builder
        # 2. build model
            # just self.model_builder.build() since changes already set through set functions

        # 3. call optimize functions

        # 4. return result
        pass




if __name__ == "__main__":
    data = DataService.load_sample_data()
    params = DataService.load_sample_params()
    builder = ESSModelBuilder()

    # Optimizer.set_data(data=data)
    # Optimizer.set_params(params=params)
    builder.set_data(data=data)
    builder.set_params(params=params)
    model = builder.build()
    result = model.optimize(max_seconds=builder.max_sec)

    print(result)
