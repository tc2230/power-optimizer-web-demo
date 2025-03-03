import os
import json
import base64
import numpy as np
import pandas as pd
import streamlit as st
from mip import Model, xsum, BINARY, INTEGER, CONTINUOUS, minimize, maximize, OptimizationStatus
from typing import Dict, List, Tuple, Optional, Type

import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from components.opt_utils import get_service_quality_index, get_effectiveness_price, verify_tendered_capacity_integrity, verify_tendered_capacity_in_bound, verify_tendered_capacity_non_negative, verify_bid_rule

class DataClient:
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

        sample_data_ref = {}
        for f in files:
            key = f.replace(".csv", "").replace("sample_", "")
            item = {
                "filename": f,
                "data": pd.read_csv(os.path.join(data_dir, f))
            }
            sample_data_ref[key] = item
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
        params["ed_bid"] = pd.DataFrame.from_dict(params["ed_bid"]).copy()

        return params

    @staticmethod
    @st.cache_data
    def load_ui_config() -> Dict[str, str]:
        """Load sample params from json files"""

        data_dir = "./data"
        params_filename = "ui_config.json"
        with open(os.path.join(data_dir, params_filename), 'r') as file:
            config = json.load(file)

        return config

    def load_data(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        """Load data from CSV files"""
        pass

    def load_params(self, data_dir: str, params_filename: str) -> Dict[str, str]:
        """Load params from json files"""
        pass

class MIPModel:
    def __init__(self):
        self.model = Model()
        pass

    def update_config(self):
        # update params from session state
        pass

    def update_data(self):
        # update data values from session state
        pass

    def update_price(self):
        # update pricing values according to session state configs
        pass

    def update_aux_values(self):
        # update auxiliary values according to session state configs
        pass

    def update(self):
        # update all variables related to model
        pass

    def _add_vars(self):
        pass

    def _add_constraints(self):
        pass

    def _add_objectives(self):
        pass

    def build(self):
        # update params
        # self._add_vars()
        # self._add_objectives()
        # self._add_constraints()
        pass

    def optimize(self, *args, **kwargs):
        pass

    @property
    def core(self):
        return self.model

class ESSModel(MIPModel):
    def __init__(self):
        super().__init__()

    def update_config(self):
        self.data_freq = st.session_state["current_params"]["sb_data_freq"]
        self.consecutive_n = int(60/self.data_freq)
        self.max_sec = st.session_state["current_params"]["input_max_sec"]
        self.c_cap = st.session_state["current_params"]["input_c_cap"]
        self.basic_tariff_per_kwh = st.session_state["current_params"]["input_basic_tariff_per_kwh"]
        self.summer = st.session_state["current_params"]["cb_summer"]
        self.e_cap = st.session_state["current_params"]["input_e_cap"]
        self.soc_init = st.session_state["current_params"]["input_soc_init"]
        self.opt_soc_init = st.session_state["current_params"]["cb_opt_soc_init"]
        self.soc_end = st.session_state["current_params"]["input_soc_end"]
        self.opt_soc_end = st.session_state["current_params"]["cb_opt_soc_end"]
        self.lb = st.session_state["current_params"]["input_lb"]
        self.ub = st.session_state["current_params"]["input_ub"]
        self.ess_degradation_cost_per_kwh_discharged = st.session_state["current_params"]["input_ess_degradation_cost_per_kwh_discharged"]
        self.factory_profit_per_kwh = st.session_state["current_params"]["input_factory_profit_per_kwh"]
        self.tendered_cap = st.session_state["current_params"]["ed_bid"]["tendered_cap(mWh)"]
        self.clearing_price_per_mwh = st.session_state["current_params"]["ed_bid"]["clearing_price(mWh)"]
        self.exec_rate = st.session_state["current_params"]["input_exec_rate"]
        self.effectiveness_level = st.session_state["current_params"]["sb_effectiveness_level"]
        self.DA_margin_price_per_mwh = st.session_state["current_params"]["ed_bid"]["marginal_price(mWh)"]
        self.dispatch_ratio = st.session_state["current_params"]["ed_bid"]["dispatch_ratio(%)"]
        self.opt_bid = st.session_state["current_params"]["cb_opt_bid"]
        self.opt_tendered_cap = st.session_state["current_params"]["cb_opt_tendered_cap"]
        self.relax_tendered_step = st.session_state["current_params"]["cb_relax_tendered_step"]
        self.tendered_lb = st.session_state["current_params"]["input_tendered_lb"]
        self.tendered_ub = st.session_state["current_params"]["input_tendered_ub"]
        self.bid = st.session_state["current_params"]["ed_bid"]["bid"].tolist()
        self.bid_win = st.session_state["current_params"]["ed_bid"]["win"].tolist()
        self.dispatch = st.session_state["current_params"]["ed_bid"]["dispatch"].tolist()
        self.limit_g_es_p = st.session_state["current_params"]["input_limit_g_es_p"]
        self.limit_es_p = st.session_state["current_params"]["input_limit_es_p"]
        self.limit_g_p = st.session_state["current_params"]["input_limit_g_p"]
        self.limit_pv_p = st.session_state["current_params"]["input_limit_pv_p"]
        self.loss_coef = st.session_state["current_params"]["input_loss_coef"]
        self.bulk_tariff_per_kwh = st.session_state["current_params"]["input_bulk_tariff_per_kwh"]
        pass

    def update_data(self):
        # retrieve data
        df_load = st.session_state["data"]["load"]["data"]
        df_pv = st.session_state["data"]["power"]["data"]

        ### set config according to updated data
        self.n = int(len(df_load)/(self.data_freq/5)) # number of time window
        # index
        self.index = df_load['time'].iloc[::int(self.data_freq/5)].values.flatten()
        # Load
        self.load = df_load['value'].iloc[::int(self.data_freq/5)].values.flatten()
        # PV
        self.pv = df_pv['value'].iloc[::int(self.data_freq/5)].values.flatten()
        pass

    def update_price(self):
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

    def update_aux_values(self):
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
        pass

    def update(self):
        self.update_config()
        self.update_data()
        self.update_price()
        self.update_aux_values()
        pass

    def _add_vars(self):
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
        self.if_exceed = [self.model.add_var(name=f"if_exceed_110%_cap_at_t{i}", var_type=BINARY) for i in range(n)]
        self.dummy_g_1 = [self.model.add_var(name=f"dummy_power_1_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        self.dummy_g_2 = [self.model.add_var(name=f"dummy_power_2_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        self.dummy_g_f = [self.model.add_var(name=f"dummy_power_from_grid_to_factory_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        self.dummy_g_es = [self.model.add_var(name=f"dummy_power_from_grid_to_ESS_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]

        # ESS charging/discharging status (or)
        self.if_charging = [self.model.add_var(name=f"ESS_is_charging_at_t{i}", var_type=BINARY) for i in range(n)]
        self.if_discharging = [self.model.add_var(name=f"ESS_is_discharging_at_t{i}", var_type=BINARY) for i in range(n)]
        self.aux_p_g_es = [self.model.add_var(name=f"aux_power_from_grid_to_ESS_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        self.aux_p_es_f = [self.model.add_var(name=f"aux_power_from_ESS_to_factory_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]
        self.aux_dummy_g_es = [self.model.add_var(name=f"aux_dummy_power_from_grid_to_ESS_t{i}", var_type=CONTINUOUS, lb=0) for i in range(n)]

        # total excessive power (aux for visualization)
        self.total_dummy = [self.model.add_var(name=f'total_excessive_power_t{i}', var_type=CONTINUOUS) for i in range(n)]
        self.total_flow_es = [self.model.add_var(name=f'total_power_flow_of_ESS_t{i}', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf')) for i in range(n)]

        # dummies for penalty calculation
        self.q_1 = self.model.add_var(name=f"max_dummy_power_1", var_type=CONTINUOUS)
        self.q_2 = self.model.add_var(name=f"max_dummy_power_2", var_type=CONTINUOUS)
        self.if_max_aux_1 = [self.model.add_var(name=f"max_func_aux_1_t{i}", var_type=BINARY) for i in range(n)]
        self.if_max_aux_2 = [self.model.add_var(name=f"max_func_aux_2_t{i}", var_type=BINARY) for i in range(n)]

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

    def _set_constraints(self):
        ### retrieve constants and data length
        n = self.n
        M = self.M

        ### add constraints
        ## basic constraints
        # set linear constraints for multiplication of decision variables
        for i in range(n):
            # either charging or discharging
            self.model.add_constr(self.if_charging[i] + self.if_discharging[i] <= 1)
            self.model.add_constr(self.aux_p_g_es[i] <= self.p_g_es[i])
            self.model.add_constr(self.aux_p_g_es[i] <= M * self.if_charging[i])
            self.model.add_constr(self.aux_p_g_es[i] >= self.p_g_es[i] + M * (self.if_charging[i]-1))
            self.model.add_constr(self.aux_dummy_g_es[i] <= self.dummy_g_es[i])
            self.model.add_constr(self.aux_dummy_g_es[i] <= M * self.if_charging[i])
            self.model.add_constr(self.aux_dummy_g_es[i] >= self.dummy_g_es[i] + M * (self.if_charging[i]-1))
            self.model.add_constr(self.aux_p_es_f[i] <= self.p_es_f[i])
            self.model.add_constr(self.aux_p_es_f[i] <= M * self.if_discharging[i])
            self.model.add_constr(self.aux_p_es_f[i] >= self.p_es_f[i] + M * (self.if_discharging[i]-1))
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
            self.model.add_constr(self.q_1 <= self.dummy_g_1[i] + M * self.if_max_aux_1[i])
            self.model.add_constr(self.q_2 >= self.dummy_g_2[i])
            self.model.add_constr(self.q_2 <= self.dummy_g_2[i] + M * self.if_max_aux_2[i])
        self.model.add_constr( xsum( self.if_max_aux_1[i] for i in range(n) ) <= n-1 )
        self.model.add_constr( xsum( self.if_max_aux_2[i] for i in range(n) ) <= n-1 )

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

        ## PV
        # flow balance
        for i in range(n):
            self.model.add_constr((self.p_pv_f[i] + self.p_pv_es[i] + self.p_pv_g[i]) == self.pv[i])
        # serving priority
        for i in range(n):
            self.model.add_constr(self.p_pv_f[i] >= self.p_pv_g[i])

        ## split excessive power for additional tariff calculation
        for i in range(n):
            self.model.add_constr(0.1*self.c_cap*self.if_exceed[i] <= self.dummy_g_1[i])
            self.model.add_constr(self.dummy_g_1[i] <= 0.1*self.c_cap)
            self.model.add_constr(self.dummy_g_2[i] >= 0)
            self.model.add_constr(self.dummy_g_2[i] <= self.if_exceed[i]*M)
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
        self.factory_income = self.model.add_var(name='factory_income', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        self.model.add_constr(self.factory_income == xsum( self.factory_profit_per_kwh*self.load[i]/self.consecutive_n for i in range(n) ))

        # PV income
        self.pv_income = self.model.add_var(name='PV_income', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        self.model.add_constr(self.pv_income == xsum( self.bulk_tariff_per_kwh*(self.p_pv_g[i]+self.p_pv_f[i])/self.consecutive_n for i in range(n) )) ############################ 全額躉售計費修改

        # total income
        self.model.add_constr(self.income == (self.dispatch_income + self.factory_income + self.pv_income))

        # fixed eletricity tariff
        self.fixed_e_cost = self.model.add_var(name='fixed_eletricity_tariff', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        self.model.add_constr(self.fixed_e_cost == self.basic_tariff_per_kwh*(1*self.c_cap + self.dummy_penalty_coef_1*self.q_1 + self.dummy_penalty_coef_2*self.q_2)/30) ############################ 全額躉售計費修改

        # usage eletricity tariff
        self.usage_e_cost = self.model.add_var(name='usage_eletricity_tariff', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        self.model.add_constr(self.usage_e_cost == xsum( self.price[i]*(self.p_g_f[i] + self.p_pv_f[i] - self.p_pv_g[i] + self.aux_p_g_es[i] + self.dummy_g_1[i] + self.dummy_g_2[i])/self.consecutive_n for i in range(n) )) ############################ 全額躉售計費修改

        # ESS degradation
        self.ess_dis_cost = self.model.add_var(name='ess_discharging_degradation_cost', var_type=CONTINUOUS, lb=float('-Inf'), ub=float('Inf'))
        self.model.add_constr(self.ess_dis_cost == xsum( self.ess_degradation_cost_per_kwh_discharged*self.aux_p_es_f[i]/self.consecutive_n for i in range(n) ))

        # total cost
        self.model.add_constr(self.cost == (self.fixed_e_cost + self.usage_e_cost + self.ess_dis_cost))

        # total revenue
        self.model.add_constr(self.revenue == (self.income - self.cost))
        self.model.objective = maximize(self.revenue)

    def build(self):
        self.update()
        self._add_vars()
        self._set_constraints()
        self._set_objectives()

    def optimize(self, *args, **kwargs):
        return self.model.optimize(*args, **kwargs)

class UIHandler:
    """Base class for UI styling and components"""
    def __init__(self):
        self.plot_client = PlotClient()

        # retrieve ui configs
        self.config = st.session_state["ui_config"]
        self.default_params = st.session_state["default_params"]

    def set_render_config(self, side_logo_path: str , caption: str|None = None):
        # set css of buttons
        st.markdown("<style>.row-widget.stButton {text-align: center;}</style>", unsafe_allow_html=True)

        # set picture and caption at the top of sidebar
        with open(side_logo_path, "rb") as f:
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
                unsafe_allow_html=True,
            )

    def render_input_data_section(self):
        # page title
        self.title = st.title('test demo')

        # expander for upload field
        self.exp_upload = st.expander('資料上傳區域', expanded=True)
        self.exp_upload.markdown('<p style="font-family:Source Sans Pro, sans-serif; font-size: 1.5rem; font-weight: bold;">於此處上傳資料</p>', unsafe_allow_html=True)
        self.exp_upload.text('目前使用採樣週期為五分鐘的單日資料。')

        self.col_upload = self.exp_upload.columns(2)

        # Create data sections
        self.upload_load = self.col_upload[0].file_uploader('工廠負載資料')
        self.upload_power = self.col_upload[1].file_uploader('太陽能發電資料')

        # update load data if file uploaded
        if self.upload_load is not None:
            # update session state
            data, filename = pd.read_csv(self.upload_load), self.upload_load.name
            st.session_state['data']["load"]["data"] = data.copy()
            st.session_state['data']["load"]["filename"] = self.upload_load.name
        else:
            filename = st.session_state["data"]["load"]["filename"]
            data = st.session_state["data"]["load"]["data"].set_index('time')

        # retrieve/plot figure
        if filename in st.session_state["fig"]:
            fig = st.session_state["fig"][filename]
        else:
            fig = self.plot_client.make_data_plot(data, name=filename, fig_title="")
            st.session_state["fig"][filename] = fig

        # subheader
        self.col_upload[0].subheader(filename)
        # display figure
        self.col_upload[0].plotly_chart(fig, use_container_width=True)
        # display dataframe
        data = data.rename(columns={'time':'時間', 'value':'負載量(kWh)'})
        self.col_upload[0].dataframe(data, use_container_width=True)

        # update load data if file uploaded
        if self.upload_power is not None:
            # update session state
            data, filename = pd.read_csv(self.upload_power), self.upload_power.name
            st.session_state['data']["power"]["data"] = data.copy()
            st.session_state['data']["power"]["filename"] = self.upload_power.name
        else:
            filename = st.session_state["data"]["power"]["filename"]
            data = st.session_state["data"]["power"]["data"].set_index('time')

        # retrieve/plot figure
        if filename in st.session_state["fig"]:
            fig = st.session_state["fig"][filename]
        else:
            fig = self.plot_client.make_data_plot(data, name=filename, fig_title="")
            st.session_state["fig"][filename] = fig

        # subheader
        self.col_upload[1].subheader(filename)
        # display figure
        self.col_upload[1].plotly_chart(fig, use_container_width=True)
        # display dataframe
        data = data.rename(columns={'time':'時間', 'value':'發電量(kWh)'})
        self.col_upload[1].dataframe(data, use_container_width=True)

    def render_sidebar(self):
        # form
        self.form = st.sidebar.form(key='Optimize', clear_on_submit=False)

        # form submit button
        self.placeholder_btn = self.form.empty()
        self.btn_opt = self.placeholder_btn.form_submit_button(
            label='Optimize', )
            # on_click=optimize_callback)

        # displayed status
        self.placeholder_status = self.form.empty()
        self.text_opt_status = self.placeholder_status.text(
            body=f'Status:',
            help=self.config["text_opt_status"]["help"]
            )
        self.form.divider()

        ## Parameter setting
        # header
        self.form.header('參數設定')

        # optimization setting
        exp_param_1 = self.form.expander('資料與求解參數')#, expanded=True
        sb_data_freq = exp_param_1.selectbox(
            label=self.config["sb_data_freq"]["label"],
            options=self.config["sb_data_freq"]["options"],
            index=self.config["sb_data_freq"]["index"],
            help=self.config["sb_data_freq"]["help"])
        st.session_state["current_params"]["sb_data_freq"] = sb_data_freq
        input_max_sec = exp_param_1.number_input(
            label=self.config["input_max_sec"]["label"],
            value=self.default_params["input_max_sec"],
            step=self.config["input_max_sec"]["step"],
            help=self.config["input_max_sec"]["help"])
        st.session_state["current_params"]["input_max_sec"] = input_max_sec

        # Price-related setting
        exp_param_2 = self.form.expander('電力價格相關')
        input_c_cap = exp_param_2.number_input(
            label=self.config["input_c_cap"]["label"],
            value=self.default_params["input_c_cap"],
            step=self.config["input_c_cap"]["step"],
            help=self.config["input_c_cap"]["help"])
        st.session_state["current_params"]["input_c_cap"] = input_c_cap
        input_basic_tariff_per_kwh = exp_param_2.number_input(
            label=self.config["input_basic_tariff_per_kwh"]["label"],
            value=self.default_params["input_basic_tariff_per_kwh"],
            step=self.config["input_basic_tariff_per_kwh"]["step"],
            format=self.config["input_basic_tariff_per_kwh"]["format"],
            help=self.config["input_basic_tariff_per_kwh"]["help"])
        st.session_state["current_params"]["input_basic_tariff_per_kwh"] = input_basic_tariff_per_kwh
        cb_summer = exp_param_2.checkbox(
            label=self.config["cb_summer"]["label"],
            value=self.default_params["cb_summer"],
            help=self.config["cb_summer"]["help"])
        st.session_state["current_params"]["cb_summer"] = cb_summer

        # ESS-related setting
        exp_param_3 = self.form.expander('儲能系統相關')
        input_e_cap = exp_param_3.number_input(
            label=self.config["input_e_cap"]["label"],
            value=self.default_params["input_e_cap"],
            step=self.config["input_e_cap"]["step"],
            help=self.config["input_e_cap"]["help"])
        st.session_state["current_params"]["input_e_cap"] = input_e_cap
        input_ub = exp_param_3.number_input(
            label=self.config["input_ub"]["label"],
            value=self.default_params["input_ub"],
            step=self.config["input_ub"]["step"],
            min_value=self.config["input_ub"]["min_value"],
            max_value=self.config["input_ub"]["max_value"],
            help=self.config["input_ub"]["help"])
        st.session_state["current_params"]["input_ub"] = input_ub
        input_lb= exp_param_3.number_input(
            label=self.config["input_lb"]["label"],
            value=self.default_params["input_lb"],
            step=self.config["input_lb"]["step"],
            min_value=self.config["input_lb"]["min_value"],
            max_value=self.config["input_lb"]["max_value"],
            help=self.config["input_lb"]["help"])
        st.session_state["current_params"]["input_lb"] = input_lb
        input_soc_init = exp_param_3.number_input(
            label=self.config["input_soc_init"]["label"],
            value=self.default_params["input_soc_init"],
            step=self.config["input_soc_init"]["step"],
            min_value=self.config["input_soc_init"]["min_value"],
            max_value=self.config["input_soc_init"]["max_value"],
            help=self.config["input_soc_init"]["help"])
        st.session_state["current_params"]["input_soc_init"] = input_soc_init
        cb_opt_soc_init = exp_param_3.checkbox(
            label=self.config["cb_opt_soc_init"]["label"],
            value=self.default_params["cb_opt_soc_init"],
            help=self.config["cb_opt_soc_init"]["help"])
        st.session_state["current_params"]["cb_opt_soc_init"] = cb_opt_soc_init
        input_soc_end = exp_param_3.number_input(
            label=self.config["input_soc_end"]["label"],
            value=self.default_params["input_soc_end"],
            step=self.config["input_soc_end"]["step"],
            min_value=self.config["input_soc_end"]["min_value"],
            max_value=self.config["input_soc_end"]["max_value"],
            help=self.config["input_soc_end"]["help"])
        st.session_state["current_params"]["input_soc_end"] = input_soc_end
        cb_opt_soc_end = exp_param_3.checkbox(
            label=self.config["cb_opt_soc_end"]["label"],
            value=self.default_params["cb_opt_soc_end"],
            help=self.config["cb_opt_soc_end"]["help"])
        st.session_state["current_params"]["cb_opt_soc_end"] = cb_opt_soc_end
        input_ess_degradation_cost_per_kwh_discharged = exp_param_3.number_input(
            label=self.config["input_ess_degradation_cost_per_kwh_discharged"]["label"],
            value=self.default_params["input_ess_degradation_cost_per_kwh_discharged"],
            step=self.config["input_ess_degradation_cost_per_kwh_discharged"]["step"],
            format=self.config["input_ess_degradation_cost_per_kwh_discharged"]["format"],
            help=self.config["input_ess_degradation_cost_per_kwh_discharged"]["help"])
        st.session_state["current_params"]["input_ess_degradation_cost_per_kwh_discharged"] = input_ess_degradation_cost_per_kwh_discharged

        # Production-related setting
        exp_param_4 = self.form.expander('生產相關')
        input_factory_profit_per_kwh = exp_param_4.number_input(
            label=self.config["input_factory_profit_per_kwh"]["label"],
            value=self.default_params["input_factory_profit_per_kwh"],
            step=self.config["input_factory_profit_per_kwh"]["step"],
            format=self.config["input_factory_profit_per_kwh"]["format"],
            help=self.config["input_factory_profit_per_kwh"]["help"])
        st.session_state["current_params"]["input_factory_profit_per_kwh"] = input_factory_profit_per_kwh

        # Trading-related setting
        exp_param_5 = self.form.expander('輔助服務投標相關')
        input_exec_rate = exp_param_5.number_input(
            label=self.config["input_exec_rate"]["label"],
            value=self.default_params["input_exec_rate"],
            step=self.config["input_exec_rate"]["step"],
            min_value=self.config["input_exec_rate"]["min_value"],
            max_value=self.config["input_exec_rate"]["max_value"],
            help=self.config["input_exec_rate"]["help"])
        st.session_state["current_params"]["input_exec_rate"] = input_exec_rate
        sb_effectiveness_level = exp_param_5.selectbox(
            label=self.config["sb_effectiveness_level"]["label"],
            options=self.config["sb_effectiveness_level"]["options"],
            index=self.config["sb_effectiveness_level"]["index"],
            help=self.config["sb_effectiveness_level"]["help"])
        st.session_state["current_params"]["sb_effectiveness_level"] = sb_effectiveness_level

        # Scenario setting
        exp_param_6 = self.form.expander('投標情境設定')
        cb_opt_bid = exp_param_6.checkbox(
            label=self.config["cb_opt_bid"]["label"],
            value=self.default_params["cb_opt_bid"],
            help=self.config["cb_opt_bid"]["help"])
        st.session_state["current_params"]["cb_opt_bid"] = cb_opt_bid
        cb_opt_tendered_cap = exp_param_6.checkbox(
            label=self.config["cb_opt_tendered_cap"]["label"],
            value=self.default_params["cb_opt_tendered_cap"],
            help=self.config["cb_opt_tendered_cap"]["help"])
        st.session_state["current_params"]["cb_opt_tendered_cap"] = cb_opt_tendered_cap
        cb_relax_tendered_step = exp_param_6.checkbox(
            label=self.config["cb_relax_tendered_step"]["label"],
            value=self.default_params["cb_relax_tendered_step"],
            help=self.config["cb_relax_tendered_step"]["help"])
        st.session_state["current_params"]["cb_relax_tendered_step"] = cb_relax_tendered_step
        input_tendered_ub = exp_param_6.number_input(
            label=self.config["input_tendered_ub"]["label"],
            value=self.default_params["input_tendered_ub"],
            step=self.config["input_tendered_ub"]["step"],
            min_value=self.config["input_tendered_ub"]["min_value"],
            max_value=self.config["input_tendered_ub"]["max_value"],
            format=self.config["input_tendered_ub"]["format"],
            help=self.config["input_tendered_ub"]["help"])
        st.session_state["current_params"]["input_tendered_ub"] = input_tendered_ub
        input_tendered_lb = exp_param_6.number_input(
            label=self.config["input_tendered_lb"]["label"],
            value=self.default_params["input_tendered_lb"],
            step=self.config["input_tendered_lb"]["step"],
            min_value=self.config["input_tendered_lb"]["min_value"],
            max_value=self.config["input_tendered_lb"]["max_value"],
            format=self.config["input_tendered_lb"]["format"],
            help=self.config["input_tendered_lb"]["help"])
        st.session_state["current_params"]["input_tendered_lb"] = input_tendered_lb
        txt_ed_bid = exp_param_6.text(
            body=self.config["txt_ed_bid"]["body"],
            help=self.config["txt_ed_bid"]["help"])
        st.session_state["current_params"]["txt_ed_bid"] = txt_ed_bid
        ed_bid = exp_param_6.data_editor(
            data=self.default_params["ed_bid"],
            use_container_width=True)
        st.session_state["current_params"]["ed_bid"] = ed_bid

        # Transmission-related setting
        exp_param_7 = self.form.expander('電力輸送相關')
        input_limit_g_es_p = exp_param_7.number_input(
            label=self.config["input_limit_g_es_p"]["label"],
            value=self.default_params["input_limit_g_es_p"],
            step=self.config["input_limit_g_es_p"]["step"],
            help=self.config["input_limit_g_es_p"]["help"])
        st.session_state["current_params"]["input_limit_g_es_p"] = input_limit_g_es_p
        input_limit_es_p = exp_param_7.number_input(
            label=self.config["input_limit_es_p"]["label"],
            value=self.default_params["input_limit_es_p"],
            step=self.config["input_limit_es_p"]["step"],
            help=self.config["input_limit_es_p"]["help"])
        st.session_state["current_params"]["input_limit_es_p"] = input_limit_es_p
        input_limit_g_p = exp_param_7.number_input(
            label=self.config["input_limit_g_p"]["label"],
            value=self.default_params["input_limit_g_p"],
            step=self.config["input_limit_g_p"]["step"],
            help=self.config["input_limit_g_p"]["help"])
        st.session_state["current_params"]["input_limit_g_p"] = input_limit_g_p
        input_limit_pv_p = exp_param_7.number_input(
            label=self.config["input_limit_pv_p"]["label"],
            value=self.default_params["input_limit_pv_p"],
            step=self.config["input_limit_pv_p"]["step"],
            help=self.config["input_limit_pv_p"]["help"])
        st.session_state["current_params"]["input_limit_pv_p"] = input_limit_pv_p
        input_loss_coef = exp_param_7.number_input(
            label=self.config["input_loss_coef"]["label"],
            value=self.default_params["input_loss_coef"],
            step=self.config["input_loss_coef"]["step"],
            min_value=self.config["input_loss_coef"]["min_value"],
            max_value=self.config["input_loss_coef"]["max_value"],
            format=self.config["input_loss_coef"]["format"],
            help=self.config["input_loss_coef"]["help"])
        st.session_state["current_params"]["input_loss_coef"] = input_loss_coef

        # PV-related setting
        exp_param_8 = self.form.expander('太陽能發電機組相關')
        input_bulk_tariff_per_kwh = exp_param_8.number_input(
            label=self.config["input_bulk_tariff_per_kwh"]["label"],
            value=self.default_params["input_bulk_tariff_per_kwh"],
            step=self.config["input_bulk_tariff_per_kwh"]["step"],
            format=self.config["input_bulk_tariff_per_kwh"]["format"],
            help=self.config["input_bulk_tariff_per_kwh"]["help"])
        st.session_state["current_params"]["input_bulk_tariff_per_kwh"] = input_bulk_tariff_per_kwh

    def render_optimization_result(self, validate_callback: callable, optimize_callback: callable):
        config = self.config

        if self.btn_opt:
            with st.spinner("ZzZZzzz..."):
                # validate params setting
                valid, msg = validate_callback()
                if not valid:
                    print(1)
                    self.placeholder_warning = st.empty()
                    self.placeholder_warning.warning(msg, icon="⚠️") # Shortcodes are not allowed when using warning container, only single character.
                    st.stop()
                else:
                    print(2)
                    try:
                        # optimize with callback
                        status, df_report, df_result = optimize_callback()
                        # update result
                        if status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
                            st.session_state["fig"]["result"] = self.plot_client.make_result_plot(df_result)# , secondary_y_limit=[0,input_tendered_cap]
                            st.session_state["data"]["result"] = df_result.copy()
                            st.session_state["data"]["report"] = df_report.copy()
                        else:
                            st.session_state["fig"]["result"] = None
                            st.session_state["data"]["result"] = None
                            st.session_state["data"]["report"] = None
                        # update session states
                        st.session_state["optimization_status"] = status
                        st.session_state["optimization_count"] += 1
                    except Exception as e:
                        self.placeholder_warning = st.empty()
                        self.placeholder_warning.warning(f"Something went wrong.", icon="💀") # Shortcodes are not allowed when using warning container, only single character.
                        st.stop()

        if st.session_state["optimization_count"] > 0:
            if st.session_state["optimization_status"] not in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
                # display message
                self.placeholder_warning = st.empty()
                self.placeholder_warning.warning("Current status is not optimal either feasible, please check https://python-mip.readthedocs.io/en/latest/classes.html#OptimizationStatus for further information.", icon="⚠️") # Shortcodes are not allowed when using warning container, only single character.
                st.stop()
            else:
                self.text_opt_status = self.placeholder_status.text(
                    body=f'Status: {st.session_state["optimization_status"].name}',
                    help=config["text_opt_status"]["help"]
                    )

                # set dataframe index
                df_report = st.session_state["data"]["report"].set_index('項目')
                df_result = st.session_state["data"]["result"].set_index('time')

                # create container
                self.exp_opt = st.expander("檢視最佳化結果與報表", expanded=True)

                # grpah
                self.exp_opt.subheader('最佳化排程圖表')
                self.exp_opt.plotly_chart(st.session_state["fig"]["result"], use_container_width=True)

                # create column
                self.col_opt = self.exp_opt.columns((2,4))

                # report
                self.col_opt[0].markdown('<p style="font-family:Source Sans Pro, sans-serif; font-size: 1.5rem; font-weight: bold;">收益報表</p>', unsafe_allow_html=True)
                self.col_opt[0].dataframe(st.session_state["data"]["report"], use_container_width=True)

                # operational result
                self.col_opt[1].markdown('<p style="font-family:Source Sans Pro, sans-serif; font-size: 1.5rem; font-weight: bold;">最佳化排程</p>', unsafe_allow_html=True)
                self.col_opt[1].dataframe(st.session_state["data"]["result"], use_container_width=True)

        st.caption(f'Optimization count : {st.session_state["optimization_count"]}')

    def render(self, validate_callback: callable, optimize_callback: callable):
        # set render config
        self.set_render_config(side_logo_path="./img/thisisfine.png")

        # retrieve ui configs
        config = st.session_state["ui_config"]

        ### Dashboard section - input data
        self.render_input_data_section()

        ### Sidebar section
        self.render_sidebar()

        ### Dashboard section - optimization result
        self.render_optimization_result(validate_callback=validate_callback, optimize_callback=optimize_callback)

class PlotClient:
    def __init__(self):
        pass

    def make_data_plot(self, df: pd.DataFrame, name: str = "_", fig_title: str = '', x: str = 'time', y: str = 'value'):
        df = df.reset_index()
        fig = px.line(df, x=x, y=y)
        fig.update_layout(
            title=fig_title,
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

    def make_result_plot(self, df: pd.DataFrame, name: str = "_", fig_title: str = '', secondary_y_limit: bool = None):
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
        fig.update_layout(title_text=fig_title, xaxis_title="time", yaxis_title="Power(kWh)", margin=dict(t=28),
                        font=dict(size=32, family="Arial", color="black"))
        # fig.update_yaxes(range=[0,df['total_power_from_grid_to_factory'].min()-100], secondary_y=True)
        # fig.update_yaxes(range=[0, secondary_y_limit], secondary_y=False)
        # fig.update_yaxes(range=secondary_y_limit, secondary_y=False)

        fig.update_yaxes(rangemode='nonnegative', scaleanchor='y', scaleratio=1, constraintoward='bottom', secondary_y=True)
        fig.update_yaxes(rangemode='normal', scaleanchor='y2', scaleratio=0.5, constraintoward='bottom', secondary_y=False)
        # st.session_state["fig"][name] = fig
        return fig

class Optimizer:
    def __init__(self, model: MIPModel, ui_handler: UIHandler):
        # model
        self.model = model

        # UI handler
        self.ui_handler = ui_handler

    def start(self):
        self.ui_handler.render(
            validate_callback=self.validate_all_setting,
            optimize_callback=self.optimize
            )

    def add(self):
        # for customize constraints / vars / objs
        pass

    def validate_SOC_setting(self) -> bool:
        input_lb = st.session_state["current_params"]["input_lb"]
        input_ub = st.session_state["current_params"]["input_ub"]
        input_soc_init = st.session_state["current_params"]["input_soc_init"]
        input_soc_end = st.session_state["current_params"]["input_soc_end"]
        # check SOC setting
        if not (
            input_lb < input_ub and
            input_lb <= input_soc_init <= input_ub and
            input_lb <= input_soc_end <= input_ub
            ):
            return False
        return True

    def validate_bid_setting(self) -> bool:
        df_ed_bid = st.session_state["current_params"]["ed_bid"]
        opt_bid = st.session_state["current_params"]["cb_opt_bid"]

        # trading-related rule
        if not verify_bid_rule(
            df_ed_bid=df_ed_bid,
            opt_bid=opt_bid
            ):
            return False
        return True

    def validate_tendered_cap_setting(self) -> bool:
        df_ed_bid = st.session_state["current_params"]["ed_bid"]
        cb_opt_tendered_cap = st.session_state["current_params"]["cb_opt_tendered_cap"]
        cb_relax_tendered_step = st.session_state["current_params"]["cb_opt_tendered_cap"]
        input_tendered_lb = st.session_state["current_params"]["input_tendered_lb"]
        input_tendered_ub = st.session_state["current_params"]["input_tendered_ub"]

        # tendered capacity
        if not cb_opt_tendered_cap:
            if not all(
                [
                    verify_tendered_capacity_integrity(df_ed_bid, relax=cb_relax_tendered_step),
                    verify_tendered_capacity_in_bound(df_ed_bid, lb=input_tendered_lb, ub=input_tendered_ub),
                    verify_tendered_capacity_non_negative(df_ed_bid)
                ]
            ):
                return False
        return True

    def validate_all_setting(self) -> Tuple[bool, str]:
        if not self.validate_SOC_setting():
            return False, "Check SOC boundary setting."
        if not self.validate_bid_setting():
            return False, "Check trading senario setting is correct."
        if not self.validate_tendered_cap_setting():
            return False, "Check tendered capacity setting is correct.(non-negativity / integrity / not in bound)"
        return True, "parameter setting is all good."

    def generate_result(self):
        # init
        model = self.model
        report = []
        result = {}

        report.append(('總收益', model.revenue.x))
        report.append(('輔助服務總收入', model.dispatch_income.x))

        report.append(('容量費', model.capacity_reserve_income.x))
        if model.capacity_reserve_income.x is None:
            report.append(('容量費(服務品質指標)', None))
        else:
            report.append(('容量費(服務品質指標)', model.capacity_reserve_income.x * model.service_quality_index))

        report.append(('效能費', model.effectiveness_income.x))
        if model.effectiveness_income.x is None:
            report.append(('效能費(服務品質指標)', None))
        else:
            report.append(('效能費(服務品質指標)', model.effectiveness_income.x * model.service_quality_index))

        report.append(('電能費', model.energy_income.x))
        report.append(('工廠生產收入', model.factory_income.x))
        report.append(('太陽能躉售收入', model.pv_income.x))
        report.append(('總收入', model.income.x))
        report.append(('基本電費', model.fixed_e_cost.x))
        report.append(('流動電費', model.usage_e_cost.x))
        report.append(('儲能設備耗損成本', model.ess_dis_cost.x))
        report.append(('總成本', model.cost.x))

        if model.core.status == 0:
            report = [round(r, 4) for r in report]
        df_report = pd.DataFrame(report, columns=['項目', '金額'])

        # optimized schedule
        result['time'] = model.index
        result['load'] = [val for val in model.load]
        result['pv'] = [val for val in model.pv]

        result['safe_range'] = [model.c_cap-100*model.aux_tendered_cap[i].x*model.bid_win[i]*model.dispatch[i]*model.dispatch_ratio[i] for i in range(model.n)]

        result['power_from_grid_to_factory'] = [v.x for v in model.p_g_f]
        result['power_from_ESS_to_factory'] = [v.x for v in model.aux_p_es_f]
        result['power_from_PV_to_factory'] = [v.x for v in model.p_pv_f]
        result['power_from_PV_to_ESS'] = [v.x for v in model.p_pv_es]
        result['power_from_PV_to_grid'] = [v.x for v in model.p_pv_g]
        result['power_from_grid_to_ESS'] = [v.x for v in model.aux_p_g_es]

        result['ESS_SOC'] = [v.x for v in model.es]
        result['ESS_is_charging'] = [v.x for v in model.if_charging]
        result['ESS_is_discharging'] = [v.x for v in model.if_discharging]

        result['exceed_contract_capacity'] = [v.x for v in model.if_exceed]
        result['excessive_power_below_110%'] = [v.x for v in model.dummy_g_1]
        result['excessive_power_over_110%'] = [v.x for v in model.dummy_g_2]
        result['excessive_power_from_grid_to_factory'] = [v.x for v in model.dummy_g_f]
        result['excessive_power_from_grid_to_ESS'] = [v.x for v in model.aux_dummy_g_es]

        if model.opt_bid:
            result['bid'] = [v.x for v in model.bid]
        else:
            result['bid'] = [v for v in model.bid]
        result['bid_win'] = [v for v in model.bid_win]
        result['dispatch'] = [v for v in model.dispatch]
        result['aux_tendered_cap(mWh)'] = [v.x/10 for v in model.aux_tendered_cap] # [v.x/10 if v.x else None for v in aux_tendered_cap]

        result['total_power_from_grid'] = [v.x for v in model.total_g]
        result['total_power_from_grid_to_factory'] = [v.x for v in model.total_g_f]
        result['total_power_from_grid_to_ESS'] = [v.x for v in model.total_g_es]
        result['total_excessive_power'] = [v.x for v in model.total_dummy]
        result['total_power_flow_of_ESS'] = [v.x for v in model.total_flow_es]
        if model.core.status == 0:
            for k, l in result.items():
                if k == 'time':
                    continue
                else:
                    if k in ['ESS_is_charging', 'ESS_is_discharging', 'bid', 'bid_win', 'dispatch']:
                        result[k] = [int(val) for val in l]
                    else:
                        result[k] = [round(val, 4) for val in l]
        df_result = pd.DataFrame(result)

        return df_report, df_result

    def optimize(self):
        self.report = []
        self.result = {}
        self.model.build()
        status = self.model.optimize(
            max_seconds=st.session_state["current_params"]["input_max_sec"]
            )

        # return result
        if status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
            df_report, df_result = self.generate_result()
            return status, df_report, df_result
        else:
            return status, pd.DataFrame(), pd.DataFrame()

if __name__ == "__main__":
    # set page config
    st.set_page_config(page_title='Power optimizer demo(Cht)', layout="wide", page_icon='./img/favicon.png')

    # init data service client
    data_client = DataClient()

    # load sample data ad default parameters if it's at first init.
    if "optimization_status" not in st.session_state:
        st.session_state['optimization_status'] = None
        st.session_state['optimization_count'] = 0
        st.session_state["fig"] = {}
        st.session_state["fig"]["result"] = None
        st.session_state["fig"]["report"] = None
        st.session_state["data"] = data_client.load_sample_data()
        st.session_state["data"]["result"] = None
        st.session_state["data"]["report"] = None
        st.session_state["ui_config"] = data_client.load_ui_config()
        st.session_state["default_params"] = data_client.load_sample_params()
        st.session_state["current_params"] = {}

    # init model
    model = ESSModel()

    # init ui handler
    ui_handler = UIHandler()

    # create and start optimizer
    optimizer = Optimizer(
        model=model,
        ui_handler=ui_handler
        )

    # start
    optimizer.start()
