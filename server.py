import os
import io
import json
import base64
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from mip import Model, xsum, BINARY, INTEGER, CONTINUOUS, minimize, maximize, OptimizationStatus
from typing import Dict, List, Tuple, Optional, Type

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

def verify_tendered_capacity_integrity(df_ed_bid, relax=0):
    if not relax:
        return all([(10*i).is_integer() for i in df_ed_bid['tendered_cap(mWh)']])
    return True

def verify_tendered_capacity_in_bound(df_ed_bid, lb=0, ub=float('Inf')):
    return all([lb <= i <= ub for i in df_ed_bid['tendered_cap(mWh)']])

def verify_tendered_capacity_non_negative(df_ed_bid):
    return all([i >= 0 for i in df_ed_bid['tendered_cap(mWh)']])

def verify_bid_rule(df_ed_bid, opt_bid=True):
    if opt_bid:
        return all([bw >= d for bw, d in zip(df_ed_bid['win'], df_ed_bid['dispatch'])])
    else:
        row = all([bw >= d for bw, d in zip(df_ed_bid['win'], df_ed_bid['dispatch'])])
        tmp = df_ed_bid['bid']*df_ed_bid['win']*df_ed_bid['dispatch']
        dispatch = all([tmp[i:i+3].sum() <= 1 for i in range(len(df_ed_bid)-2)])
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

class MIPModel:
    def __init__(self):
        self.model = Model()
        pass

    def update(self):
        # update params and aux from session state
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

        # update ui parameter to session state
        params = st.session_state["params"]

        for k, v in params.items():
            print(k)
            if k not in st.session_state:
                if 'body' in v:
                    st.session_state[k] = v['body']
                elif 'index' in v:
                    st.session_state[k] = v['options'][v['index']]
                elif 'data' in v:
                    st.session_state[k] = v['data']
                else:
                    st.session_state[k] = v['value']
        # self.data_freq = params["sb_data_freq"]["options"][params["sb_data_freq"]["index"]]
        # self.max_sec = params["input_max_sec"]["value"]
        # self.c_cap = params["input_c_cap"]["value"]
        # self.basic_tariff_per_kwh = params["input_basic_tariff_per_kwh"]["value"]
        # self.summer = params["cb_summer"]["value"]
        # self.e_cap = params["input_e_cap"]["value"]
        # self.soc_init = params["input_soc_init"]["value"]
        # self.opt_soc_init = params["cb_opt_soc_init"]["value"]
        # self.soc_end = params["input_soc_end"]["value"]
        # self.opt_soc_end = params["cb_opt_soc_end"]["value"]
        # self.lb = params["input_lb"]["value"]
        # self.ub = params["input_ub"]["value"]
        # self.ess_degradation_cost_per_kwh_discharged = params["input_ess_degradation_cost_per_kwh_discharged"]["value"]
        # self.factory_profit_per_kwh = params["input_factory_profit_per_kwh"]["value"]
        # self.tendered_cap = params["ed_bid"]["data"]["tendered_cap(mWh)"]
        # self.clearing_price_per_mwh = params["ed_bid"]["data"]["clearing_price(mWh)"]
        # self.exec_rate = params["input_exec_rate"]["value"]
        # self.effectiveness_level = params["sb_effectiveness_level"]["options"][params["sb_effectiveness_level"]["index"]]
        # self.DA_margin_price_per_mwh = params["ed_bid"]["data"]["marginal_price(mWh)"]
        # self.dispatch_ratio = params["ed_bid"]["data"]["dispatch_ratio(%)"]
        # self.opt_bid = params["cb_opt_bid"]["value"]
        # self.opt_tendered_cap = params["cb_opt_tendered_cap"]["value"]
        # self.relax_tendered_step = params["cb_relax_tendered_step"]["value"]
        # self.tendered_lb = params["input_tendered_lb"]["value"]
        # self.tendered_ub = params["input_tendered_ub"]["value"]
        # self.bid = params["ed_bid"]["data"]["bid"].tolist()
        # self.bid_win = params["ed_bid"]["data"]["win"].tolist()
        # self.dispatch = params["ed_bid"]["data"]["dispatch"].tolist()
        # self.limit_g_es_p = params["input_limit_g_es_p"]["value"]
        # self.limit_es_p = params["input_limit_es_p"]["value"]
        # self.limit_g_p = params["input_limit_g_p"]["value"]
        # self.limit_pv_p = params["input_limit_pv_p"]["value"]
        # self.loss_coef = params["input_loss_coef"]["value"]
        # self.bulk_tariff_per_kwh = params["input_bulk_tariff_per_kwh"]["value"]

    def update(self):
        # apply parameter and data
        df_load = st.session_state["data"]["load"]["data"]
        df_pv = st.session_state["data"]["power"]["data"]
        params = st.session_state["params"]

        self.data_freq = st.session_state["sb_data_freq"]#["options"][st.session_state["sb_data_freq"]["index"]]
        self.max_sec = st.session_state["input_max_sec"]#["value"]
        self.c_cap = st.session_state["input_c_cap"]#["value"]
        self.basic_tariff_per_kwh = st.session_state["input_basic_tariff_per_kwh"]#["value"]
        self.summer = st.session_state["cb_summer"]#["value"]
        self.e_cap = st.session_state["input_e_cap"]#["value"]
        self.soc_init = st.session_state["input_soc_init"]#["value"]
        self.opt_soc_init = st.session_state["cb_opt_soc_init"]#["value"]
        self.soc_end = st.session_state["input_soc_end"]#["value"]
        self.opt_soc_end = st.session_state["cb_opt_soc_end"]#["value"]
        self.lb = st.session_state["input_lb"]#["value"]
        self.ub = st.session_state["input_ub"]#["value"]
        self.ess_degradation_cost_per_kwh_discharged = st.session_state["input_ess_degradation_cost_per_kwh_discharged"]#["value"]
        self.factory_profit_per_kwh = st.session_state["input_factory_profit_per_kwh"]#["value"]
        self.tendered_cap = st.session_state["ed_bid"]["tendered_cap(mWh)"]#["data"]["tendered_cap(mWh)"]
        self.clearing_price_per_mwh = st.session_state["ed_bid"]["clearing_price(mWh)"]#["data"]["clearing_price(mWh)"]
        self.exec_rate = st.session_state["input_exec_rate"]#["value"]
        self.effectiveness_level = st.session_state["sb_effectiveness_level"]#["options"][st.session_state["sb_effectiveness_level"]["index"]]
        self.DA_margin_price_per_mwh = st.session_state["ed_bid"]["marginal_price(mWh)"]#["data"]["marginal_price(mWh)"]
        self.dispatch_ratio = st.session_state["ed_bid"]["dispatch_ratio(%)"]#["data"]["dispatch_ratio(%)"]
        self.opt_bid = st.session_state["cb_opt_bid"]#["value"]
        self.opt_tendered_cap = st.session_state["cb_opt_tendered_cap"]#["value"]
        self.relax_tendered_step = st.session_state["cb_relax_tendered_step"]#["value"]
        self.tendered_lb = st.session_state["input_tendered_lb"]#["value"]
        self.tendered_ub = st.session_state["input_tendered_ub"]#["value"]
        self.bid = st.session_state["ed_bid"]["bid"].tolist()#["data"]["bid"].tolist()
        self.bid_win = st.session_state["ed_bid"]["win"].tolist()#["data"]["win"].tolist()
        self.dispatch = st.session_state["ed_bid"]["dispatch"].tolist()#["data"]["dispatch"].tolist()
        self.limit_g_es_p = st.session_state["input_limit_g_es_p"]#["value"]
        self.limit_es_p = st.session_state["input_limit_es_p"]#["value"]
        self.limit_g_p = st.session_state["input_limit_g_p"]#["value"]
        self.limit_pv_p = st.session_state["input_limit_pv_p"]#["value"]
        self.loss_coef = st.session_state["input_loss_coef"]#["value"]
        self.bulk_tariff_per_kwh = st.session_state["input_bulk_tariff_per_kwh"]#["value"]

        # params = st.session_state["params"]

        # self.data_freq = params["sb_data_freq"]["options"][params["sb_data_freq"]["index"]]
        # self.max_sec = params["input_max_sec"]["value"]
        # self.c_cap = params["input_c_cap"]["value"]
        # self.basic_tariff_per_kwh = params["input_basic_tariff_per_kwh"]["value"]
        # self.summer = params["cb_summer"]["value"]
        # self.e_cap = params["input_e_cap"]["value"]
        # self.soc_init = params["input_soc_init"]["value"]
        # self.opt_soc_init = params["cb_opt_soc_init"]["value"]
        # self.soc_end = params["input_soc_end"]["value"]
        # self.opt_soc_end = params["cb_opt_soc_end"]["value"]
        # self.lb = params["input_lb"]["value"]
        # self.ub = params["input_ub"]["value"]
        # self.ess_degradation_cost_per_kwh_discharged = params["input_ess_degradation_cost_per_kwh_discharged"]["value"]
        # self.factory_profit_per_kwh = params["input_factory_profit_per_kwh"]["value"]
        # self.tendered_cap = params["ed_bid"]["data"]["tendered_cap(mWh)"]
        # self.clearing_price_per_mwh = params["ed_bid"]["data"]["clearing_price(mWh)"]
        # self.exec_rate = params["input_exec_rate"]["value"]
        # self.effectiveness_level = params["sb_effectiveness_level"]["options"][params["sb_effectiveness_level"]["index"]]
        # self.DA_margin_price_per_mwh = params["ed_bid"]["data"]["marginal_price(mWh)"]
        # self.dispatch_ratio = params["ed_bid"]["data"]["dispatch_ratio(%)"]
        # self.opt_bid = params["cb_opt_bid"]["value"]
        # self.opt_tendered_cap = params["cb_opt_tendered_cap"]["value"]
        # self.relax_tendered_step = params["cb_relax_tendered_step"]["value"]
        # self.tendered_lb = params["input_tendered_lb"]["value"]
        # self.tendered_ub = params["input_tendered_ub"]["value"]
        # self.bid = params["ed_bid"]["data"]["bid"].tolist()
        # self.bid_win = params["ed_bid"]["data"]["win"].tolist()
        # self.dispatch = params["ed_bid"]["data"]["dispatch"].tolist()
        # self.limit_g_es_p = params["input_limit_g_es_p"]["value"]
        # self.limit_es_p = params["input_limit_es_p"]["value"]
        # self.limit_g_p = params["input_limit_g_p"]["value"]
        # self.limit_pv_p = params["input_limit_pv_p"]["value"]
        # self.loss_coef = params["input_loss_coef"]["value"]
        # self.bulk_tariff_per_kwh = params["input_bulk_tariff_per_kwh"]["value"]


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

# User --[data, params]--> Optimizer --[data, params]--> Model Builder: build(), optimize() --[result]--> Optimizer --[result, plots]--> User
class UIHandler:
    """Base class for UI styling and components"""
    def __init__(self):
        self.plot_client = PlotClient()

    # def update_params(self, param_name: str):
    #     def callback():
    #         if "index" in st.session_state["params"][param_name]:
    #             st.session_state["params"][param_name]["index"] = st.session_state[param_name]
    #         else:
    #             st.session_state["params"][param_name]["value"] = st.session_state[param_name]

    #     return callback

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

    def render(self, validate_callback: callable, optimize_callback: callable):
        params = st.session_state["params"]

        print('\n-----------------------------\n', params["input_c_cap"]["value"], '\n-----------------------------\n')
        print('\n-----------------------------\n', st.session_state["input_c_cap"], '\n-----------------------------\n')

        ### global streamlit config
        # st.set_page_config(page_title='Power Optimizer test(Cht)', layout="wide", page_icon='./img/favicon.png')
        st.markdown("<style>.row-widget.stButton {text-align: center;}</style>", unsafe_allow_html=True) # for button css

        ### Dashboard section - input data
        # page title
        self.title = st.title('最佳化工具 Demo')

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
            df = pd.read_csv(self.upload_load)
            st.session_state['data']["load"]["data"] = df.copy()
            st.session_state['data']["load"]["filename"] = self.upload_load.name

        # plot load data
        filename = st.session_state["data"]["load"]["filename"]
        data = st.session_state["data"]["load"]["data"].set_index('time')
        # subheader
        self.col_upload[0].subheader(filename)
        # get plotly figure
        fig = self.plot_client.make_data_plot(data, title="")
        self.col_upload[0].plotly_chart(fig, use_container_width=True)
        # translation
        data = data.rename(columns={'time':'時間', 'value':'負載量(kWh)'})
        # show dataframe
        self.col_upload[0].dataframe(data, use_container_width=True)

        # update load data if file uploaded
        if self.upload_power is not None:
            # update session state
            df = pd.read_csv(self.upload_power)
            st.session_state['data']["power"]["data"] = df.copy()
            st.session_state['data']["power"]["filename"] = self.upload_power.name

        # plot power data
        filename = st.session_state["data"]["power"]["filename"]
        data = st.session_state["data"]["power"]["data"].set_index('time')
        # subheader
        self.col_upload[1].subheader(filename)
        # get plotly figure
        fig = self.plot_client.make_data_plot(data, title="")
        self.col_upload[1].plotly_chart(fig, use_container_width=True)
        # translation
        data = data.rename(columns={'time':'時間', 'value':'發電量(kWh)'})
        # show dataframe
        self.col_upload[1].dataframe(data, use_container_width=True)

        # # optimize result
        # if params["text_opt_status"]["body"]:
        #     with st.spinner("ZzZZzzz..."):
        #         df_load = st.session_state["data"]["load"]["data"]
        #         df_power = st.session_state["data"]["power"]["data"]

        #     ## verify settings
        #     if not (
        #         params["input_lb"]["value"] < params["input_ub"]["value"] and
        #         params["input_lb"]["value"] <= params["input_soc_init"]["value"] <= params["input_lb"]["value"] and
        #         params["input_lb"]["value"] <= input_soc_end <= params["input_ub"]["value"]
        #         ):
        #         self.placeholder_warning = st.empty()
        #         self.placeholder_warning.warning('Check SOC boundary setting.', icon=":warning:")
        #         st.stop()

        ### Sidebar section
        # set sidebar picture
        self.set_sidebar_markdown(img_path="./img/thisisfine.png")

        ## form general setting
        form = st.sidebar.form(key='Optimize', clear_on_submit=False)

        # button
        self.placeholder_btn = form.empty()
        btn_opt = self.placeholder_btn.form_submit_button(
            label='Optimize',
            on_click=optimize_callback
            )

        # status
        self.placeholder_status = form.empty()
        self.text_opt_status = self.placeholder_status.text(
            body=f'Status: {params["text_opt_status"]["body"]}',
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

            )
        input_max_sec = exp_param_1.number_input(
            label=params["input_max_sec"]["label"],
            value=params["input_max_sec"]["value"],
            step=params["input_max_sec"]["step"],
            help=params["input_max_sec"]["help"],

            )

        # Price-related setting
        exp_param_2 = form.expander('電力價格相關')
        input_c_cap = exp_param_2.number_input(
            label=params["input_c_cap"]["label"],
            value=params["input_c_cap"]["value"],
            step=params["input_c_cap"]["step"],
            help=params["input_c_cap"]["help"],

            )
        input_basic_tariff_per_kwh = exp_param_2.number_input(
            label=params["input_basic_tariff_per_kwh"]["label"],
            value=params["input_basic_tariff_per_kwh"]["value"],
            step=params["input_basic_tariff_per_kwh"]["step"],
            format=params["input_basic_tariff_per_kwh"]["format"],
            help=params["input_basic_tariff_per_kwh"]["help"],

            )
        cb_summer = exp_param_2.checkbox(
            label=params["cb_summer"]["label"],
            value=params["cb_summer"]["value"],
            help=params["cb_summer"]["help"],

            )

        # ESS-related setting
        exp_param_3 = form.expander('儲能系統相關')
        input_e_cap = exp_param_3.number_input(
            label=params["input_e_cap"]["label"],
            value=params["input_e_cap"]["value"],
            step=params["input_e_cap"]["step"],
            help=params["input_e_cap"]["help"],

            )
        input_ub = exp_param_3.number_input(
            label=params["input_ub"]["label"],
            value=params["input_ub"]["value"],
            step=params["input_ub"]["step"],
            min_value=params["input_ub"]["min_value"],
            max_value=params["input_ub"]["max_value"],
            help=params["input_ub"]["help"],

            )
        input_lb= exp_param_3.number_input(
            label=params["input_lb"]["label"],
            value=params["input_lb"]["value"],
            step=params["input_lb"]["step"],
            min_value=params["input_lb"]["min_value"],
            max_value=params["input_lb"]["max_value"],
            help=params["input_lb"]["help"],

            )
        input_soc_init = exp_param_3.number_input(
            label=params["input_soc_init"]["label"],
            value=params["input_soc_init"]["value"],
            step=params["input_soc_init"]["step"],
            min_value=params["input_soc_init"]["min_value"],
            max_value=params["input_soc_init"]["max_value"],
            help=params["input_soc_init"]["help"],

            )
        cb_opt_soc_init = exp_param_3.checkbox(
            label=params["cb_opt_soc_init"]["label"],
            value=params["cb_opt_soc_init"]["value"],
            help=params["cb_opt_soc_init"]["help"],

            )
        input_soc_end = exp_param_3.number_input(
            label=params["input_soc_end"]["label"],
            value=params["input_soc_end"]["value"],
            step=params["input_soc_end"]["step"],
            min_value=params["input_soc_end"]["min_value"],
            max_value=params["input_soc_end"]["max_value"],
            help=params["input_soc_end"]["help"],

            )
        cb_opt_soc_end = exp_param_3.checkbox(
            label=params["cb_opt_soc_end"]["label"],
            value=params["cb_opt_soc_end"]["value"],
            help=params["cb_opt_soc_end"]["help"],

            )
        input_ess_degradation_cost_per_kwh_discharged = exp_param_3.number_input(
            label=params["input_ess_degradation_cost_per_kwh_discharged"]["label"],
            value=params["input_ess_degradation_cost_per_kwh_discharged"]["value"],
            step=params["input_ess_degradation_cost_per_kwh_discharged"]["step"],
            format=params["input_ess_degradation_cost_per_kwh_discharged"]["format"],
            help=params["input_ess_degradation_cost_per_kwh_discharged"]["help"],

            )

        # Production-related setting
        exp_param_4 = form.expander('生產相關')
        input_factory_profit_per_kwh = exp_param_4.number_input(
            label=params["input_factory_profit_per_kwh"]["label"],
            value=params["input_factory_profit_per_kwh"]["value"],
            step=params["input_factory_profit_per_kwh"]["step"],
            format=params["input_factory_profit_per_kwh"]["format"],
            help=params["input_factory_profit_per_kwh"]["help"],

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

            )
        sb_effectiveness_level = exp_param_5.selectbox(
            label=params["sb_effectiveness_level"]["label"],
            options=params["sb_effectiveness_level"]["options"],
            index=params["sb_effectiveness_level"]["index"],
            help=params["sb_effectiveness_level"]["help"],

            )
        # input_DA_margin_price_per_mwh = exp_param_5.number_input(label='日前電能邊際價格(每mWh)', value=4757.123, step=0.25, format="%.3f", help=description['DA_margin_price_per_mwh'])
        # input_dispatch_ratio = exp_param_5.number_input(label='預估調度比例(%)', value=60, step=1, min_value=0, max_value=100, help=description['dispatch_ratio'])

        # Scenario setting
        exp_param_6 = form.expander('投標情境設定')
        cb_opt_bid = exp_param_6.checkbox(
            label=params["cb_opt_bid"]["label"],
            value=params["cb_opt_bid"]["value"],
            help=params["cb_opt_bid"]["help"],

            )
        cb_opt_tendered_cap = exp_param_6.checkbox(
            label=params["cb_opt_tendered_cap"]["label"],
            value=params["cb_opt_tendered_cap"]["value"],
            help=params["cb_opt_tendered_cap"]["help"],

            )
        cb_relax_tendered_step = exp_param_6.checkbox(
            label=params["cb_relax_tendered_step"]["label"],
            value=params["cb_relax_tendered_step"]["value"],
            help=params["cb_relax_tendered_step"]["help"],

            )
        input_tendered_ub = exp_param_6.number_input(
            label=params["input_tendered_ub"]["label"],
            value=params["input_tendered_ub"]["value"],
            step=params["input_tendered_ub"]["step"],
            min_value=params["input_tendered_ub"]["min_value"],
            max_value=params["input_tendered_ub"]["max_value"],
            format=params["input_tendered_ub"]["format"],
            help=params["input_tendered_ub"]["help"],

            )
        input_tendered_lb = exp_param_6.number_input(
            label=params["input_tendered_lb"]["label"],
            value=params["input_tendered_lb"]["value"],
            step=params["input_tendered_lb"]["step"],
            min_value=params["input_tendered_lb"]["min_value"],
            max_value=params["input_tendered_lb"]["max_value"],
            format=params["input_tendered_lb"]["format"],
            help=params["input_tendered_lb"]["help"],

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

            )
        input_limit_es_p = exp_param_7.number_input(
            label=params["input_limit_es_p"]["label"],
            value=params["input_limit_es_p"]["value"],
            step=params["input_limit_es_p"]["step"],
            help=params["input_limit_es_p"]["help"],

            )
        input_limit_g_p = exp_param_7.number_input(
            label=params["input_limit_g_p"]["label"],
            value=params["input_limit_g_p"]["value"],
            step=params["input_limit_g_p"]["step"],
            help=params["input_limit_g_p"]["help"],

            )
        input_limit_pv_p = exp_param_7.number_input(
            label=params["input_limit_pv_p"]["label"],
            value=params["input_limit_pv_p"]["value"],
            step=params["input_limit_pv_p"]["step"],
            help=params["input_limit_pv_p"]["help"],

            )
        input_loss_coef = exp_param_7.number_input(
            label=params["input_loss_coef"]["label"],
            value=params["input_loss_coef"]["value"],
            step=params["input_loss_coef"]["step"],
            min_value=params["input_loss_coef"]["min_value"],
            max_value=params["input_loss_coef"]["max_value"],
            format=params["input_loss_coef"]["format"],
            help=params["input_loss_coef"]["help"],

            )

        # PV-related setting
        exp_param_8 = form.expander('太陽能發電機組相關')
        input_bulk_tariff_per_kwh = exp_param_8.number_input(
            label=params["input_bulk_tariff_per_kwh"]["label"],
            value=params["input_bulk_tariff_per_kwh"]["value"],
            step=params["input_bulk_tariff_per_kwh"]["step"],
            format=params["input_bulk_tariff_per_kwh"]["format"],
            help=params["input_bulk_tariff_per_kwh"]["help"],

            )

        # for k in sorted(list(st.session_state.keys())):
        #     if isinstance(st.session_state[k], pd.DataFrame):
        #         print(st.session_state[k].head())
        #     else:
        #         print('\n-----------------------------------\n', k, st.session_state[k])

        ### Dashboard section - optimization result
        with st.spinner("ZzZZzzz..."):
            # validate params setting
            valid, msg = validate_callback()
            if not valid:
                self.placeholder_warning = st.empty()
                self.placeholder_warning.warning(msg, icon="⚠️") # Shortcodes are not allowed when using warning container, only single character.
                st.stop()
            else:
                # optimize with callback
                status, df_report, df_result = optimize_callback()

                # # retrieve optimization status
                # status = st.session_state["params"]["text_opt_status"]["value"]

                # set dataframe index
                df_report = df_report.set_index('項目')
                df_result = df_result.set_index('time')

                # create container
                self.exp_opt = st.expander("檢視最佳化結果與報表", expanded=True)

                # grpah
                self.exp_opt.subheader('最佳化排程圖表')
                plt_result = self.plot_client.make_result_plot(df_result)# , secondary_y_limit=[0,input_tendered_cap]
                self.exp_opt.plotly_chart(plt_result, use_container_width=True)

                # create column
                self.col_opt = self.exp_opt.columns((2,4))

                # report
                self.col_opt[0].markdown('<p style="font-family:Source Sans Pro, sans-serif; font-size: 1.5rem; font-weight: bold;">收益報表</p>', unsafe_allow_html=True)
                self.col_opt[0].dataframe(df_report, use_container_width=True)

                # operational result
                self.col_opt[1].markdown('<p style="font-family:Source Sans Pro, sans-serif; font-size: 1.5rem; font-weight: bold;">最佳化排程</p>', unsafe_allow_html=True)
                self.col_opt[1].dataframe(df_result, use_container_width=True)

                # update optimize button and status
                self.btn_opt = self.placeholder_btn.form_submit_button(label=' Optimize ')

                self.placeholder_status.empty()
                st.session_state["params"]["text_opt_status"]["value"] = self.placeholder_status.text(
                    body=f'Status: {status}',
                    help=params["text_opt_status"]["help"]
                    )

                # count optimization
                st.caption(f'Optimization count : {st.session_state["optimization_count"]}')

class PlotClient:
    def __init__(self):
        pass

    def make_data_plot(self, df, title='', x='time', y='value'):
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

    def make_result_plot(self, df, title='', secondary_y_limit=None):
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

class Optimizer:
    def __init__(self, data_service: DataService, model_class: Type[MIPModel], ui_handler_class: Type[UIHandler]):
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
        if "optimization_status" not in st.session_state:
                st.session_state['optimization_status'] = ""
        if "optimization_count" not in st.session_state:
                st.session_state['optimization_count'] = 0
        if "df_report" not in st.session_state:
                st.session_state['df_report'] = None
        if "df_result" not in st.session_state:
                st.session_state['df_result'] = None

        # model
        self.model = model_class()

        # UI handler
        self.ui_handler = ui_handler_class()


    def start(self):
        self.ui_handler.render(
            validate_callback=self.validate_all_setting,
            optimize_callback=self.optimize
            )

    def add(self):
        # for customize constraints / vars / objs
        pass

    def validate_SOC_setting(self) -> bool:
        # params = st.session_state["params"]
        # input_lb = params["input_lb"]["value"]
        # input_ub = params["input_ub"]["value"]
        # input_soc_init = params["input_soc_init"]["value"]
        # input_soc_end = params["input_soc_end"]["value"]

        input_lb = st.session_state["input_lb"]
        input_ub = st.session_state["input_ub"]
        input_soc_init = st.session_state["input_soc_init"]
        input_soc_end = st.session_state["input_soc_end"]
        # check SOC setting
        if not (
            input_lb < input_ub and
            input_lb <= input_soc_init <= input_ub and
            input_lb <= input_soc_end <= input_ub
            ):
            return False
        return True

    def validate_bid_setting(self) -> bool:
        # params = st.session_state["params"]
        # df_ed_bid = params["ed_bid"]
        # opt_bid = params["cb_opt_bid"]["value"]

        df_ed_bid = st.session_state["ed_bid"]
        opt_bid = st.session_state["cb_opt_bid"]

        # trading-related rule
        if not verify_bid_rule(
            # df_ed_bid=df_ed_bid["data"],
            df_ed_bid=df_ed_bid,
            opt_bid=opt_bid
            ):
            return False
        return True
            # , "Check SOC boundary setting."
            # self.placeholder_warning.warning('Check trading scenario setting is correct.', icon=":warning:")
            # st.stop()

    def validate_tendered_cap_setting(self) -> bool:
        # params = st.session_state["params"]
        # df_ed_bid = params["ed_bid"]["data"]
        # cb_opt_tendered_cap = params["cb_opt_tendered_cap"]["value"]
        # cb_relax_tendered_step = params["cb_opt_tendered_cap"]["value"]
        # input_tendered_lb = params["input_tendered_lb"]["value"]
        # input_tendered_ub = params["input_tendered_ub"]["value"]

        df_ed_bid = st.session_state["ed_bid"]
        cb_opt_tendered_cap = st.session_state["cb_opt_tendered_cap"]#["value"]
        cb_relax_tendered_step = st.session_state["cb_opt_tendered_cap"]#["value"]
        input_tendered_lb = st.session_state["input_tendered_lb"]#["value"]
        input_tendered_ub = st.session_state["input_tendered_ub"]#["value"]

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
        # self.placeholder_warning.warning('Check tendered capacity setting is correct.(non-negativity / integrity / not in bound)', icon=":warning:")
        # st.stop()

    def validate_all_setting(self) -> Tuple[bool, str]:
        if not self.validate_SOC_setting():
            return False, "Check SOC boundary setting."
        if not self.validate_bid_setting():
            return False, "Check trading senario setting is correct."
        if not self.validate_tendered_cap_setting():
            return False, "Check tendered capacity setting is correct.(non-negativity / integrity / not in bound)"
        return True, "parameter setting is all good."

    def retrieve_result(self):
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
        result['ESS_is_charging'] = [v.x for v in model.b_chg]
        result['ESS_is_discharging'] = [v.x for v in model.b_dch]

        result['exceed_contract_capacity'] = [v.x for v in model.b_exceed]
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
            # max_seconds=st.session_state["params"]["input_max_sec"]["value"]
            max_seconds=st.session_state["input_max_sec"]#["value"]
            )

        # update session states
        st.session_state["optimization_status"] = status.name
        st.session_state["optimization_count"] += 1

        df_report, df_result = self.retrieve_result()
        return status.name, df_report, df_result

if __name__ == "__main__":
    st.set_page_config(page_title='Power Optimizer test(Cht)', layout="wide", page_icon='./img/favicon.png')

    data_service = DataService()
    model = ESSModel
    ui_handler = UIHandler
    optimizer = Optimizer(
        data_service=data_service,
        model_class=model,
        ui_handler_class=ui_handler
        )
    optimizer.start()

    # data_service = DataService()
    # data = DataService.load_sample_data()
    # params = DataService.load_sample_params()
    # builder = ESSModelBuilder()

    # # Optimizer.set_data(data=data)
    # # Optimizer.set_params(params=params)
    # # builder.set_data(data=data)
    # # builder.set_params(params=params)
    # model = builder.build()
    # result = model.optimize(max_seconds=builder.max_sec)

    # print(result)
