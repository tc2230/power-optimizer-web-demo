

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
