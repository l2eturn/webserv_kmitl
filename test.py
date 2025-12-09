import json
from datetime import datetime
import numpy as np
from scipy.optimize import minimize
import sqlite3 
import os 

# ====================================================
# TFN helpers
# ====================================================
def geom_mean_triplet(triplets):
    arr = np.array(triplets, dtype=float)
    gm = np.prod(arr, axis=0) ** (1.0 / arr.shape[0])
    return (float(gm[0]), float(gm[1]), float(gm[2]))

def mul_trip(a,b): return (a[0]*b[0], a[1]*b[1], a[2]*b[2])
def sub_trip(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def R_trip(t): return (t[0] + 4.0*t[1] + t[2]) / 6.0
def fmt_trip(t): return f"({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})"
def inv_trip(t):
    l,m,u = t; l=max(l,1e-12); m=max(m,1e-12); u=max(u,1e-12)
    return (1.0/u, 1.0/m, 1.0/l)
def div_trip(a, b):
    (la,ma,ua)=a; (lb,mb,ub)=b; lb=max(lb,1e-12); mb=max(mb,1e-12); ub=max(ub,1e-12)
    return (la/ub, ma/mb, ua/lb)
def _snap_diag_to_one(trip):
    l,m,u=trip
    if abs(l-1)<1e-9 and abs(m-1)<1e-9 and abs(u-1)<1e-9: return (1.0,1.0,1.0)
    return trip
def _fmt_num(x): return f"{x:.2f}".rstrip("0").rstrip(".") or "0"
def _fmt_tfn_smart(t): return f"({_fmt_num(t[0])},{_fmt_num(t[1])},{_fmt_num(t[2])})"

# ====================================================
# Align Helpers
# ====================================================
def align_BO_to_group(BO_list, expert_best_idx, group_best_idx):
    if expert_best_idx == group_best_idx:
        out = list(BO_list); out[group_best_idx] = (1.0,1.0,1.0); return out
    a_beg = BO_list[group_best_idx]
    out = [div_trip(BO_list[j], a_beg) for j in range(len(BO_list))]
    out[group_best_idx] = (1.0,1.0,1.0); return out

def align_OW_to_group(OW_list, expert_worst_idx, group_worst_idx):
    if expert_worst_idx == group_worst_idx:
        out = list(OW_list); out[group_worst_idx] = (1.0,1.0,1.0); return out
    a_WgWe = OW_list[group_worst_idx]
    out = [div_trip(OW_list[i], a_WgWe) for i in range(len(OW_list))]
    out[group_worst_idx] = (1.0,1.0,1.0); return out

def _is_unit_trip(t, tol=1e-6): return abs(t[0]-1) <= tol and abs(t[1]-1) <= tol and abs(t[2]-1) <= tol
def _trip_dist_to_unit(t): return (t[0]-1.0)**2 + (t[1]-1.0)**2 + (t[2]-1.0)**2

def infer_best_idx_from_BO(BO_list):
    if not BO_list: return 0
    for i, t in enumerate(BO_list):
        if _is_unit_trip(t): return i
    return int(np.argmin([_trip_dist_to_unit(t) for t in BO_list]))

def infer_worst_idx_from_OW(OW_list):
    if not OW_list: return 0
    for i, t in enumerate(OW_list):
        if _is_unit_trip(t): return i
    return int(np.argmin([_trip_dist_to_unit(t) for t in OW_list]))

THRESH = {
    3:{3:0.1667,4:0.1667,5:0.1667,6:0.1667,7:0.1667,8:0.1667,9:0.1667},
    4:{3:0.1121,4:0.1529,5:0.1898,6:0.2206,7:0.2527,8:0.2577,9:0.2683},
    5:{3:0.1354,4:0.1994,5:0.2306,6:0.2546,7:0.2716,8:0.2844,9:0.2960},
    6:{3:0.1330,4:0.1990,5:0.2643,6:0.3044,7:0.3144,8:0.3221,9:0.3262},
    7:{3:0.1294,4:0.2457,5:0.2819,6:0.3029,7:0.3144,8:0.3251,9:0.3403},
    8:{3:0.1309,4:0.2521,5:0.2958,6:0.3154,7:0.3408,8:0.3620,9:0.3657},
    9:{3:0.1359,4:0.2681,5:0.3062,6:0.3337,7:0.3517,8:0.3620,9:0.3662},
}

def _compute_CRI(BO, OW, worst_idx):
    n = len(BO)
    if n == 0: return []
    # Safety check: ถ้า worst_idx เกินขอบเขต ให้รีเซ็ตเป็น 0 (กัน Error)
    if worst_idx >= n: worst_idx = 0
    
    BW = BO[worst_idx]
    denom = R_trip(sub_trip(mul_trip(BW, BW), BW))
    if abs(denom) < 1e-12: denom = 1e-12
    out=[]
    for j in range(n):
        num = R_trip(sub_trip(mul_trip(BO[j], OW[j]), BW))
        out.append(num/denom)
    return out

def solve_n2_analytic(BO_agg, best_idx, worst_idx):
    assert len(BO_agg) == 2
    a_bw = max(1e-12, BO_agg[worst_idx][1])
    w_best = 1.0 / (1.0 + a_bw); w_worst = a_bw / (1.0 + a_bw)
    R = np.zeros(2, dtype=float); R[best_idx] = w_best; R[worst_idx] = w_worst
    return R

def run_group(group_name, criteria_names, E1_BO_raw, E1_OW_raw, E2_BO_raw, E2_OW_raw, E3_BO_raw, E3_OW_raw,
              E1_best_idx=None, E1_worst_idx=None, E2_best_idx=None, E2_worst_idx=None, E3_best_idx=None, E3_worst_idx=None,
              best_idx=None, worst_idx=None, # แก้ไข: เปลี่ยน Default เป็น None
              init_L=0.1, init_M=0.2, init_U=0.3, verbose=True):
    
    n = len(criteria_names)
    
    # --- Safety Check: ป้องกันข้อมูลไม่ครบ ---
    if len(E1_BO_raw) != n or len(E2_BO_raw) != n or len(E3_BO_raw) != n:
        print(f"Error: Data length mismatch in {group_name}. Expected {n}, got {len(E1_BO_raw)}")
        return None

    # หา Best/Worst ของแต่ละ Expert ถ้าไม่ได้ระบุ
    if E1_best_idx is None: E1_best_idx = infer_best_idx_from_BO(E1_BO_raw)
    if E1_worst_idx is None: E1_worst_idx = infer_worst_idx_from_OW(E1_OW_raw)
    if E2_best_idx is None: E2_best_idx = infer_best_idx_from_BO(E2_BO_raw)
    if E2_worst_idx is None: E2_worst_idx = infer_worst_idx_from_OW(E2_OW_raw)
    if E3_best_idx is None: E3_best_idx = infer_best_idx_from_BO(E3_BO_raw)
    if E3_worst_idx is None: E3_worst_idx = infer_worst_idx_from_OW(E3_OW_raw)

    # ★★★ แก้ไขจุดที่ Error: หา Best/Worst ของกลุ่มอัตโนมัติ ถ้าไม่ได้ระบุมา ★★★
    # ใช้ค่าของ Expert 1 เป็นตัวตั้งต้น (Master)
    if best_idx is None: best_idx = E1_best_idx
    if worst_idx is None: worst_idx = E1_worst_idx

    # Safety Check อีกรอบสำหรับ Index
    if best_idx >= n: best_idx = 0
    if worst_idx >= n: worst_idx = 0 if n > 0 else 0

    def _prepare_cri_pairs(BO_raw, OW_raw, worst_idx_expert):
        OWc = list(OW_raw)
        if 0 <= worst_idx_expert < len(OWc): OWc[worst_idx_expert] = (1.0,1.0,1.0)
        return list(BO_raw), OWc

    E1_BO_cri, E1_OW_cri = _prepare_cri_pairs(E1_BO_raw, E1_OW_raw, E1_worst_idx)
    E2_BO_cri, E2_OW_cri = _prepare_cri_pairs(E2_BO_raw, E2_OW_raw, E2_worst_idx)
    E3_BO_cri, E3_OW_cri = _prepare_cri_pairs(E3_BO_raw, E3_OW_raw, E3_worst_idx)

    E1_BO = align_BO_to_group(E1_BO_raw, E1_best_idx, best_idx)
    E1_OW = align_OW_to_group(E1_OW_raw, E1_worst_idx, worst_idx)
    E2_BO = align_BO_to_group(E2_BO_raw, E2_best_idx, best_idx)
    E2_OW = align_OW_to_group(E2_OW_raw, E2_worst_idx, worst_idx)
    E3_BO = align_BO_to_group(E3_BO_raw, E3_best_idx, best_idx)
    E3_OW = align_OW_to_group(E3_OW_raw, E3_worst_idx, worst_idx)

    BO_agg = []; OW_agg = []
    for i in range(n):
        b = geom_mean_triplet([E1_BO[i], E2_BO[i], E3_BO[i]])
        w = geom_mean_triplet([E1_OW[i], E2_OW[i], E3_OW[i]])
        if i == best_idx:  b = _snap_diag_to_one(b)
        if i == worst_idx: w = _snap_diag_to_one(w)
        BO_agg.append(b); OW_agg.append(w)

    CRI_E1_raw = _compute_CRI(E1_BO_cri, E1_OW_cri, E1_worst_idx)
    CRI_E2_raw = _compute_CRI(E2_BO_cri, E2_OW_cri, E2_worst_idx)
    CRI_E3_raw = _compute_CRI(E3_BO_cri, E3_OW_cri, E3_worst_idx)

    if n == 2:
        R_opt = solve_n2_analytic(BO_agg, best_idx, worst_idx)
        return {'k_star': 0.0, 'R_opt': R_opt, 'ranking': np.argsort(R_opt)[::-1], 'BO_agg': BO_agg, 'OW_agg': OW_agg, 'CRI': {'E1_raw': CRI_E1_raw, 'E2_raw': CRI_E2_raw, 'E3_raw': CRI_E3_raw}}

    # n >= 3 Optimization
    L, M, U = np.full(n, init_L), np.full(n, init_M), np.full(n, init_U)
    k_current = 1.0 

    def unpack(x): return x[:n], x[n:2*n], x[2*n:3*n], x[3*n]
    def objective(x): return unpack(x)[3]
    
    def build_constraints():
        cons=[]
        for j in range(n):
            Bj_l,Bj_m,Bj_u=BO_agg[j]; jW_l,jW_m,jW_u=OW_agg[j]
            cons+=[
                {'type':'ineq','fun':lambda x,j=j,Bj_u=Bj_u:unpack(x)[3]-abs(unpack(x)[0][best_idx]-Bj_u*unpack(x)[0][j])},
                {'type':'ineq','fun':lambda x,j=j,Bj_m=Bj_m:unpack(x)[3]-abs(unpack(x)[1][best_idx]-Bj_m*unpack(x)[1][j])},
                {'type':'ineq','fun':lambda x,j=j,Bj_l=Bj_l:unpack(x)[3]-abs(unpack(x)[2][best_idx]-Bj_l*unpack(x)[2][j])},
                {'type':'ineq','fun':lambda x,j=j,jW_u=jW_u:unpack(x)[3]-abs(unpack(x)[0][j]-jW_u*unpack(x)[0][worst_idx])},
                {'type':'ineq','fun':lambda x,j=j,jW_m=jW_m:unpack(x)[3]-abs(unpack(x)[1][j]-jW_m*unpack(x)[1][worst_idx])},
                {'type':'ineq','fun':lambda x,j=j,jW_l=jW_l:unpack(x)[3]-abs(unpack(x)[2][j]-jW_l*unpack(x)[2][worst_idx])},
                {'type':'ineq','fun':lambda x,j=j:unpack(x)[0][j]}, 
                {'type':'ineq','fun':lambda x,j=j:unpack(x)[1][j]-unpack(x)[0][j]},
                {'type':'ineq','fun':lambda x,j=j:unpack(x)[2][j]-unpack(x)[1][j]},
            ]
        cons.append({'type':'eq','fun':lambda x:np.sum((unpack(x)[0]+4*unpack(x)[1]+unpack(x)[2])/6)-1})
        return cons

    x0=np.concatenate([L,M,U,[k_current]])
    bounds=[(0,1)]*(3*n)+[(0,1)]
    
    try:
        res=minimize(objective,x0,method='SLSQP',bounds=bounds,constraints=build_constraints(), options={'maxiter':5000,'ftol':1e-12,'disp':False})
        if not res.success: return None
        L_opt,M_opt,U_opt,k_opt=unpack(res.x)
        R_opt=(L_opt+4*M_opt+U_opt)/6
        return {'k_star':float(k_opt), 'R_opt':R_opt, 'ranking':np.argsort(R_opt)[::-1], 'BO_agg':BO_agg, 'OW_agg':OW_agg, 'CRI':{'E1_raw':CRI_E1_raw,'E2_raw':CRI_E2_raw,'E3_raw':CRI_E3_raw}}
    except Exception as e:
        print(f"Optimization error: {e}")
        return None

# ====================================================
# Main Save Function
# ====================================================

def run_full_fbwm_and_save_new(db_dir, project_name, input_data=None):
    globals_all = []
    
    def process_group(group_key, criteria_list):
        if not input_data or group_key not in input_data: 
            print(f"No data for group {group_key}")
            return None
        
        g_data = input_data[group_key]
        try:
            E1_BO = g_data["E1"]["BO"]; E1_OW = g_data["E1"]["OW"]
            E2_BO = g_data["E2"]["BO"]; E2_OW = g_data["E2"]["OW"]
            E3_BO = g_data["E3"]["BO"]; E3_OW = g_data["E3"]["OW"]
            
            # Safety Check
            n = len(criteria_list)
            if len(E1_BO) != n: 
                print(f"Skipping {group_key}: Expected {n}, got {len(E1_BO)}")
                return None

            # ไม่ต้องส่ง best_idx/worst_idx ไป ระบบจะหาเองจากข้อมูล
            return run_group(f"Group {group_key}", criteria_list, E1_BO, E1_OW, E2_BO, E2_OW, E3_BO, E3_OW, verbose=False)
        except KeyError: 
            print(f"KeyError in group {group_key}")
            return None

    # 1. Main
    main_criteria = ["C1", "C2", "C3", "C4", "C5"]
    main_res = process_group("Main", main_criteria)
    if not main_res: return None

    # 2. Subs
    sub_groups = {
        "C1": ["s11", "s12", "s13"],
        "C2": ["s21", "s22", "s23", "s24"],
        "C3": ["s31", "s32", "s33", "s34"],
        "C4": ["s41", "s42", "s43"],
        "C5": ["s51", "s52", "s53"]
    }

    for idx, (main_code, sub_list) in enumerate(sub_groups.items()):
        if idx >= len(main_res['R_opt']): continue
        main_weight = float(main_res['R_opt'][idx])
        sub_res = process_group(main_code, sub_list)
        
        if sub_res:
            local_weights = sub_res['R_opt']
            for i, w in enumerate(local_weights):
                global_w = main_weight * float(w)
                globals_all.append((sub_list[i], global_w))
    
    # 3. Save
    if not globals_all: return None
    
    GLOBAL_NAMES = [n for (n, _) in globals_all]
    GLOBAL_WEIGHTS = [w for (_, w) in globals_all]
    s = sum(GLOBAL_WEIGHTS)
    if s > 0: GLOBAL_WEIGHTS = [w/s for w in GLOBAL_WEIGHTS]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = "".join(c if c.isalnum() else "_" for c in project_name).strip("_") or "project"
    db_filename = f"{timestamp}_{clean_name}.db"
    db_path = os.path.join(db_dir, db_filename)

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS weights (id INTEGER PRIMARY KEY AUTOINCREMENT, criterion TEXT NOT NULL, weight REAL NOT NULL)")
        for name, w in zip(GLOBAL_NAMES, GLOBAL_WEIGHTS):
            cur.execute("INSERT INTO weights (criterion, weight) VALUES (?, ?)", (name, w))
        cur.execute("CREATE TABLE IF NOT EXISTS project_info (key TEXT PRIMARY KEY, value TEXT)")
        cur.execute("INSERT OR REPLACE INTO project_info VALUES (?, ?)", ("project_name", project_name))
        cur.execute("INSERT OR REPLACE INTO project_info VALUES (?, ?)", ("created_at", datetime.now().isoformat()))
        conn.commit(); conn.close()
        return db_filename
    except Exception as e:
        print(f"DB Error: {e}")
        return None