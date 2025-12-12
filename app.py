from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import json
import os
import sqlite3
from datetime import datetime

# Import modules
from logic import calculate_topsis
# Import functions from test.py
from test import THRESH, _compute_CRI, R_trip, mul_trip, sub_trip, run_full_fbwm_and_save_new 

app = Flask(__name__)
app.secret_key = 'super_secret_key_change_this' 

DB_DIR = "weights_projects"
os.makedirs(DB_DIR, exist_ok=True) 
DEFAULT_JSON_PATH = "weights_global_topsis.json"

# --- Config ข้อมูล (CRITERIA_INFO: อัปเดตชื่อเกณฑ์ใหม่และเพิ่มหน่วย) ---
CRITERIA_INFO = {
    "groups": {
        "C1": {"name": "ต้นทุน (Cost)", "members": ["s11", "s12", "s13"]},
        "C2": {"name": "ระยะเวลา (Time)", "members": ["s21", "s22", "s23", "s24"]},
        "C3": {"name": "ลักษณะของสินค้า (Product)", "members": ["s31", "s32", "s33", "s34"]},
        "C4": {"name": "ความน่าเชื่อถือ (Reliability)", "members": ["s41", "s42", "s43"]},
        # แก้ไขชื่อกลุ่ม C5
        "C5": {"name": "ความเสี่ยงด้านความเสียหาย (Risk of Damage)", "members": ["s51", "s52", "s53"]},
    },
    "names": {
        # ★★★ คำอธิบายเกณฑ์ที่อัปเดตใหม่ ★★★
        "s11": "ค่าภาษีนำเข้า (Tax)", 
        "s12": "ค่าใช้จ่ายในการขนส่งจากท่าถึงท่า", 
        "s13": "ค่าใช้จ่ายในการขนส่งจากท่าถึงไซต์งาน",
        "s21": "ระยะเวลาขนส่งสินค้าจากโกดังของลูกค้าถึงท่า", 
        "s22": "ระยะเวลานำเข้าสินค้าจากท่าถึงท่า", 
        "s23": "ระยะเวลานำเข้าสินค้าจากท่าถึงไซต์งาน", 
        "s24": "ระยะเวลารอการผ่านศุลกากร",
        "s31": "ขนาดสินค้า (Size)", 
        "s32": "น้ำหนักสินค้า (Weight)", 
        "s33": "ปริมาณการขนส่ง (Quantity)", 
        "s34": "บรรจุภัณฑ์ (Packaging Prioritise)",
        "s41": "ตรงต่อเวลา (On time)",
        "s42": "การติดตามสินค้า (Tracking)", 
        "s43": "การจัดการเหตุฉุกเฉิน (Emergency Management)",
        "s51": "ความเสียหายระหว่างโหมดการขนส่ง",
        "s52": "ความเสียหายจากการขนส่ง",
        "s53": "ความเสี่ยงจากอุบัติเหตุระหว่างทาง"
    },
    "units": {
        "s11": "บาท", "s12": "บาท", "s13": "บาท",
        "s21": "วัน", "s22": "วัน", "s23": "วัน", "s24": "วัน",
        "s31": "QbM", "s32": "Kg", "s33": "ชิ้น", "s34": "คะแนน 1-5",
        "s41": "คะแนน 1-5", "s42": "คะแนน 1-5", "s43": "คะแนน 1-5",
        "s51": "คะแนน 1-5", "s52": "คะแนน 1-5", "s53": "คะแนน 1-5",
    }
}
# ----------------------------------------------------
# --- Helper Functions (ต้องวางไว้ก่อน Routes เสมอ) ---
# ----------------------------------------------------
def get_all_projects():
    projects = []
    try:
        for filename in os.listdir(DB_DIR):
            if filename.endswith(".db"):
                parts = filename.split('_')
                formatted_date = ""
                if len(parts) >= 2:
                    date_part = parts[0]
                    time_part = parts[1]
                    try:
                        dt_obj = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")
                        formatted_date = dt_obj.strftime("%d/%m/%Y %H:%M")
                    except ValueError:
                        formatted_date = f"{date_part}-{time_part}"
                
                if len(parts) >= 3:
                    raw_name = "_".join(parts[2:]) 
                    clean_name = raw_name[:-3].replace("_", " ")
                else:
                    clean_name = filename[:-3].replace("_", " ")

                projects.append({
                    "id": filename, 
                    "name": clean_name,
                    "file": filename,
                    "display_date": formatted_date
                })
    except FileNotFoundError:
        pass
    projects.sort(key=lambda x: x['file'], reverse=True)
    return projects

def get_latest_project_db():
    projects = get_all_projects()
    return os.path.join(DB_DIR, projects[0]['file']) if projects else None

def load_weights_from_db(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT criterion, weight FROM weights")
        data = cur.fetchall()
        conn.close()
        criteria = [d[0] for d in data]; weights = [d[1] for d in data]
        all_s_names = list(CRITERIA_INFO['names'].keys())
        weights_map = dict(zip(criteria, weights))
        ordered_criteria = []; ordered_weights = []
        for s in all_s_names:
            if s in weights_map:
                ordered_criteria.append(s); ordered_weights.append(weights_map[s])
        return ordered_criteria, ordered_weights
    except Exception as e: return [], []

def load_criteria_data(path=DEFAULT_JSON_PATH):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("criteria", []), data.get("weights", [])
        except Exception as e: pass
    latest_db = get_latest_project_db()
    if latest_db: return load_weights_from_db(latest_db)
    criteria = list(CRITERIA_INFO['names'].keys())
    return criteria, [1/len(criteria)] * len(criteria)

def get_ranked_criteria(path=DEFAULT_JSON_PATH):
    if path.endswith(".db"): criteria, weights = load_weights_from_db(path)
    else: criteria, weights = load_criteria_data(path)
    weights_map = dict(zip(criteria, weights))
    ranked_groups = []
    main_scores = []
    for cid, info in CRITERIA_INFO['groups'].items():
        total_w = sum(weights_map.get(m, 0) for m in info['members'])
        main_scores.append({"code": cid, "name": info['name'], "score": total_w})
    main_scores.sort(key=lambda x: x['score'], reverse=True)
    ranked_groups.append({"title": "1. น้ำหนัก Main Criteria", "members": main_scores, "main": True})
    group_ids = ["C1", "C2", "C3", "C4", "C5"]
    for idx, cid in enumerate(group_ids):
        info = CRITERIA_INFO['groups'][cid]
        subs = []
        for mid in info['members']:
            subs.append({"code": mid, "name": CRITERIA_INFO['names'].get(mid, mid), "score": weights_map.get(mid, 0)})
        subs.sort(key=lambda x: x['score'], reverse=True)
        ranked_groups.append({"title": f"{idx+2}. น้ำหนัก Sub - {info['name']}", "members": subs, "main": False})
    return ranked_groups

def get_full_structure():
    structure = []
    # 1. Main Criteria
    main_members = list(CRITERIA_INFO['groups'].keys()) 
    
    structure.append({
        "id": "Main",
        "name": "1. Main (หลัก)",
        "members": main_members
    })
    # 2. Sub Criteria
    criteria_order = ["C1", "C2", "C3", "C4", "C5"]
    for cid in criteria_order:
        group_info = CRITERIA_INFO["groups"][cid]
        structure.append({
            "id": cid,
            "name": f"{cid} - {group_info['name']}",
            "members": group_info["members"]
        })
    return structure

# ----------------------------------------------------
## Routes
# ----------------------------------------------------

@app.route('/')
def index(): return redirect(url_for('project_select'))

@app.route('/info')
def info_page():
    project_name = session.get('current_project_name', 'รายงานผลการตัดสินใจ')
    db_path = session.get('current_project_db')
    path_to_load = db_path if db_path and os.path.exists(db_path) else DEFAULT_JSON_PATH
    ranking_data = get_ranked_criteria(path_to_load) 
    return render_template('info.html', ranking_data=ranking_data, project_name=project_name)

@app.route('/project_select')
def project_select():
    class Utils:
        @staticmethod
        def now(): return datetime.now()
    app.jinja_env.globals['now'] = Utils.now
    projects = get_all_projects()
    latest_project = projects[0] if projects else None
    latest_ranking_data = get_ranked_criteria(os.path.join(DB_DIR, latest_project['file'])) if latest_project else None
    
    if latest_project and 'current_project_name' not in session:
        session['current_project_name'] = latest_project['name']

    return render_template('project_select.html', projects=projects, latest_project=latest_project, latest_ranking_data=latest_ranking_data)

@app.route('/use_project/<filename>')
def use_project(filename):
    db_path = os.path.join(DB_DIR, filename)
    if not os.path.exists(db_path): return "Error: Project file not found.", 404
    
    all_projects = get_all_projects()
    selected_project = next((p for p in all_projects if p['file'] == filename), None)
    
    project_name = selected_project['name'] if selected_project else filename[:-3].replace("_", " ")

    session['current_project_db'] = db_path
    session['current_project_name'] = project_name
    
    return redirect(url_for('input_form'))

@app.route('/form')
def input_form():
    db_path = session.get('current_project_db')
    
    if 'current_project_name' not in session:
        session['current_project_name'] = 'รายงานผลการตัดสินใจ'

    if db_path: criteria, weights = load_weights_from_db(db_path)
    else: criteria, weights = load_criteria_data(DEFAULT_JSON_PATH) 
    if not criteria or len(criteria) == 0: return redirect(url_for('project_select')) 
    session['current_weights'] = weights 
    session['current_criteria'] = criteria 
    
    criteria_names = CRITERIA_INFO['names']
    criteria_units = CRITERIA_INFO['units'] # ดึงหน่วยวัด

    return render_template('input_form.html', criteria=criteria, names=criteria_names, units=criteria_units)


#@app.route('/calculate', methods=['POST'])
#def calculate():
#    criteria = session.get('current_criteria')
#    weights = session.get('current_weights')
#    if not criteria or not weights: return redirect(url_for('project_select')) 
    
#    modes = ["Air", "Sea", "Road"]
#    user_matrix = []
#    try:
#        for mode in modes:
#            row = []
#            for c in criteria:
#                val = request.form.get(f"{mode}_{c}")
#                if val is None: raise ValueError(f"Missing input for {mode}_{c}")
#                row.append(float(val))
#            user_matrix.append(row)
        
#        results = calculate_topsis(user_matrix, weights, criteria)
        
#        db_path = session.get('current_project_db')
#        ranking_data = []
#        if db_path and os.path.exists(db_path): ranking_data = get_ranked_criteria(db_path)
        
#        project_name = session.get('current_project_name', 'รายงานผลการตัดสินใจ') 
#        print_date = datetime.now().strftime("%d/%m/%Y %H:%M")
        
#        criteria_names = CRITERIA_INFO['names']
#        criteria_units = CRITERIA_INFO['units']

#        return render_template('result.html', results=results, ranking_data=ranking_data, user_matrix=user_matrix, criteria=criteria, criteria_names=criteria_names, criteria_units=criteria_units, modes=modes, project_name=project_name, print_date=print_date,matrix=matrix,criteria_list=criteria_names)
#    except Exception as e: return f"Error in TOPSIS calculation: {e}", 400
@app.route('/calculate', methods=['POST'])
def calculate():
    criteria = session.get('current_criteria')
    weights = session.get('current_weights')
    if not criteria or not weights: return redirect(url_for('project_select')) 
    
    modes = ["Air", "Sea", "Road"]
    user_matrix = []
    
    try:
        # 1. ดึงข้อมูลจากฟอร์ม
        for mode in modes:
            row = []
            for c in criteria:
                val = request.form.get(f"{mode}_{c}")
                if val is None: raise ValueError(f"Missing input for {mode}_{c}")
                row.append(float(val))
            user_matrix.append(row)
        
        # 2. คำนวณ TOPSIS
        results = calculate_topsis(user_matrix, weights, criteria)
        
        # 3. เตรียมข้อมูลอื่นๆ
        db_path = session.get('current_project_db')
        ranking_data = []
        if db_path and os.path.exists(db_path): ranking_data = get_ranked_criteria(db_path)
        
        project_name = session.get('current_project_name', 'รายงานผลการตัดสินใจ') 
        print_date = datetime.now().strftime("%d/%m/%Y %H:%M")
        
        criteria_names = CRITERIA_INFO['names']
        criteria_units = CRITERIA_INFO['units']

        # --- ส่วนที่เพิ่มเพื่อแก้ Error และทำกราฟ ---
        # สร้างลิสต์ชื่อเกณฑ์ภาษาไทยเรียงตามลำดับสำหรับกราฟ
        # (ถ้าส่ง criteria_names ที่เป็น dict ไปตรงๆ กราฟอาจจะงงได้)
        chart_labels = []
        for c in criteria:
            name = criteria_names.get(c, c) # ดึงชื่อไทย ถ้าไม่มีใช้รหัสเดิม
            chart_labels.append(name)

        return render_template('result.html', 
                               results=results, 
                               ranking_data=ranking_data, 
                               user_matrix=user_matrix, 
                               criteria=criteria, 
                               criteria_names=criteria_names, 
                               criteria_units=criteria_units, 
                               modes=modes, 
                               project_name=project_name, 
                               print_date=print_date,
                               
                               # !!! จุดที่แก้ !!!
                               matrix=user_matrix,          # เปลี่ยนจาก matrix เป็น user_matrix
                               criteria_list=chart_labels   # ส่งชื่อไทยไปให้กราฟ
                               )
                               
    except Exception as e: return f"Error in TOPSIS calculation: {e}", 400
@app.route('/new_project_input')
def new_project_input():
    structure = get_full_structure()
    # การแก้ไข: รวมชื่อเกณฑ์หลัก (C1-C5) และเกณฑ์ย่อย (s11-s53) เข้าด้วยกัน
    criteria_names = {**CRITERIA_INFO['names'], **{k: v['name'] for k, v in CRITERIA_INFO['groups'].items()}}
    
    return render_template('new_project.html', 
                            structure=structure, 
                            now=datetime.now, 
                            criteria_names=criteria_names)
    
@app.route('/api/check_cri', methods=['POST'])
def check_cri():
    data = request.json
    BO_raw = data.get('BO_raw')
    OW_raw = data.get('OW_raw')
    worst_idx = data.get('worst_idx')
    n = len(BO_raw)

    if n < 3:
        return jsonify({"success": True, "max_cri": 0.0, "threshold": 0.0, "passed": True, "n": n, "cri_details": []})
        
    try:
        OWc = list(OW_raw)
        if 0 <= worst_idx < n: OWc[worst_idx] = (1.0, 1.0, 1.0) 
        CRI_raw = _compute_CRI(BO_raw, OWc, worst_idx)
        
        CRI = [float(x) if x >= 0 else 0.0 for x in CRI_raw]
        CRI[worst_idx] = 0.0 
        max_cri = max(CRI)
        mvals = [t[1] for t in BO_raw] + [t[1] for t in OW_raw]
        scale = int(round(max(mvals))); scale = min(9, max(3, scale))
        threshold = THRESH[scale][n]
        passed = (max_cri <= threshold)

        return jsonify({
            "success": True, "max_cri": max_cri, "threshold": threshold, 
            "passed": passed, "scale": scale, "n": n, "cri_details": CRI
        })
    except Exception as e: return jsonify({"success": False, "error": str(e)}), 400

@app.route('/create_project', methods=['POST'])
def create_project():
    project_name = request.form.get('project_name', 'New Project')
    
    input_data = {}
    structure = get_full_structure()
    experts = ["E1", "E2", "E3"]

    try:
        for group in structure:
            group_id = group['id']
            criteria_list = group['members']
            
            input_data[group_id] = { "E1": {}, "E2": {}, "E3": {} }
            
            for eid in experts:
                bo_list = []; ow_list = []
                for c in criteria_list:
                    bo_l = float(request.form.get(f"{eid}_{c}_BO_l"))
                    bo_m = float(request.form.get(f"{eid}_{c}_BO_m"))
                    bo_u = float(request.form.get(f"{eid}_{c}_BO_u"))
                    bo_list.append((bo_l, bo_m, bo_u))
                    
                    ow_l = float(request.form.get(f"{eid}_{c}_OW_l"))
                    ow_m = float(request.form.get(f"{eid}_{c}_OW_m"))
                    ow_u = float(request.form.get(f"{eid}_{c}_OW_u"))
                    ow_list.append((ow_l, ow_m, ow_u))
                
                input_data[group_id][eid]["BO"] = bo_list
                input_data[group_id][eid]["OW"] = ow_list

    except (TypeError, ValueError) as e:
        return f"Error reading form data: {e} <br>กรุณากรอกข้อมูลให้ครบทุกกลุ่ม", 400

    db_filename = run_full_fbwm_and_save_new(DB_DIR, project_name, input_data=input_data)
    if not db_filename: return "Error: F-BWM calculation failed.", 500
    
    db_path = os.path.join(DB_DIR, db_filename)
    session['current_project_db'] = db_path
    session['current_project_name'] = project_name
    return redirect(url_for('project_summary'))

@app.route('/project_summary')
def project_summary():
    db_path = session.get('current_project_db')
    project_name = session.get('current_project_name', 'New Project')
    if not db_path or not os.path.exists(db_path): return redirect(url_for('project_select')) 
    ranking_data = get_ranked_criteria(db_path)
    return render_template('project_summary.html', project_name=project_name, ranking_data=ranking_data)

@app.route('/delete_project/<filename>')
def delete_project(filename):
    file_path = os.path.join(DB_DIR, filename)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            if session.get('current_project_db') == file_path:
                session.pop('current_project_db', None)
                session.pop('current_project_name', None)
    except Exception as e: print(f"Error: {e}")
    return redirect(url_for('project_select'))

if __name__ == '__main__':
    app.run(debug=True)