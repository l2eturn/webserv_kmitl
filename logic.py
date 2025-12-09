import numpy as np

# กำหนด B (Benefit - ยิ่งมากยิ่งดี) หรือ C (Cost - ยิ่งน้อยยิ่งดี)
# อ้างอิงจาก Log: s33, s41, s42, s43 เป็น Benefit (B) นอกนั้น Cost (C)
IMPACTS_BY_NAME = {
    "s11": "C", "s12": "C", "s13": "C",
    "s21": "C", "s22": "C", "s23": "C", "s24": "C",
    "s31": "C", "s32": "C", "s33": "B", "s34": "C",
    "s41": "B", "s42": "B", "s43": "B",
    "s51": "C", "s52": "C", "s53": "C",
}

def get_impact_list(criteria_names):
    """สร้าง List ของ B/C ตามลำดับ Criteria ที่ส่งมา"""
    impacts = []
    for c in criteria_names:
        im = IMPACTS_BY_NAME.get(c, "B").upper() # Default เป็น Benefit กันเหนียว
        impacts.append(im)
    return impacts

def calculate_topsis(matrix, weights, criteria_names):
    """
    matrix: List of Lists (3 modes x N criteria)
    weights: List of floats
    return: ผลลัพธ์พร้อมนำไปแสดงผล
    """
    modes = ["Air", "Sea", "Road"]
    impacts = get_impact_list(criteria_names)
    
    M = np.array(matrix, dtype=float)
    w = np.array(weights, dtype=float).reshape(1, -1)
    
    # 1. Vector Normalization
    denom = np.sqrt((M ** 2).sum(axis=0))
    denom[denom == 0.0] = 1.0  # ป้องกันหาร 0
    R = M / denom

    # 2. Weighted Normalization
    V = R * w

    # 3. Find Ideal Best/Worst
    m, n = M.shape
    ideal_best = np.zeros(n)
    ideal_worst = np.zeros(n)
    
    for j in range(n):
        col = V[:, j]
        if impacts[j] == "B":
            ideal_best[j] = np.max(col)
            ideal_worst[j] = np.min(col)
        else: # Cost (C)
            ideal_best[j] = np.min(col)
            ideal_worst[j] = np.max(col)

    # 4. Calculate Distances
    d_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

    # 5. Closeness Coefficient
    denom_dm = d_plus + d_minus
    denom_dm[denom_dm == 0.0] = 1e-12
    cc = d_minus / denom_dm

    # จัดเตรียมข้อมูลส่งกลับ
    results = []
    for i, mode in enumerate(modes):
        results.append({
            "mode": mode,
            "score": cc[i],
            "rank": 0
        })
    
    # Sort ตามคะแนนมากไปน้อย
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # ใส่ Ranking
    for idx, res in enumerate(results):
        res["rank"] = idx + 1
        
    return results