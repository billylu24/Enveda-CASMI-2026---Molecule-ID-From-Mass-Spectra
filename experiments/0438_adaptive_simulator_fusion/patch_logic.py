"""
Adaptive Simulator Fusion Logic for CASMI26 (0.438 -> 0.44x baseline upgrade)

Background:
In the 0.438 reproduction (Cell 20 of notebook.py):
- When CELL19_ICE = False:
  - If gl_of(mid) is present (i.e. [M+H]+ spectrum), ICE scores are completely zeroed:
    o = gl_fuse.rerank_multi(fsm, fk, fsc, forms, [{}, gl], [ICE_LAM, GL_LAM])
  - If gl_of(mid) is absent, ICE scores are applied at lam=1.0:
    o = ice_fuse.rerank(fsm, fk, fsc, forms, ice, lam=ICE_LAM)
- Hard equal fusion (b420: CELL19_ICE = True, ICE=1.0, GL=1.0) scored 0.402 (severe drop).
- GL-only (b416: ICE=0.0, GL=1.0) scored 0.417.
- Blind-fallback (v14: ICE=0.0 when GL present, ICE=1.0 when GL absent) scored 0.438.

Innovation:
Instead of a binary hard switch, we implement:
1. Confidence-Aware Adaptive ICE Weight (soft ensemble when GL margin is small):
   - For a given molecule, calculate the discrimination margin of GLACIER across its top candidates:
     gl_vals = [gl[k] for k in fk if k in gl and gl[k] is not None]
     gl_margin = (max(gl_vals) - sorted(gl_vals)[-2]) if len(gl_vals) >= 2 else 0.0
   - If GL is highly confident (margin >= tau_high): lam_ice = 0.0 (trust GL completely, exactly as 0.438 does).
   - If GL is uncertain (margin < tau_low): lam_ice = min_ice_weight (e.g. 0.20~0.25).
     This introduces gentle complementary evidence from ICEBERG without overwhelming GLACIER.
2. Top-1 Conservative Anchor (RRF Guard):
   - If the main engines (v4n + Engine 2) have an overwhelming consensus on Top-1:
     rrf_margin = fsc[0] - fsc[1]
     If rrf_margin exceeds a safe guard threshold, preserve Top-1 from being demoted by simulator noise.
"""

def compute_gl_margin(gl_dict, candidate_keys):
    """Compute the score margin between top-1 and top-2 GLACIER candidates."""
    if not gl_dict or not isinstance(gl_dict, dict):
        return 0.0
    scores = [gl_dict[k] for k in candidate_keys if k in gl_dict and gl_dict[k] is not None]
    if len(scores) < 2:
        return 0.0
    sorted_s = sorted(scores, reverse=True)
    return float(sorted_s[0] - sorted_s[1])

def get_adaptive_weights(gl_dict, ice_dict, candidate_keys, 
                         gl_lam_base=1.0, 
                         ice_soft_weight=0.20, 
                         gl_margin_threshold=0.03):
    """
    Determine (lam_ice, lam_gl) based on GLACIER confidence.
    """
    has_gl = bool(gl_dict and any(v is not None for v in gl_dict.values()))
    has_ice = bool(ice_dict and any(v is not None for v in ice_dict.values()))
    
    if not has_gl:
        # GLACIER blind spot (e.g. negative or non-[M+H]+): full ICE fallback
        return (1.0 if has_ice else 0.0, 0.0)
    
    # GLACIER is available
    if not has_ice:
        return (0.0, gl_lam_base)
        
    margin = compute_gl_margin(gl_dict, candidate_keys)
    if margin < gl_margin_threshold:
        # GLACIER has low discrimination among isomers -> gentle ICE assistance
        return (ice_soft_weight, gl_lam_base)
    else:
        # GLACIER is confident -> pure GLACIER (exact 0.438 behavior)
        return (0.0, gl_lam_base)
