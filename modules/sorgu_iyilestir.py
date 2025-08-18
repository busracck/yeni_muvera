
# modules/sorgu_iylestir.py — verbose + clamp non-improvements to 0% change
from __future__ import annotations
import os, json, re, time
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ============== CONFIG (edit here) ==============
MIN_IMPROVE = 0.0003               # ~0.03% absolute relative improvement
MAX_ATTEMPTS = 3                   # try up to N; if not improved, add anyway with 0% change
ONLY_IMPROVED = False              # do NOT skip non-improved rows
OLLAMA_MODEL = "gemma3:4b"
# ===============================================

# ---- paths ----
try:
    from config import output_dir as _OUT
except Exception:
    _OUT = os.path.join("data", "output")
os.makedirs(_OUT, exist_ok=True)

SORGU_IN_CSV  = os.path.join(_OUT, "icerik_sorgu_top10.csv")
SORGU_OUT_CSV = os.path.join(_OUT, "icerik_sorgu_iyilestirme.csv")

# ---- model ----
try:
    from config import model as _cfg_model  # instance or callable
    st_model = _cfg_model() if callable(_cfg_model) else _cfg_model
except Exception:
    st_model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")

# prompt builder (use user's module if available)
try:
    from prompt.sorgu_prompt import generate_sorgu_prompt as _gen_sorgu_prompt
except Exception:
    _gen_sorgu_prompt = None

# ============== UTIL ==============
def now() -> str:
    return time.strftime("%H:%M:%S", time.localtime())

def fmt_sec(s: float) -> str:
    if s < 1: return f"{s*1000:.0f} ms"
    m, r = divmod(s, 60)
    return f"{int(m)}m {r:.1f}s" if m >= 1 else f"{s:.2f} s"

def _read_csv_robust(path: str) -> pd.DataFrame:
    t0 = time.time()
    print(f"[{now()}] ⬇️  Reading CSV: {path}", flush=True)
    for enc in (None, "utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc) if enc else pd.read_csv(path)
            print(f"[{now()}] ✅ CSV loaded ({len(df)} rows) in {fmt_sec(time.time()-t0)}", flush=True)
            return df
        except Exception:
            continue
    raise RuntimeError(f"CSV okunamadı: {path}")

def _pick_col(df: pd.DataFrame, names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns: return n
        if n.lower() in low: return low[n.lower()]
    def norm(s): return re.sub(r"[İIı]", "i", s, flags=re.I).lower()
    norm_map = {norm(c): c for c in df.columns}
    for n in names:
        if norm(n) in norm_map: return norm_map[norm(n)]
    return None

def _norm_score(x) -> float:
    s = str(x if x is not None else "").replace("%","").replace(",",".").strip()
    m = re.findall(r"[-+]?\d*\.?\d+", s)
    if not m: return 0.0
    v = float(m[0])
    return round(v/100.0, 6) if v > 1.5 else round(v, 6)

def _similarity(a_text: str, b_text: str) -> float:
    if not a_text or not b_text: return 0.0
    a = st_model.encode(a_text, convert_to_tensor=True, normalize_embeddings=True)
    b = st_model.encode(b_text, convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(a, b).item())

def _run_llm(prompt: str) -> str:
    from ollama import chat
    t0 = time.time()
    print(f"[{now()}] 🔁 LLM call → {OLLAMA_MODEL} (chars: {len(prompt)})", flush=True)
    resp = chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    out = resp.get("message", {}).get("content", str(resp))
    print(f"[{now()}] ✅ LLM done in {fmt_sec(time.time()-t0)}", flush=True)
    return out

def _parse_llm_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m: raise ValueError("LLM yanıtında JSON bulunamadı.")
    return json.loads(m.group(0))

# ---- prompt builders ----
SORGU_SYS = (
    "Sen bir SEO ve içerik geliştirme uzmanısın. "
    "Görevin, kullanıcı sorgusuna göre mevcut metni küçük dokunuşlarla iyileştirmek. "
    "h1/h2: sorguyu doğrudan karşılayan başlık; p/div: +5–10 kelime; li: +1–2 kelime. "
    "Anlamı bozma, pazarlama klişeleri ekleme. Sadece geçerli JSON döndür."
)
SORGU_HUM = """
Girdi:
Kullanıcı Sorgusu: "{query}"
Mevcut İçerik: "{current}"
HTML Bölümü: "{tag}"
Eski Skor: {old}

Beklenen çıktı (JSON):
{{
  "Kullanıcı Sorgusu": "{query}",
  "Eski Metin": "{current}",
  "Geliştirilmiş Metin": "Buraya geliştirilmiş hali gelecek",
  "HTML Bölümü": "{tag}"
}}
""".strip()

def _build_sorgu_prompt(query, current, tag, old):
    if _gen_sorgu_prompt:
        try:
            return _gen_sorgu_prompt(query, current, tag, old)
        except Exception:
            pass
    return f"{SORGU_SYS}\n{SORGU_HUM}".format(query=query, current=current, tag=tag, old=old)

# ---- core improve ----
def _try_improve(query_text, current_text, html_tag, old_score,
                 min_improve=MIN_IMPROVE, max_attempts=MAX_ATTEMPTS):
    best_text = current_text
    best_score = old_score if old_score > 0 else _similarity(query_text, current_text)

    for attempt in range(1, max_attempts+1):
        print(f"    [{now()}] attempt {attempt}/{max_attempts} | baseline={best_score:.4f}", flush=True)
        prompt = _build_sorgu_prompt(query_text, best_text, html_tag, best_score)
        data = _parse_llm_json(_run_llm(prompt))

        cand = data.get("Geliştirilmiş Metin")
        if not isinstance(cand, str) or not cand.strip():
            print("    ↪️  LLM returned empty candidate; keeping current text", flush=True)
            cand = best_text

        new_score = _similarity(query_text, cand)
        print(f"    [{now()}] scored new={new_score:.4f} (delta={(new_score-best_score):+.4f})", flush=True)

        if new_score >= best_score * (1.0 + min_improve):
            print(f"    🎯 improved ≥ {min_improve*100:.3f}% — accepting", flush=True)
            return cand, new_score

        if new_score > best_score:
            print("    ⬆️  slight improvement; updating baseline and retrying", flush=True)
            best_text, best_score = cand, new_score

    print("⚖️  no sufficient improvement; returning best so far", flush=True)
    return best_text, best_score

# ============== FLOW ==============
def run_sorgu_flow(min_improve=MIN_IMPROVE, max_attempts=MAX_ATTEMPTS, only_improved=ONLY_IMPROVED) -> str:
    t_flow = time.time()
    df = _read_csv_robust(SORGU_IN_CSV)

    c_query = _pick_col(df, ["Kullanıcı Sorgusu","Kullanici Sorgusu","Sorgu","Query","Aranan Sorgu"])
    c_html  = _pick_col(df, ["HTML Kaynağı","HTML Kaynagi","HTML Bölümü","HTML Section"])
    c_text  = _pick_col(df, ["Web İçeriği","Web Icerigi","İçerik","Icerik","Metin","Content"])
    c_score = _pick_col(df, ["Benzerlik Skoru","Skor","Score","Similarity Score","similarity_score"])
    for need, name in [(c_query,"Kullanıcı Sorgusu"), (c_html,"HTML Kaynağı"), (c_text,"Web İçeriği")]:
        if not need: raise KeyError(f"Eksik kolon: {name}")

    total = len(df)
    kept = improved = 0
    rows = []

    print(f"[{now()}] 🚀 SORGU flow start | rows={total} | MIN_IMPROVE={min_improve} | MAX_ATTEMPTS={max_attempts}", flush=True)

    for idx, r in df.iterrows():
        r_t0 = time.time()
        q   = str(r[c_query] or "")
        tag = str(r[c_html] or "")
        cur = str(r[c_text] or "")
        old = _norm_score(r[c_score]) if c_score else _similarity(q, cur)

        print(f"\n[{now()}] → Row {idx + 1}/{total} | tag='{tag}' | old={old:.4f}", flush=True)
        cand, new = _try_improve(q, cur, tag, old, min_improve, max_attempts)

        improved_flag = new > old
        improved += 1 if improved_flag else 0
        kept += 1
        change_pct = ((new - old) / max(old, 1e-8) * 100.0) if improved_flag else 0.0

        rows.append({
            "HTML Bölümü": tag,
            "Kullanıcı Sorgusu": q,
            "Eski Metin": cur,
            "Geliştirilmiş Metin": cand,
            "Eski Skor": round(float(old), 6),
            "Yeni Skor": round(float(new), 6),
            "Yüzde Değişim": round(float(change_pct), 2),
        })
        msg = "✅ kept (Δ=+{:.2f}%)".format(change_pct) if improved_flag else "✅ kept (no improvement; Δ=0.00%)"
        print(f"   {msg}", flush=True)
        print(f"   ⏱ row time: {fmt_sec(time.time()-r_t0)}", flush=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(SORGU_OUT_CSV, index=False, encoding="utf-8")
    print(f"\n[{now()}] 💾 SORGU saved → {SORGU_OUT_CSV} (rows kept={kept}, improved={improved})", flush=True)
    print(f"[{now()}] 🏁 SORGU flow finished in {fmt_sec(time.time()-t_flow)}\n", flush=True)
    return SORGU_OUT_CSV

def main():
    t_all = time.time()
    print(f"[{now()}] ⚙️  START sorgu_iylestir.py", flush=True)
    print(f"    INPUT : {SORGU_IN_CSV}", flush=True)
    print(f"    OUTPUT: {SORGU_OUT_CSV}", flush=True)
    run_sorgu_flow()
    print(f"[{now()}] ✅ DONE", flush=True)

if __name__ == "__main__":
    main()
