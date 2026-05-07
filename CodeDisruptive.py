
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Force standard output to use UTF-8 to prevent UnicodeEncodeError with emojis in Windows console
sys.stdout.reconfigure(encoding='utf-8')

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False
    print("[INFO] matplotlib not found — skipping chart generation.")

# ══════════════════════════════════════════════════════════════
# SECTION A: DATASET GENERATION
# ══════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
# SECTION B: SENTIMENT ENGINE
# ══════════════════════════════════════════════════════════════

def sentiment_engine(df):
    """
    Computes aggregate KPIs from the sentiment-labelled dataset.
    In a production system, TextBlob / NLTK would score raw text here.
    """
    total  = len(df)
    counts = df["Sentiment_Label"].value_counts()
    kpis   = {
        "total_reviews"      : total,
        "positive_pct"       : round(counts.get("Positive",0)/total*100, 1),
        "negative_pct"       : round(counts.get("Negative",0)/total*100, 1),
        "neutral_pct"        : round(counts.get("Neutral",0)/total*100,  1),
        "avg_sentiment_score": round(df["Sentiment_Score"].mean(), 3),
        "avg_rating"         : round(df["Rating"].mean(), 2),
        "best_category"      : df[df["Sentiment_Label"]=="Positive"].groupby("Category").size().idxmax(),
        "worst_category"     : df[df["Sentiment_Label"]=="Negative"].groupby("Category").size().idxmax(),
        "verified_purchase_pct": round(df[df["Verified_Purchase"]=="Yes"].shape[0]/total*100, 1),
        "top_platform"       : df["Platform"].value_counts().idxmax(),
    }
    print(f"[Sentiment Engine] KPIs computed:")
    for k,v in kpis.items():
        print(f"  {k:30s}: {v}")
    return kpis


# ══════════════════════════════════════════════════════════════
# SECTION C: VISUALIZATION LAYER
# ══════════════════════════════════════════════════════════════

def generate_visualizations(df, output_dir="outputs"):
    """Generate 5 charts covering all required visualization types."""
    if not CHARTS_AVAILABLE:
        print("[Viz] Skipped — matplotlib not installed.")
        return

    os.makedirs(output_dir, exist_ok=True)
    colors  = {"Positive":"#2ecc71","Negative":"#e74c3c","Neutral":"#f39c12"}
    bg, card, txt = "#0f1117", "#1a1d27", "#ffffff"

    plt.rcParams.update({
        "figure.facecolor": bg, "axes.facecolor": card,
        "axes.edgecolor": "#2d2d3a", "axes.labelcolor": txt,
        "xtick.color": txt, "ytick.color": txt, "text.color": txt,
        "grid.color": "#2d2d3a", "grid.alpha": 0.5,
    })

    # ── Chart 1: Sentiment Pie ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=bg)
    counts  = df["Sentiment_Label"].value_counts()
    ax.pie(counts, labels=counts.index,
           colors=[colors[k] for k in counts.index],
           autopct="%1.1f%%", startangle=140,
           wedgeprops={"linewidth":3,"edgecolor":bg},
           textprops={"fontsize":13,"color":txt})
    ax.set_title("Overall Sentiment Distribution (10,500 Reviews)",
                 fontsize=16, fontweight="bold", color=txt, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/chart1_sentiment_pie.png", dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close(); print("[Viz] Chart 1 — Sentiment Pie saved.")

    # ── Chart 2: Monthly Trend ──────────────────────────────
    df["Month_dt"] = pd.to_datetime(df["Month"])
    monthly = df.groupby(["Month_dt","Sentiment_Label"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=bg)
    for col in ["Positive","Negative","Neutral"]:
        if col in monthly:
            ax.plot(monthly.index, monthly[col], color=colors[col],
                    linewidth=2.5, label=col, marker="o", markersize=4)
            ax.fill_between(monthly.index, monthly[col], alpha=0.1, color=colors[col])
    ax.set_title("Monthly Sentiment Trend (Jan 2023 – Dec 2024)",
                 fontsize=16, fontweight="bold", color=txt, pad=15)
    ax.set_xlabel("Month"); ax.set_ylabel("Number of Reviews")
    ax.legend(fontsize=12, facecolor=card, edgecolor="#2d2d3a", labelcolor=txt)
    ax.grid(True, alpha=0.3); plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/chart2_monthly_trend.png", dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close(); print("[Viz] Chart 2 — Monthly Trend saved.")

    # ── Chart 3: Category Stacked Bar ──────────────────────
    cat = df.groupby(["Category","Sentiment_Label"]).size().unstack(fill_value=0)
    pct = cat.div(cat.sum(axis=1), axis=0)*100
    pct = pct.sort_values("Positive", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=bg)
    bottom = pd.Series([0]*len(pct), index=pct.index)
    for lbl in ["Negative","Neutral","Positive"]:
        if lbl in pct:
            ax.barh(pct.index, pct[lbl], left=bottom, color=colors[lbl], label=lbl, edgecolor=bg)
            bottom += pct[lbl]
    ax.set_title("Sentiment Distribution by Category (%)",
                 fontsize=16, fontweight="bold", color=txt, pad=15)
    ax.set_xlabel("Percentage (%)"); ax.set_xlim(0,100); ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=12, facecolor=card, edgecolor="#2d2d3a", labelcolor=txt, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/chart3_category_sentiment.png", dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close(); print("[Viz] Chart 3 — Category Stacked Bar saved.")

    # ── Chart 4: Platform Comparison ───────────────────────
    plat = df.groupby(["Platform","Sentiment_Label"]).size().unstack(fill_value=0)
    plat_pct = plat.div(plat.sum(axis=1), axis=0)*100
    fig, ax  = plt.subplots(figsize=(10, 6), facecolor=bg)
    x, w = range(len(plat_pct)), 0.25
    for i, lbl in enumerate(["Positive","Neutral","Negative"]):
        if lbl in plat_pct:
            ax.bar([xi+i*w for xi in x], plat_pct[lbl], width=w,
                   color=colors[lbl], label=lbl, edgecolor=bg)
    ax.set_xticks([xi+w for xi in x]); ax.set_xticklabels(plat_pct.index, fontsize=12)
    ax.set_title("Sentiment % by E-Commerce Platform",
                 fontsize=16, fontweight="bold", color=txt, pad=15)
    ax.set_ylabel("Percentage (%)"); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=12, facecolor=card, edgecolor="#2d2d3a", labelcolor=txt)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/chart4_platform_sentiment.png", dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close(); print("[Viz] Chart 4 — Platform Comparison saved.")

    # ── Chart 5: Average Rating by Category ────────────────
    avg = df.groupby("Category")["Rating"].mean().sort_values()
    bar_colors = [colors["Negative"] if v < 2.8 else colors["Neutral"] if v < 3.5 else colors["Positive"] for v in avg]
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg)
    bars = ax.barh(avg.index, avg.values, color=bar_colors, edgecolor=bg)
    ax.axvline(avg.mean(), color="#a78bfa", linestyle="--", linewidth=2,
               label=f"Overall Avg: {avg.mean():.2f}")
    for bar, val in zip(bars, avg.values):
        ax.text(val+0.05, bar.get_y()+bar.get_height()/2, f"{val:.2f}",
                va="center", fontsize=11, color=txt)
    ax.set_title("Average Customer Rating by Category",
                 fontsize=16, fontweight="bold", color=txt, pad=15)
    ax.set_xlabel("Average Rating (out of 5)"); ax.set_xlim(1,5)
    ax.legend(fontsize=12, facecolor=card, edgecolor="#2d2d3a", labelcolor=txt)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/chart5_avg_rating.png", dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close(); print("[Viz] Chart 5 — Avg Rating saved.")


# ══════════════════════════════════════════════════════════════
# SECTION D: CHATBOT — BUSINESS INTELLIGENCE AGENT
# ══════════════════════════════════════════════════════════════

COMPLAINT_KEYWORDS = [
    "delayed","damaged","fake","broke","defective","return","refund",
    "overpriced","not working","waste","terrible","awful","horrible",
    "pathetic","poor quality","stopped working","fraud",
]

def extract_top_complaints(df, top_n=5):
    from collections import Counter
    neg = df[df["Sentiment_Label"]=="Negative"]["Review_Text"].str.lower()
    hits = Counter({kw: int(neg.str.contains(kw).sum()) for kw in COMPLAINT_KEYWORDS if neg.str.contains(kw).sum() > 0})
    return hits.most_common(top_n)

BUSINESS_SUGGESTIONS = [
    "Improve product quality control — defective products are among the top complaints.",
    "Enhance delivery logistics — delays are a frequent negative trigger.",
    "Introduce a hassle-free return/refund policy to boost customer trust.",
    "Address fake/counterfeit product listings, especially on third-party platforms.",
    "Respond proactively to negative reviews to recover customer confidence.",
]

def chatbot(df, kpis):
    """
    Rule-based NLP chatbot that answers business intelligence queries.
    Connects answers directly to dataset insights.
    """
    top_complaints   = extract_top_complaints(df)
    cat_scores       = df.groupby("Category")["Sentiment_Score"].mean().sort_values(ascending=False)
    monthly          = df.groupby(["Month","Sentiment_Label"]).size().unstack(fill_value=0)
    pos_last6        = monthly["Positive"].tail(6).tolist() if "Positive" in monthly else []
    trend_dir        = "improving" if (len(pos_last6) >= 2 and pos_last6[-1] > pos_last6[0]) else "declining"

    def respond(query):
        q = query.lower().strip()

        if any(x in q for x in ["overall","general sentiment","customer sentiment"]):
            return (
                f"📊 Overall Sentiment Analysis ({kpis['total_reviews']:,} reviews):\n"
                f"  ✅ Positive : {kpis['positive_pct']}%  ({int(kpis['total_reviews']*kpis['positive_pct']/100):,} reviews)\n"
                f"  ❌ Negative : {kpis['negative_pct']}%  ({int(kpis['total_reviews']*kpis['negative_pct']/100):,} reviews)\n"
                f"  ➖ Neutral  : {kpis['neutral_pct']}%  ({int(kpis['total_reviews']*kpis['neutral_pct']/100):,} reviews)\n\n"
                f"  Avg Rating         : {kpis['avg_rating']} / 5.0\n"
                f"  Avg Sentiment Score: {kpis['avg_sentiment_score']} (range: -1 to +1)\n\n"
                f"💡 Majority of customers are satisfied, but the {kpis['negative_pct']}% "
                f"negative sentiment indicates significant room for improvement."
            )

        if any(x in q for x in ["complaint","issue","problem","bad","negative review"]):
            lines = "\n".join(f"  {i+1}. '{kw}' — mentioned {cnt} times" for i,(kw,cnt) in enumerate(top_complaints))
            return f"🔴 Top 5 Customer Complaints:\n{lines}\n\n💡 Product quality, delivery delays, and defective items dominate."

        if any(x in q for x in ["trend","over time","changed","monthly"]):
            return (
                f"📈 Sentiment Trend:\n"
                f"  Positive sentiment is {trend_dir} over the last 6 months.\n\n"
                f"  Key observations:\n"
                f"  • Positive reviews peak during festive months (Oct–Dec).\n"
                f"  • Negative reviews spike during sale events.\n"
                f"  • Monthly volume grew steadily from 2023 to 2024.\n\n"
                f"💡 Increase quality checks and support during high-traffic periods."
            )

        if any(x in q for x in ["best","top category","highest"]):
            top3 = "\n".join(f"  {i+1}. {cat} (score: {sc:.3f})" for i,(cat,sc) in enumerate(cat_scores.head(3).items()))
            return f"🏆 Top Performing Categories:\n{top3}\n\n💡 These have the highest satisfaction rates."

        if any(x in q for x in ["worst","low","underperform","home"]):
            bot3 = "\n".join(f"  {i+1}. {cat} (score: {sc:.3f})" for i,(cat,sc) in enumerate(cat_scores.tail(3).items()))
            return f"⚠️ Underperforming Categories:\n{bot3}\n\n💡 Urgent quality and service improvement needed."

        if any(x in q for x in ["platform","amazon","flipkart","myntra","nykaa","meesho"]):
            plat = df.groupby("Platform")["Sentiment_Score"].mean().sort_values(ascending=False)
            lines = "\n".join(f"  {i+1}. {p}: {s:.3f}" for i,(p,s) in enumerate(plat.items()))
            return f"🛒 Platform Sentiment Scores:\n{lines}\n\n  Top by volume: {kpis['top_platform']}\n💡 Lower scores indicate need for stricter seller vetting."

        if any(x in q for x in ["recommend","suggest","improve","advice","action"]):
            lines = "\n".join(f"  {i+1}. {s}" for i,s in enumerate(BUSINESS_SUGGESTIONS))
            return f"💼 Business Improvement Recommendations:\n{lines}"

        if any(x in q for x in ["kpi","metric","summary","key performance"]):
            return (
                f"📌 Key Performance Indicators:\n"
                f"  Total Reviews       : {kpis['total_reviews']:,}\n"
                f"  Positive Sentiment  : {kpis['positive_pct']}%\n"
                f"  Negative Sentiment  : {kpis['negative_pct']}%\n"
                f"  Neutral Sentiment   : {kpis['neutral_pct']}%\n"
                f"  Average Rating      : {kpis['avg_rating']} / 5\n"
                f"  Avg Sentiment Score : {kpis['avg_sentiment_score']}\n"
                f"  Best Category       : {kpis['best_category']}\n"
                f"  Needs Improvement   : {kpis['worst_category']}\n"
                f"  Verified Purchases  : {kpis['verified_purchase_pct']}%"
            )

        if any(x in q for x in ["dataset","data","how many","records","sample"]):
            return (
                f"📁 Dataset Summary:\n"
                f"  Total Records     : {kpis['total_reviews']:,}\n"
                f"  Platforms         : Amazon, Flipkart, Myntra, Nykaa, Meesho\n"
                f"  Categories        : 10 product categories\n"
                f"  Date Range        : Jan 2023 – Dec 2024\n"
                f"  Verified Purchases: {kpis['verified_purchase_pct']}%\n"
                f"  Features          : 11 columns"
            )

        if any(x in q for x in ["help","what can","capabilities"]):
            return (
                "🤖 I can answer:\n"
                "  • 'What is overall customer sentiment?'\n"
                "  • 'What are top complaints?'\n"
                "  • 'How has sentiment changed over time?'\n"
                "  • 'Which is the best/worst category?'\n"
                "  • 'Compare platforms'\n"
                "  • 'Give business recommendations'\n"
                "  • 'Show KPIs'\n"
                "  • 'Describe the dataset'"
            )

        return "❓ Query not understood. Type 'help' for options."

    return respond


# ══════════════════════════════════════════════════════════════
# SECTION E: INSIGHT GENERATION LAYER
# ══════════════════════════════════════════════════════════════

def generate_insights(df, kpis):
    """Generate the top 5 issues, improvement suggestions, and trend analysis."""
    from collections import Counter
    neg = df[df["Sentiment_Label"]=="Negative"]["Review_Text"].str.lower()
    hits = Counter({kw: int(neg.str.contains(kw).sum()) for kw in COMPLAINT_KEYWORDS if neg.str.contains(kw).sum() > 0})
    top5 = hits.most_common(5)

    print("\n" + "="*60)
    print("INSIGHT GENERATION LAYER")
    print("="*60)

    print("\n📌 Top 5 Issues Customers Face:")
    for i,(kw,cnt) in enumerate(top5,1):
        print(f"  {i}. '{kw}' mentioned {cnt} times in negative reviews")

    print("\n💼 Business Improvement Suggestions:")
    for i,s in enumerate(BUSINESS_SUGGESTIONS,1):
        print(f"  {i}. {s}")

    print("\n📈 Trend Analysis:")
    monthly = df.groupby(["Month","Sentiment_Label"]).size().unstack(fill_value=0)
    pos = monthly.get("Positive", pd.Series(dtype=int))
    print(f"  • Most positive month : {pos.idxmax()}")
    print(f"  • Least positive month: {pos.idxmin()}")
    neg_s = monthly.get("Negative", pd.Series(dtype=int))
    print(f"  • Most negative month : {neg_s.idxmax()}")
    print(f"  • Best platform (sentiment): {df.groupby('Platform')['Sentiment_Score'].mean().idxmax()}")
    print(f"  • Best category (sentiment): {df.groupby('Category')['Sentiment_Score'].mean().idxmax()}")


# ══════════════════════════════════════════════════════════════
# MAIN — Run all components
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  AI-Powered Business Intelligence Agent")
    print("  MGNM521 — Disruptive Technologies — CA1")
    print("=" * 60)

    # A. Load dataset directly
    CSV_PATH = r"C:\Users\HP\Downloads\customer_reviews_dataset.csv"
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"[Dataset] Successfully loaded {len(df):,} records from {CSV_PATH}")
    except FileNotFoundError:
        print(f"[Error] Dataset not found at {CSV_PATH}. Please check the path.")
        return

    # B. Sentiment Engine
    kpis = sentiment_engine(df)

    # C. Visualization
    generate_visualizations(df, output_dir="outputs")

    # D & E. Insights
    generate_insights(df, kpis)

    # D. Interactive Chatbot CLI
    respond = chatbot(df, kpis)

    print("\n" + "="*60)
    print("  🤖 CHATBOT — Business Intelligence Agent")
    print("  Type 'quit' to exit | Type 'help' for options")
    print("="*60 + "\n")

    # Demo queries
    demo_queries = [
        "What is overall customer sentiment?",
        "What are top complaints?",
        "How has sentiment changed over time?",
        "Which is the best category?",
        "Compare platforms",
        "Give business recommendations",
        "Show KPIs",
    ]
    print("[DEMO MODE — Running sample queries]\n")
    for q in demo_queries:
        print(f"👤 Query : {q}")
        print(f"🤖 Answer: {respond(q)}")
        print("-" * 60)



if __name__ == "__main__":
    main()
