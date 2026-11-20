"""
Builds the three documents in /Report from the files produced by the notebooks:
  * cleaned_profiling_report.html  – self-contained data-profiling report of the cleaned dataset
  * Brief_Report.pdf               – Task 7 brief report (results + conclusions)
  * Theory_Concepts.pdf            – definitions of every concept used in the project
Run AFTER the notebooks:   python build_reports.py
"""
import base64, io, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "..", "Data", "Processed")
RAW = os.path.join(HERE, "..", "Data", "Raw", "patient_health_records_raw.csv")
VIZ = os.path.join(HERE, "..", "Visualization")
OUT = os.path.join(HERE, "..", "Report")

S = json.load(open(os.path.join(PROC, "results_summary.json")))
raw = pd.read_csv(RAW)
clean = pd.read_csv(os.path.join(PROC, "cleaned_patient_health_records.csv"))
NUMCOLS = ["age", "bmi", "blood_pressure", "cholesterol", "glucose"]


# =====================================================================
# 1. HTML profiling report
# =====================================================================
def b64_file(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def b64_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def hist_img(col):
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    sns.histplot(clean[col], bins=25, color="#3b7dd8", ax=ax)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(labelsize=7)
    return b64_fig(fig)


def df_html(df, classes="tbl"):
    return df.to_html(classes=classes, border=0, escape=False)


def build_html():
    r_missing = int(raw.isna().sum().sum())
    ov = pd.DataFrame({
        "Metric": ["Number of rows", "Number of columns", "Missing cells", "Duplicate rows", "Duplicate patient_id",
                   "Numeric columns", "Categorical columns", "Target balance (High risk)"],
        "RAW": [raw.shape[0], raw.shape[1], r_missing, int(raw.duplicated().sum()), int(raw["patient_id"].duplicated().sum()),
                5, 2, f"{raw['disease_risk'].mean() * 100:.1f}%"],
        "CLEANED": [clean.shape[0], clean.shape[1], int(clean.isna().sum().sum()), int(clean.duplicated().sum()),
                    int(clean["patient_id"].duplicated().sum()), 5, 2, f"{clean['disease_risk'].mean() * 100:.1f}%"],
    }).set_index("Metric")

    rows = []
    for c in NUMCOLS:
        s_raw, s_cl = raw[c], clean[c]
        rows.append({"Variable": c,
                     "Missing (raw)": f"{s_raw.isna().sum()} ({s_raw.isna().mean() * 100:.1f}%)",
                     "Missing (clean)": int(s_cl.isna().sum()),
                     "Mean (raw → clean)": f"{s_raw.mean():.1f} → {s_cl.mean():.1f}",
                     "Std (raw → clean)": f"{s_raw.std():.1f} → {s_cl.std():.1f}",
                     "Min (raw → clean)": f"{s_raw.min():.1f} → {s_cl.min():.1f}",
                     "Max (raw → clean)": f"{s_raw.max():.1f} → {s_cl.max():.1f}",
                     "Skew (raw → clean)": f"{s_raw.skew():.2f} → {s_cl.skew():.2f}",
                     "Distribution (clean)": f'<img src="data:image/png;base64,{hist_img(c)}" width="180">'})
    numtbl = pd.DataFrame(rows).set_index("Variable")

    cat_rows = []
    for c in ["gender", "region"]:
        vc_r = raw[c].value_counts(dropna=False)
        vc_c = clean[c].value_counts()
        for k in vc_c.index:
            cat_rows.append({"Variable": c, "Category": k, "Raw count": int(vc_r.get(k, 0)), "Cleaned count": int(vc_c[k])})
        cat_rows.append({"Variable": c, "Category": "(missing)", "Raw count": int(raw[c].isna().sum()), "Cleaned count": 0})
    cattbl = pd.DataFrame(cat_rows).set_index(["Variable", "Category"])

    miss = pd.DataFrame(S["missing_summary"]).rename(columns={"index": "column"})
    miss = miss.set_index(miss.columns[0])[["missing_count", "missing_%"]]

    imp = pd.DataFrame(S["imputation_results"]).set_index("method")[["age_NRMSE", "bmi_NRMSE", "cholesterol_NRMSE", "glucose_NRMSE", "avg_NRMSE", "avg_cat_accuracy"]].round(3)
    imp2 = pd.DataFrame(S["imp_after_flag"]).set_index("method")[["avg_NRMSE", "avg_cat_accuracy"]].round(3)
    out = pd.DataFrame(S["outlier_eval"]).set_index("treatment")[["rows_kept", "data_loss_%", "NRMSE_vs_truth", "cells_outside_true_range", "cv_AUC"]].round(3)
    pipe = pd.DataFrame(S["pipeline_eval"]).set_index("pipeline").round(3)
    model = pd.DataFrame(S["model_table"]).set_index(["model", "metric"]).round(3)

    corr = clean[NUMCOLS + ["disease_risk"]].corr()
    fig, ax = plt.subplots(figsize=(6, 4.6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, square=True, cbar=False)
    corr_img = b64_fig(fig)

    def img(name, w="100%"):
        return f'<img src="data:image/png;base64,{b64_file(os.path.join(VIZ, name))}" style="max-width:{w}">'

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cleaned Patient Health Records – Profiling Report</title>
<style>
 body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6fa;color:#1d2433}}
 header{{background:linear-gradient(120deg,#1f4e9c,#2e9e6b);color:#fff;padding:28px 40px}}
 header h1{{margin:0 0 6px;font-size:26px}} header p{{margin:0;opacity:.9}}
 main{{max-width:1150px;margin:0 auto;padding:20px 28px 60px}}
 section{{background:#fff;border-radius:10px;padding:18px 24px;margin:18px 0;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
 h2{{margin-top:4px;color:#1f4e9c;border-bottom:2px solid #e3e8f2;padding-bottom:6px;font-size:20px}}
 table.tbl{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}
 table.tbl th{{background:#e9eff9;text-align:left;padding:6px 9px}} table.tbl td{{padding:6px 9px;border-top:1px solid #e6eaf2}}
 .kpis{{display:flex;gap:14px;flex-wrap:wrap}} .kpi{{flex:1;min-width:150px;background:#eef5ff;border-radius:8px;padding:12px 16px}}
 .kpi b{{display:block;font-size:24px;color:#1f4e9c}} .good b{{color:#2e9e6b}} .good{{background:#eaf7f0}}
 .scroll{{overflow-x:auto}} img{{border-radius:6px}} .note{{font-size:13px;color:#4a5568}}
</style></head><body>
<header><h1>Data Profiling Report – Cleaned Patient Health Records</h1>
<p>Practical-2 · Data Cleanser · generated from <code>Data/Processed/cleaned_patient_health_records.csv</code></p></header>
<main>
<section><h2>1. Overview</h2>
<div class="kpis">
 <div class="kpi"><b>{clean.shape[0]}</b>patients (rows)</div>
 <div class="kpi"><b>{clean.shape[1]}</b>columns</div>
 <div class="kpi"><b>{r_missing} → 0</b>missing cells</div>
 <div class="kpi good"><b>0</b>implausible outliers left</div>
 <div class="kpi good"><b>{S['final_pipeline_error'] if 'final_pipeline_error' in S else ''}</b></div>
</div>
{df_html(ov)}
<p class="note">Final pipeline: <b>{S['best_pipeline']}</b> – IQR-flagged outliers were turned into missing values and every missing value was then imputed with <b>{S['best_num2']}</b>.</p></section>

<section><h2>2. Missing-value summary (raw dataset)</h2>{df_html(miss)}{img('01_missing_values_map.png', '760px')}</section>

<section><h2>3. Numeric variables – raw vs cleaned</h2><div class="scroll">{df_html(numtbl)}</div></section>

<section><h2>4. Categorical variables</h2>{df_html(cattbl)}</section>

<section><h2>5. Imputation techniques – error vs true values</h2>
<p class="note">Normalised RMSE (lower is better) on the cells that were missing, with outliers still present:</p>{df_html(imp)}
<p class="note">After outlier flagging (final pipeline conditions):</p>{df_html(imp2)}{img('03_imputation_error_comparison.png')}</section>

<section><h2>6. Outlier treatments</h2>{df_html(out)}{img('10_outlier_method_evaluation.png')}{img('09_before_after_outlier_treatment.png')}</section>

<section><h2>7. Pipeline order experiment</h2>{df_html(pipe)}</section>

<section><h2>8. Raw vs cleaned distributions</h2>{img('11_raw_vs_cleaned_distributions.png')}</section>

<section><h2>9. Correlations of the cleaned data</h2>
<img src="data:image/png;base64,{corr_img}" width="520"> {img('13_correlation_heatmap_before_after.png')}</section>

<section><h2>10. Downstream model check (5-fold × 3 repeats)</h2>{df_html(model)}{img('14_model_performance_baseline_vs_cleaned.png', '640px')}</section>

<section><h2>11. Sample of the cleaned data</h2><div class="scroll">{df_html(clean.head(15).set_index('patient_id'))}</div></section>
</main></body></html>"""
    path = os.path.join(OUT, "cleaned_profiling_report.html")
    open(path, "w", encoding="utf-8").write(html.replace("<div class=\"kpi good\"><b></b></div>", ""))
    print("HTML ->", path)


# =====================================================================
# 2. PDFs
# =====================================================================
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

FD = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", FD + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", FD + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DV-BI", FD + "DejaVuSans-BoldOblique.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="DV", fontSize=9.2, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=BODY, fontName="DV-B", fontSize=19, leading=24, textColor=colors.HexColor("#1f4e9c"), alignment=0, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=BODY, fontName="DV-B", fontSize=12.5, leading=16, textColor=colors.HexColor("#1f4e9c"), alignment=0, spaceBefore=9, spaceAfter=4)
H3 = ParagraphStyle("h3", parent=BODY, fontName="DV-B", fontSize=10, leading=13, alignment=0, spaceBefore=5, spaceAfter=2)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=14, bulletIndent=3, spaceAfter=2)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10.2, alignment=0, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="DV-B", textColor=colors.white)
FORM = ParagraphStyle("form", parent=BODY, fontName="DV-I", alignment=1, backColor=colors.HexColor("#eef3fb"), borderPadding=4, spaceBefore=3, spaceAfter=7)


def P(t, st=BODY):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(i, BUL, bulletText="•") for i in items]


def table(data, widths, header=True):
    rows = []
    for ri, row in enumerate(data):
        rows.append([Paragraph(str(c), CELLB if (header and ri == 0) else CELL) for c in row])
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c4da")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e9c"))]
        st += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fc")])]
    t.setStyle(TableStyle(st))
    return t


def fig(name, width=16.5 * cm):
    from PIL import Image as PILImage
    path = os.path.join(VIZ, name)
    w, h = PILImage.open(path).size
    return Image(path, width=width, height=width * h / w)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(colors.HexColor("#667"))
    canvas.drawString(2 * cm, 1.1 * cm, "Practical-2 · Data Cleanser · Data Preprocessing & Feature Engineering")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def make_doc(name, story, title):
    doc = SimpleDocTemplate(os.path.join(OUT, name), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm, title=title, author="Practical-2")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("PDF ->", name)


def f3(x):
    return f"{x:.3f}"


def build_brief():
    st = []
    st += [P("Brief Report – Data Cleanser", H1),
           P("<b>Project:</b> Data Preprocessing &amp; Feature Engineering (Practical-2) &nbsp;|&nbsp; <b>Dataset:</b> 500 synthetic patient health records "
             "(age, gender, region, BMI, blood pressure, cholesterol, glucose) &nbsp;|&nbsp; <b>Target:</b> disease_risk (0 = low, 1 = high)"),
           P("1. What was done", H2)]
    st += bullets([
        f"The raw file contains <b>{S['total_missing_cells']} missing cells (4.22 %)</b> in {S['rows_with_missing']} of 500 rows (32.8 %), "
        "and about 12–13 extreme values in each of BMI, blood pressure, cholesterol and glucose.",
        "<b>Part A:</b> six imputation techniques (mean/median, most-frequent, missing-indicator + random sample, KNN, MICE) were applied and "
        "scored against the hidden true values (normalised RMSE for numbers, accuracy for categories).",
        "<b>Part B:</b> outliers were detected and treated with Z-score, IQR, percentile cap and winsorization, then compared on rows kept, "
        "error vs truth, and downstream ROC-AUC.",
        "<b>Part C:</b> the best order of operations was selected, the final cleaned dataset was saved, features were engineered and the result was validated with two classifiers."])

    st.append(P("2. Missing-value imputation", H2))
    imp = S["imputation_results"]
    after = {r["method"]: r for r in S["imp_after_flag"]}
    data = [["Technique", "Avg NRMSE – raw data (outliers present)", "Avg NRMSE – after outlier flagging", "Categorical accuracy (raw)"]]
    for r in imp:
        a = after.get(r["method"])
        data.append([r["method"], f3(r["avg_NRMSE"]), f3(a["avg_NRMSE"]) if a else "–", f"{r['avg_cat_accuracy']:.2f}"])
    st += [table(data, [5.6 * cm, 3.9 * cm, 3.9 * cm, 3.6 * cm]), Spacer(1, 4)]
    st += bullets([
        "<b>Most effective strategy:</b> with outliers still in the data, the <b>median</b> was the most robust (1.026) while KNN and random sampling were hurt by "
        "extreme values. Once outliers were neutralised, <b>MICE</b> was clearly best (0.979 vs 1.226 for mean/median and 1.126 for KNN) because it uses "
        "the relationships between age, BMI, blood pressure, cholesterol and glucose.",
        "For gender and region no technique beat the most-frequent category reliably (only 25 missing cells each)."])
    st.append(fig("03_imputation_error_comparison.png", 16 * cm))

    st.append(P("3. Outlier handling", H2))
    data = [["Treatment", "Rows kept", "Data loss", "Error vs truth", "Extreme cells left", "ROC-AUC"]]
    for r in S["outlier_eval"]:
        data.append([r["treatment"], r["rows_kept"], f"{r['data_loss_%']:.1f}%", f3(r["NRMSE_vs_truth"]), r["cells_outside_true_range"], f3(r["cv_AUC"])])
    st += [table(data, [5.2 * cm, 2 * cm, 2 * cm, 2.6 * cm, 3 * cm, 2.2 * cm]), Spacer(1, 4)]
    st += bullets([
        "<b>Method that preserved data quality best: Winsorization (5 %)</b> – all 500 rows kept, no value left outside the plausible range, ROC-AUC 0.875 "
        "(0.833 without treatment).",
        "Z-score and IQR <i>removal</i> gave cleaner values but discarded 7.6 % and 12.4 % of the patients. The 1 %/99 % percentile cap was ineffective because "
        "about 2.5 % of each column was corrupted, so the cap value was itself an outlier. The classic Z-score also under-detects because outliers inflate the standard deviation."])
    st.append(fig("10_outlier_method_evaluation.png", 16.5 * cm))

    st.append(PageBreak())
    st.append(P("4. Final pipeline and improvement in usability", H2))
    data = [["Pipeline", "Error vs truth", "Extreme cells", "ROC-AUC"]]
    for r in S["pipeline_eval"]:
        data.append([r["pipeline"], f3(r["NRMSE_vs_truth_all_cells"]), r["cells_outside_true_range"], f3(r["cv_AUC"])])
    st += [table(data, [8.2 * cm, 3 * cm, 3 * cm, 2.8 * cm]), Spacer(1, 4)]
    st.append(P("<b>Selected pipeline: B – flag outliers (IQR rule) as missing, then impute every gap with MICE.</b> It has the lowest error vs the true values "
                "(0.266, about 28 % lower than impute-then-cap or cap-then-impute), because extreme values are re-estimated from the patient's other measurements "
                "instead of being clamped to an arbitrary limit."))
    sc = S["scorecard"]
    data = [["Quality indicator", "Raw (as received)", "Final cleaned"]]
    for k in ["missing cells", "cells outside true range", "avg |skewness| (4 cols)", "max glucose", "max cholesterol", "max blood pressure", "BMI range", "NRMSE vs truth"]:
        data.append([k, sc["RAW (as received)"][k], sc["FINAL (cleaned)"][k]])
    st += [table(data, [6 * cm, 5 * cm, 5 * cm]), Spacer(1, 6)]
    mt = S["model_table"]
    data = [["Model", "Metric", "Baseline (mean impute, no outlier handling)", "Cleaned", "Change"]]
    for r in mt:
        data.append([r["model"], r["metric"], f3(r["Baseline"]), f3(r["Cleaned"]), f"{r['Improvement']:+.3f}"])
    st += [table(data, [3.6 * cm, 2.4 * cm, 5.6 * cm, 2.2 * cm, 2.2 * cm]), Spacer(1, 4)]
    st.append(fig("14_model_performance_baseline_vs_cleaned.png", 11 * cm))

    st.append(P("5. Conclusions", H2))
    st += bullets([
        "<b>Imputation:</b> use robust simple methods (median / mode) when extreme values are present, and multivariate MICE once outliers have been neutralised.",
        "<b>Outliers:</b> cap (winsorize) instead of deleting when the data set is small; better still, treat implausible values as missing and re-estimate them.",
        "<b>Usability:</b> 0 missing cells, 0 implausible values, skewness 4.27 → 0.04, and the correlations with the target are restored; Logistic Regression ROC-AUC rose "
        "from 0.831 to 0.874 and accuracy from 74.1 % to 79.3 %. Random Forest, which is naturally robust, gained less (AUC 0.854 → 0.862).",
        "<b>Feature engineering:</b> age groups, WHO BMI classes, blood-pressure stages, cholesterol and glucose bands and a metabolic-flag count were added; "
        "<i>ml_ready_features.csv</i> (500 × 18) is encoded and scaled.",
        "<b>Limitations:</b> synthetic data (which made exact scoring possible), moderate sample size (500), and tunable thresholds (|z| &gt; 3, 1.5 × IQR, 5 % winsor limit) that "
        "should be reviewed with domain experts on real data."])
    make_doc("Brief_Report.pdf", st, "Brief Report – Data Cleanser")


def build_theory():
    st = [P("Theory Concepts – Data Cleaning &amp; Feature Engineering", H1),
          P("Definitions and short explanations of every technique used in the Data Cleanser project, written in the context of patient health records.")]

    st.append(P("1. Data preprocessing", H2))
    st.append(P("<b>Data preprocessing</b> is the set of steps that turns raw, inconsistent data into a form that a machine-learning algorithm can use correctly. "
                "Typical steps are handling missing values, treating outliers, encoding categorical variables, scaling numeric variables and creating new informative features. "
                "<b>Feature engineering</b> is the creation or transformation of input variables so that the model can learn the underlying pattern more easily. "
                "The principle <i>“garbage in, garbage out”</i> applies: a model can only be as good as the data it is trained on."))

    st.append(P("2. Missing data", H2))
    st.append(P("A value is <b>missing</b> when it was not recorded (device failure, skipped question, entry error). The mechanism matters for choosing a remedy:"))
    st.append(table([
        ["Mechanism", "Meaning", "Example"],
        ["MCAR – Missing Completely At Random", "Probability of missingness is unrelated to any variable", "A lab sample is lost at random"],
        ["MAR – Missing At Random", "Missingness depends on other <i>observed</i> variables", "Older patients skip the BMI measurement more often"],
        ["MNAR – Missing Not At Random", "Missingness depends on the missing value itself", "Patients with very high glucose avoid the test"]],
        [4.6 * cm, 6.4 * cm, 6 * cm]))
    st.append(Spacer(1, 4))
    st.append(P("<b>Deletion</b> (listwise) removes every row with a gap – it is simple but wastes data and can bias results unless data are MCAR. "
                "<b>Imputation</b> replaces a gap with an estimated value. Every method below trades simplicity against accuracy and the preservation of variance and relationships."))

    st.append(P("3. Imputation techniques", H2))
    st.append(P("3.1 Simple imputer (mean / median) – numerical", H3))
    st.append(P("Each gap in a numeric column is replaced by one constant: the column mean or the column median."))
    st.append(P("x<sub>imputed</sub> = mean(x<sub>observed</sub>) &nbsp; or &nbsp; median(x<sub>observed</sub>)", FORM))
    st.append(P("<b>Pros:</b> fast, no model needed. <b>Cons:</b> shrinks the variance, distorts correlations and creates a spike at the centre. The median is preferred when the "
                "column is skewed or contains outliers because it is not pulled by extreme values."))
    st.append(P("3.2 Simple imputer / most-frequent (mode) – categorical", H3))
    st.append(P("Gaps in a categorical column (region, gender) are filled with the category that occurs most often. It keeps the variable valid but over-represents the majority class."))
    st.append(P("3.3 Missing indicator + random sample imputation", H3))
    st.append(P("A binary <b>missing-indicator</b> column (1 = value was missing) is added so the model can learn from the fact that data were absent. "
                "<b>Random sample imputation</b> then fills each gap with a value drawn at random from the observed values of the same column, which preserves the "
                "original distribution (mean and spread). It adds randomness and ignores relationships between variables."))
    st.append(P("3.4 KNN imputer", H3))
    st.append(P("For a row with a gap, the algorithm finds the <i>k</i> most similar rows (distance computed on the features that are available, after standardisation) and "
                "replaces the gap with the (distance-weighted) average of those neighbours."))
    st.append(P("d(a, b) = sqrt( Σ<sub>j</sub> (a<sub>j</sub> − b<sub>j</sub>)<super>2</super> ) , &nbsp; x̂ = Σ w<sub>i</sub> x<sub>i</sub> / Σ w<sub>i</sub> , &nbsp; w<sub>i</sub> = 1 / d<sub>i</sub>", FORM))
    st.append(P("<b>Pros:</b> uses multivariate information, can capture non-linear local patterns. <b>Cons:</b> sensitive to scaling and outliers, slow on large data, needs a choice of <i>k</i>."))
    st.append(P("3.5 MICE – Multivariate Imputation by Chained Equations", H3))
    st.append(P("MICE (implemented as <i>IterativeImputer</i>) first fills all gaps with a simple guess, then repeatedly models each incomplete variable as a function of all the others "
                "(here Bayesian Ridge regression) and updates the imputed values; the cycle is repeated until the estimates stabilise. The method assumes the data are MAR and "
                "preserves relationships between variables. Repeating the process with random draws produces <i>multiple</i> imputed datasets that can express the uncertainty of the imputation."))

    st.append(P("4. Outliers", H2))
    st.append(P("An <b>outlier</b> is an observation that is far away from the bulk of the data. It may be a <i>measurement or entry error</i> (BMI of 8, glucose of 500 caused by a "
                "faulty device), which should be corrected, or a <i>genuine rare case</i>, which should be kept. Outliers inflate the mean and standard deviation, distort "
                "correlations and strongly affect distance-based and linear models, while tree-based models are comparatively robust."))
    st.append(P("4.1 Z-score method", H3))
    st.append(P("z = (x − μ) / σ &nbsp;&nbsp;→&nbsp; a value is an outlier if |z| &gt; 3 (more than three standard deviations from the mean)", FORM))
    st.append(P("Works well for roughly normal data. <b>Weakness:</b> μ and σ are themselves influenced by the outliers (the <i>masking effect</i>), so extreme points can hide."))
    st.append(P("4.2 Modified Z-score (robust, bonus)", H3))
    st.append(P("M = 0.6745 · (x − median) / MAD , &nbsp; MAD = median(|x − median|) &nbsp;→&nbsp; outlier if |M| &gt; 3.5. It uses the median and MAD, so it is not affected by the outliers it tries to find.", FORM))
    st.append(P("4.3 IQR method (Tukey fences)", H3))
    st.append(P("IQR = Q3 − Q1 &nbsp;→&nbsp; lower fence = Q1 − 1.5 · IQR , &nbsp; upper fence = Q3 + 1.5 · IQR", FORM))
    st.append(P("Values outside the fences are flagged. It makes no normality assumption and is robust because quartiles are not moved by a few extreme points; it can over-flag in skewed data."))
    st.append(P("4.4 Percentile method", H3))
    st.append(P("Values below the 1st percentile or above the 99th percentile are replaced (capped) by those percentile values. It always touches a fixed share of the data "
                "(2 % here); if more than 1 % per tail is corrupted, the cap value itself is still an outlier."))
    st.append(P("4.5 Winsorization", H3))
    st.append(P("Winsorization replaces the lowest p % of values by the p-th percentile value and the highest p % by the (100 − p)-th percentile value (here p = 5). "
                "Unlike trimming/removal, no observation is lost, so sample size and row alignment are preserved; extreme values keep their direction but lose their leverage."))
    st.append(P("4.6 Removal vs capping vs treating as missing", H3))
    st.append(table([
        ["Option", "Advantage", "Disadvantage"],
        ["Remove rows", "Cleanest remaining values", "Loses data (12.4 % with IQR here); may remove genuine cases"],
        ["Cap / winsorize", "Keeps all rows, limits leverage", "Capped values are not real measurements"],
        ["Flag as missing, then impute", "Re-estimates a plausible value from other variables", "Depends on the quality of the imputation model"]],
        [4.2 * cm, 6 * cm, 6.8 * cm]))

    st.append(P("5. Feature engineering used", H2))
    st += bullets([
        "<b>Binning (discretisation):</b> continuous values grouped into clinically meaningful classes – age groups, WHO BMI classes (underweight &lt; 18.5, normal 18.5–24.9, overweight 25–29.9, obese ≥ 30), "
        "blood-pressure stages (normal &lt; 120, elevated 120–129, stage-1 130–139, stage-2 ≥ 140 mmHg), cholesterol (&lt; 200 desirable, 200–239 borderline, ≥ 240 high) and fasting glucose (&lt; 100 normal, 100–125 prediabetic, ≥ 126 diabetic range).",
        "<b>Aggregate feature:</b> <i>metabolic_flags</i> = number of the four risk factors (BMI ≥ 25, BP ≥ 130, cholesterol ≥ 200, glucose ≥ 100) that are present (0–4).",
        "<b>Encoding:</b> binary encoding for gender, one-hot encoding for the nominal variable region, ordinal encoding for ordered categories.",
        "<b>Scaling – standardisation:</b> z = (x − μ) / σ, so that all numeric inputs have mean 0 and standard deviation 1 (important for distance-based and regularised models)."])

    st.append(P("6. Evaluation metrics", H2))
    st += bullets([
        "<b>RMSE</b> = sqrt( mean( (x̂ − x)<super>2</super> ) ): typical size of the imputation error in the original unit.",
        "<b>Normalised RMSE (NRMSE)</b> = RMSE / σ<sub>true</sub>: makes columns with different units comparable; a value near 1 means “no better than guessing the average”.",
        "<b>Accuracy</b> (categorical imputation) = share of imputed categories equal to the true category.",
        "<b>ROC-AUC:</b> probability that a randomly chosen high-risk patient receives a higher score than a randomly chosen low-risk patient (0.5 = random, 1 = perfect).",
        "<b>Cross-validation:</b> the data are split repeatedly into training and test folds (here 5-fold, repeated) so that the performance estimate does not depend on one lucky split.",
        "<b>Skewness:</b> asymmetry of a distribution (0 for a symmetric one); large values indicate long tails, usually caused by outliers."])

    st.append(P("7. Recommended workflow", H2))
    st += bullets([
        "Profile the data: missingness map, distributions, box plots.",
        "Decide whether the extreme values are errors or genuine cases (use domain knowledge).",
        "Mark implausible values as missing, then impute with a multivariate method such as MICE; use median / mode for simple columns.",
        "Cap any remaining extreme values (winsorization) and verify with summary statistics before vs after.",
        "Engineer, encode and scale features; validate with a downstream model using cross-validation.",
        "Document every decision so that the pipeline is reproducible."])
    make_doc("Theory_Concepts.pdf", st, "Theory Concepts – Data Cleaning & Feature Engineering")


if __name__ == "__main__":
    build_html()
    build_brief()
    build_theory()
