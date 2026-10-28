import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="MailGuard: Spam Filter", page_icon="🛡️", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app.py lives in the repo root
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
VEC_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "emails.csv")

SPAM_COLOR, HAM_COLOR = "#ef5350", "#66bb6a"

EXAMPLES = {
    "Spam: prize scam": "Congratulations! You have been selected to win a $1000 gift card. "
                        "Click here now to claim your free prize. Limited time offer, act now!",
    "Spam: pharmacy": "Buy cheap viagra online. No prescription needed. Huge discount, "
                      "order today and get free shipping worldwide!!!",
    "Legit: meeting": "Hi team, attaching the notes from yesterday's project meeting. "
                      "Please review the action items and let me know if anything is missing.",
    "Legit: colleague": "Hey, are we still on for lunch tomorrow? I can also send over the "
                        "report draft before then if you want to take a look.",
}


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading model (first run may train it, please wait)...")
def load_artifacts():
    """Load saved model files. If they are missing, train once, then load."""
    if not all(os.path.exists(p) for p in (MODEL_PATH, VEC_PATH, METRICS_PATH)):
        os.chdir(BASE_DIR)  # train_pipeline.py uses relative paths
        from download_data import download_and_prepare_dataset
        from train_pipeline import run_training_pipeline

        if not os.path.exists(DATA_PATH):
            download_and_prepare_dataset()
        run_training_pipeline()

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    return model, vectorizer, metrics


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset():
    """Dataset is optional: only used by the Data & Evaluation tab."""
    try:
        if not os.path.exists(DATA_PATH):
            os.chdir(BASE_DIR)
            from download_data import download_and_prepare_dataset

            download_and_prepare_dataset()
        df = pd.read_csv(DATA_PATH)
        if df["label"].dtype == object:
            df["label"] = df["label"].map({"spam": 1, "not_spam": 0})
        df["text"] = df["text"].fillna("")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        return df
    except Exception:
        return None


def raw_scores(model, X):
    """Higher score = more spam-like."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def to_spam_prob(model, X):
    """0..1 spam score. For SVMs this is a squashed margin, not a calibrated probability."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return 1 / (1 + np.exp(-model.decision_function(X)))


@st.cache_data(show_spinner="Scoring the held-out test set...")
def test_scores(_model, _vectorizer, texts, labels):
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X = _vectorizer.transform(X_te)
    return raw_scores(_model, X), np.asarray(y_te)


try:
    model, vectorizer, metrics = load_artifacts()
except Exception as e:
    st.error(f"Could not load or train the model: {e}")
    st.stop()

df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
if "Train_Time_sec" not in df_metrics:
    df_metrics["Train_Time_sec"] = 0.0
best_row = df_metrics.sort_values(["F1-Score", "Precision"], ascending=False).iloc[0]
has_coef = hasattr(model, "coef_")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🛡️ MailGuard")
    st.caption("Intelligent email spam classification with NLP + ML")
    st.metric("Active model", best_row["Model"])
    st.metric("Vocabulary size", f"{len(vectorizer.get_feature_names_out()):,}")
    st.divider()
    st.caption("Built by Arman Rizvi · 3rd Year B.Tech, Data Science & AI")

st.title("🛡️ MailGuard: Intelligent Email Spam Classification")
tab_home, tab_live, tab_batch, tab_compare, tab_insights, tab_eval = st.tabs(
    ["🏠 Home", "🔍 Live Filter", "📂 Batch Scan", "📊 Model Comparison",
     "🧠 Spam Insights", "🧪 Data & Evaluation"]
)

# ================================================================ HOME
with tab_home:
    st.subheader("Protecting inboxes with NLP and Machine Learning")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected model", best_row["Model"])
    c2.metric("Precision", f"{best_row['Precision'] * 100:.2f}%")
    c3.metric("F1-Score", f"{best_row['F1-Score'] * 100:.2f}%")
    c4.metric("Accuracy", f"{best_row['Accuracy'] * 100:.2f}%")

    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
    radar = go.Figure()
    for _, r in df_metrics.iterrows():
        vals = [r[m] for m in metric_cols]
        radar.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=metric_cols + metric_cols[:1], name=r["Model"], fill="toself", opacity=0.5
        ))
    low = max(0.0, float(df_metrics[metric_cols].min().min()) - 0.01)
    radar.update_layout(
        title="All models at a glance (click legend items to hide or show)",
        polar=dict(radialaxis=dict(range=[low, 1.0])), height=480,
    )
    st.plotly_chart(radar, width="stretch")
    st.info("Open the **Live Filter** tab to scan an email, or **Batch Scan** to upload a CSV.")

# ================================================================ LIVE FILTER
with tab_live:
    st.subheader("Live Spam Scanner")
    st.session_state.setdefault("email_text", "")

    def load_example():
        choice = st.session_state.get("example_choice")
        if choice in EXAMPLES:
            st.session_state["email_text"] = EXAMPLES[choice]

    left, right = st.columns([3, 2])
    with left:
        st.selectbox("Try an example", ["(none)"] + list(EXAMPLES), key="example_choice",
                     on_change=load_example)
        text = st.text_area("Paste the email content", key="email_text", height=260,
                            placeholder="Dear user, you have won a $1000 gift card...")
        threshold = st.slider("Spam threshold (lower = stricter filter)", 0.05, 0.95, 0.50, 0.05)
        go_btn = st.button("Analyze Email", type="primary", width="stretch")

    with right:
        if go_btn:
            if not text or len(text.strip()) < 5:
                st.warning("Please enter a valid email body.")
            else:
                vec = vectorizer.transform([text])
                p = float(to_spam_prob(model, vec)[0])
                is_spam = p >= threshold

                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=p * 100,
                    number={"suffix": "%"}, title={"text": "Spam score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": SPAM_COLOR if is_spam else HAM_COLOR},
                        "steps": [{"range": [0, threshold * 100], "color": "rgba(102,187,106,0.25)"},
                                  {"range": [threshold * 100, 100], "color": "rgba(239,83,80,0.25)"}],
                        "threshold": {"line": {"color": "white", "width": 3}, "value": threshold * 100},
                    },
                ))
                gauge.update_layout(height=300, margin=dict(t=60, b=10, l=20, r=20))
                st.plotly_chart(gauge, width="stretch")

                if is_spam:
                    st.error("🚨 SPAM DETECTED")
                else:
                    st.success("✅ SAFE / LEGITIMATE")
                if not hasattr(model, "predict_proba"):
                    st.caption("Score is a squashed SVM margin, not a calibrated probability.")

                st.session_state["last_vec"] = vec
        else:
            st.caption("Results appear here after you click **Analyze Email**.")

    # Which words pushed the decision?
    if go_btn and has_coef and text and len(text.strip()) >= 5:
        row = vectorizer.transform([text]).tocsr()
        feats = vectorizer.get_feature_names_out()
        contrib = row.data * model.coef_[0][row.indices]
        if len(contrib):
            dfc = pd.DataFrame({"Word": feats[row.indices], "Contribution": contrib})
            dfc = dfc.reindex(dfc["Contribution"].abs().sort_values(ascending=False).index).head(12)
            dfc["Pushes toward"] = np.where(dfc["Contribution"] > 0, "Spam", "Legitimate")
            fig = px.bar(dfc, x="Contribution", y="Word", orientation="h", color="Pushes toward",
                         color_discrete_map={"Spam": SPAM_COLOR, "Legitimate": HAM_COLOR},
                         title="Why this verdict? Words in YOUR email that mattered most")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("None of the words in this email are in the model's vocabulary.")

# ================================================================ BATCH SCAN
with tab_batch:
    st.subheader("Batch Scan: upload many emails at once")
    up = st.file_uploader("Upload a CSV file", type=["csv"])
    if up is not None:
        try:
            df_up = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")
            df_up = None
        if df_up is not None and len(df_up.columns):
            default = list(df_up.columns).index("text") if "text" in df_up.columns else 0
            col = st.selectbox("Which column has the email text?", df_up.columns, index=default)
            thr = st.slider("Spam threshold", 0.05, 0.95, 0.50, 0.05, key="batch_thr")
            Xb = vectorizer.transform(df_up[col].fillna("").astype(str))
            probs = to_spam_prob(model, Xb)
            out = df_up.copy()
            out["spam_score"] = probs.round(4)
            out["verdict"] = np.where(probs >= thr, "Spam", "Legitimate")

            a, b = st.columns([1, 2])
            counts = out["verdict"].value_counts().reset_index()
            counts.columns = ["Verdict", "Count"]
            a.plotly_chart(px.pie(counts, names="Verdict", values="Count", hole=0.5, color="Verdict",
                                  color_discrete_map={"Spam": SPAM_COLOR, "Legitimate": HAM_COLOR}),
                           width="stretch")
            b.plotly_chart(px.histogram(out, x="spam_score", color="verdict", nbins=30,
                                        color_discrete_map={"Spam": SPAM_COLOR, "Legitimate": HAM_COLOR},
                                        title="Distribution of spam scores"), width="stretch")
            st.dataframe(out, width="stretch")
            st.download_button("Download results as CSV", out.to_csv(index=False).encode(),
                               "mailguard_results.csv", "text/csv")
    else:
        st.caption("Tip: any CSV with a column of email text works. Results include a score and verdict per row.")

# ================================================================ COMPARISON
with tab_compare:
    st.subheader("Model Comparison")
    st.write(f"**{best_row['Model']}** was selected automatically: highest F1-Score "
             f"({best_row['F1-Score']:.4f}), with Precision as the tie-breaker.")

    f1, f2, f3 = st.columns(3)
    sel_models = f1.multiselect("Models", df_metrics["Model"].tolist(), default=df_metrics["Model"].tolist())
    sel_metrics = f2.multiselect("Metrics", ["Accuracy", "Precision", "Recall", "F1-Score"],
                                 default=["Accuracy", "Precision", "Recall", "F1-Score"])
    ymin = f3.slider("Zoom: y-axis minimum", 0.0, 0.99, 0.95, 0.01)

    sub = df_metrics[df_metrics["Model"].isin(sel_models)]
    if sub.empty or not sel_metrics:
        st.warning("Select at least one model and one metric.")
    else:
        melt = sub.melt(id_vars="Model", value_vars=sel_metrics, var_name="Metric", value_name="Score")
        fig = px.bar(melt, x="Model", y="Score", color="Metric", barmode="group", text_auto=".3f")
        fig.update_layout(yaxis_range=[ymin, 1.0], title="Algorithm comparison (hover for exact values)")
        st.plotly_chart(fig, width="stretch")

        g1, g2 = st.columns(2)
        sc = px.scatter(sub, x="Recall", y="Precision", size=sub["Train_Time_sec"].clip(lower=0.05),
                        color="Model", hover_data=["Accuracy", "F1-Score", "Train_Time_sec"],
                        title="Precision vs Recall (bubble size = training time)")
        g1.plotly_chart(sc, width="stretch")
        tt = px.bar(sub, x="Model", y="Train_Time_sec", color="Model", text_auto=".2f",
                    title="Training time (seconds)")
        g2.plotly_chart(tt, width="stretch")

        st.dataframe(sub.style.format({c: "{:.4f}" for c in sub.columns if c != "Model"}),
                     width="stretch", hide_index=True)

# ================================================================ INSIGHTS
with tab_insights:
    st.subheader("Model Insights: How does the AI think?")
    if not has_coef:
        st.warning("Feature importance is not available for this model type.")
    else:
        coefs = model.coef_[0]
        feats = vectorizer.get_feature_names_out()
        i1, i2 = st.columns([2, 1])
        top_n = i1.slider("How many top words?", 5, 50, 20)
        show = i2.radio("Show", ["Both", "Spam signals", "Legitimate signals"], horizontal=True)

        order = coefs.argsort()
        rows = []
        if show in ("Both", "Spam signals"):
            rows += [{"Word": feats[i], "Weight": coefs[i], "Indicator": "Spam Signal"} for i in order[-top_n:][::-1]]
        if show in ("Both", "Legitimate signals"):
            rows += [{"Word": feats[i], "Weight": coefs[i], "Indicator": "Legitimate Signal"} for i in order[:top_n]]
        fig = px.bar(pd.DataFrame(rows), x="Weight", y="Word", color="Indicator", orientation="h",
                     color_discrete_map={"Spam Signal": SPAM_COLOR, "Legitimate Signal": HAM_COLOR})
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=max(400, 22 * len(rows)))
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### 🔎 Look up any word")
        q = st.text_input("Type a word or phrase from the vocabulary (e.g. free, meeting, click here)").strip().lower()
        if q:
            hits = np.where(feats == q)[0]
            if len(hits):
                w = coefs[hits[0]]
                rank = int((coefs > w).sum()) + 1
                st.metric(f"Weight of '{q}'", f"{w:+.3f}",
                          "pushes toward SPAM" if w > 0 else "pushes toward LEGITIMATE")
                st.caption(f"Rank {rank:,} of {len(coefs):,} (1 = strongest spam signal)")
            else:
                st.info("That word is not in the model's vocabulary.")

        hist = px.histogram(x=coefs, nbins=60, labels={"x": "Weight"},
                            title="Distribution of all word weights (most words are near 0)")
        st.plotly_chart(hist, width="stretch")

# ================================================================ DATA & EVALUATION
with tab_eval:
    st.subheader("Data & Evaluation")
    df_data = load_dataset()
    if df_data is None:
        st.info("The dataset isn't available here, so these charts are hidden. "
                "Add `data/emails.csv` to the repo (run `download_data.py`) to enable them.")
    else:
        d1, d2 = st.columns(2)
        dist = df_data["label"].map({0: "Legitimate", 1: "Spam"}).value_counts().reset_index()
        dist.columns = ["Class", "Count"]
        d1.plotly_chart(px.pie(dist, names="Class", values="Count", hole=0.5, color="Class",
                               color_discrete_map={"Spam": SPAM_COLOR, "Legitimate": HAM_COLOR},
                               title=f"Class balance ({len(df_data):,} emails)"), width="stretch")
        lens = df_data.assign(Length=df_data["text"].str.split().str.len().clip(upper=600),
                              Class=df_data["label"].map({0: "Legitimate", 1: "Spam"}))
        d2.plotly_chart(px.histogram(lens, x="Length", color="Class", nbins=50, barmode="overlay",
                                     opacity=0.65, title="Email length in words (capped at 600)",
                                     color_discrete_map={"Spam": SPAM_COLOR, "Legitimate": HAM_COLOR}),
                        width="stretch")

        scores, y_true = test_scores(model, vectorizer, df_data["text"], df_data["label"])
        lo, hi = float(np.percentile(scores, 1)), float(np.percentile(scores, 99))
        default_t = 0.0 if hasattr(model, "decision_function") else 0.5
        st.markdown("#### 🎚️ Move the decision threshold and watch the trade-off")
        t = st.slider("Decision threshold", float(min(lo, -1.0)) if default_t == 0 else 0.0,
                      float(max(hi, 1.0)) if default_t == 0 else 1.0, default_t, 0.05)

        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1v = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{prec:.4f}")
        m2.metric("Recall", f"{rec:.4f}")
        m3.metric("F1-Score", f"{f1v:.4f}")
        m4.metric("False positives (legit marked spam)", fp)

        e1, e2 = st.columns(2)
        cm = px.imshow([[tn, fp], [fn, tp]], text_auto=True, color_continuous_scale="Blues",
                       x=["Pred Legit", "Pred Spam"], y=["Actual Legit", "Actual Spam"],
                       title="Confusion matrix (held-out test set)")
        e1.plotly_chart(cm, width="stretch")

        p_arr, r_arr, _ = precision_recall_curve(y_true, scores)
        pr = go.Figure(go.Scatter(x=r_arr, y=p_arr, mode="lines", name="PR curve"))
        pr.add_trace(go.Scatter(x=[rec], y=[prec], mode="markers", name="Current threshold",
                                marker=dict(size=14, color=SPAM_COLOR)))
        pr.update_layout(title="Precision-Recall curve", xaxis_title="Recall", yaxis_title="Precision")
        e2.plotly_chart(pr, width="stretch")

        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        roc = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc:.4f})"))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
        roc.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(roc, width="stretch")
        st.caption("Raising the threshold cuts false positives (higher precision) but misses more spam (lower recall).")