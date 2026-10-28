import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import joblib
import pandas as pd
import json
import os

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
app.title = "MailGuard: Spam Filter"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
vec_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
except Exception as e:
    model = None
    vectorizer = None
    metrics = None
    print(f"Failed to load models or metrics: {e}")

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("Live Filter", href="/predict")),
        dbc.NavItem(dbc.NavLink("Model Comparison", href="/comparison")),
        dbc.NavItem(dbc.NavLink("Spam Insights", href="/insights")),
    ],
    brand="🛡️ MailGuard",
    brand_href="/",
    color="dark",
    dark=True,
    className="mb-4 border-bottom border-secondary",
)

home_layout = dbc.Container([
    html.H1("Intelligent Email Spam Classification", className="text-center mt-5 mb-4 text-light"),
    html.P("Protecting inboxes with state-of-the-art NLP and Machine Learning.", className="text-center text-muted fs-5 mb-5"),
    
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("10,900+", className="text-info text-center"),
            html.P("Emails Analyzed", className="text-center text-muted")
        ])), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("NLP Powered", className="text-success text-center"),
            html.P("TF-IDF Vectorization", className="text-center text-muted")
        ])), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("99.7% Precision", className="text-warning text-center"),
            html.P("Protecting Legit Mail", className="text-center text-muted")
        ])), md=4),
    ], className="mb-5"),
    
    html.Div(dbc.Button("Scan an Email", color="primary", size="lg", href="/predict"), className="d-flex justify-content-center")
], fluid=True)


predict_layout = dbc.Container([
    html.H2("Live Spam Scanner", className="mb-4 text-light"),
    html.P("Paste the content of a suspicious email below. Our ML model will instantly predict if it's safe or a threat.", className="text-muted"),
    
    dbc.Textarea(id="email-input", placeholder="Dear user, you have won a $1000 gift card...", style={"height": "250px"}, className="mb-3"),
    dbc.Button("Analyze Email", id="analyze-btn", color="success", className="mb-4 w-100", size="lg"),
    
    dbc.Spinner(html.Div(id="prediction-result"))
])

def create_comparison_plot():
    if not metrics:
        return html.Div("Metrics not found.")
        
    df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
    df_melt = df_metrics.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score"], var_name="Metric", value_name="Score")
    
    fig = px.bar(
        df_melt, x="Model", y="Score", color="Metric", barmode="group", text_auto=".3f",
        title="Algorithm Comparison"
    )
    fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis_title="Score (0 to 1)", yaxis_range=[0.95, 1.0])
    
    return dcc.Graph(figure=fig), df_metrics

comparison_layout = html.Div()
if metrics:
    comp_graph, df_metrics = create_comparison_plot()
    comparison_layout = dbc.Container([
        html.H2("Model Comparison", className="mb-4 text-light"),
        html.P("Notice how all models perform incredibly well, but Linear SVC wins the tie-breaker by having higher Precision, minimizing false positives.", className="text-muted mb-4"),
        dbc.Card(dbc.CardBody([comp_graph]), className="mb-4 shadow-sm"),
        dbc.Card(dbc.CardBody([
            html.H4("Metrics Table", className="card-title text-center mb-4"),
            dbc.Table.from_dataframe(df_metrics, striped=True, bordered=True, hover=True, color="dark")
        ]), className="mb-5")
    ])

def create_feature_importance_plot(trained_model, trained_vectorizer, top_n=20):
    if not trained_model or not hasattr(trained_model, 'coef_'):
        return dbc.Alert("Feature importance not supported for this model.", color="warning")
    
    coefs = trained_model.coef_[0]
    features = trained_vectorizer.get_feature_names_out()
    
    top_spam_idx = coefs.argsort()[-top_n:][::-1]
    top_ham_idx = coefs.argsort()[:top_n]
    
    data = [{"Word": features[i], "Weight": coefs[i], "Indicator": "Spam Signal"} for i in top_spam_idx] + \
           [{"Word": features[i], "Weight": coefs[i], "Indicator": "Legitimate Signal"} for i in top_ham_idx]
        
    df_feat = pd.DataFrame(data)
    
    fig = px.bar(
        df_feat, x="Weight", y="Word", color="Indicator", orientation='h',
        color_discrete_map={"Spam Signal": "#ef5350", "Legitimate Signal": "#66bb6a"}
    )
    fig.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=40, b=20))
    return dcc.Graph(figure=fig)

insights_layout = dbc.Container([
    html.H2("Model Insights: How does the AI think?", className="mb-4 text-light"),
    html.P("This chart extracts the exact coefficients from the Linear SVC model. Words with a high positive weight push the prediction toward 'Spam', while negative weights pull it toward 'Legitimate'.", className="text-muted mb-4"),
    dbc.Card(dbc.CardBody([
        html.H4("Top Predictive Features (TF-IDF)", className="card-title text-center mb-4"),
        create_feature_importance_plot(model, vectorizer)
    ]), className="mb-5 shadow-sm")
])


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/predict': return predict_layout
    elif pathname == '/insights': return insights_layout
    elif pathname == '/comparison': return comparison_layout
    else: return home_layout

@app.callback(
    Output("prediction-result", "children"),
    Input("analyze-btn", "n_clicks"),
    State("email-input", "value"),
    prevent_initial_call=True
)
def predict_email(n_clicks, text):
    if not text or len(text.strip()) < 5:
        return dbc.Alert("Please enter a valid email body.", color="warning")
    if not model or not vectorizer:
        return dbc.Alert("Model not loaded.", color="danger")
    
    try:
        clean_input = text.lower()
        vec_input = vectorizer.transform([clean_input])
        prediction = model.predict(vec_input)[0]
        
        prob_text = ""
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(vec_input)[0][1] * 100 if prediction == 1 else model.predict_proba(vec_input)[0][0] * 100
            prob_text = f"Confidence: {prob:.1f}%"
        elif hasattr(model, "decision_function"):
            score = model.decision_function(vec_input)[0]
            prob_text = f"Decision Score: {score:.2f} (Dist. from hyperplane)"
        
        if prediction == 1:
            return dbc.Card(dbc.CardBody([
                html.H3("🚨 SPAM DETECTED", className="text-danger text-center"),
                html.P(prob_text, className="text-center text-muted mb-0")
            ]), color="danger", outline=True, className="mt-4")
        else:
            return dbc.Card(dbc.CardBody([
                html.H3("✅ SAFE / LEGITIMATE", className="text-success text-center"),
                html.P(prob_text, className="text-center text-muted mb-0")
            ]), color="success", outline=True, className="mt-4")
            
    except Exception as e:
        return dbc.Alert(f"An error occurred: {str(e)}", color="danger")

if __name__ == "__main__":
    app.run(debug=True, port=8050)
