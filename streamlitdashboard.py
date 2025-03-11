
import plotly.graph_objs as go
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import matplotlib
import plotly.express as px
import subprocess, shlex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
matplotlib.use("Agg")  # Pour éviter les problèmes de backend


 

# ==============================

# 1) CONFIGURATION STREAMLIT

# ==============================

st.set_page_config(

    page_title="Engie - Tableau de Bord Avancé + Sentiment",

    layout="wide"

)

 

########################################

# 1) CHARGEMENT DU CSV

########################################

@st.cache_data

def load_data(csv_file: str):

    df_ = pd.read_csv(csv_file, sep=";", encoding="utf-8", parse_dates=["created_at"])

    df_["created_at"] = df_["created_at"].dt.tz_localize(None)

    return df_

 

DATA_PATH = "C:/Users/iyous/Downloads/final/final/cleaned_tweets_with_complaints_and_replies.csv"

df = load_data(DATA_PATH)

 

if "month" not in df.columns:

    df["month"] = df["created_at"].dt.to_period("M").astype(str)

 

########################################

# 2) KPI SIMPLES

########################################

df["year"] = df["created_at"].dt.year

tweets_per_month = df.groupby("month")["id"].count()

mean_tweets_per_month = tweets_per_month.mean()

var_tweets_per_month = tweets_per_month.pct_change().fillna(0).mean() * 100

 

engie_accounts = ["engiepartfr", "engiepartsav", "engiegroup"]

for account in engie_accounts:

    if account not in df.columns:

        df[account] = df["cleaned_text"].apply(lambda x: account in x.lower())

 

mention_rate = (df[engie_accounts].sum(axis=1) > 0).mean() * 100

 

keywords_critical = ["délai", "panne", "urgence", "scandale", "problème", "service",

                     "facture", "arnaque", "honteux"]

if "contains_keywords" not in df.columns:

    df["contains_keywords"] = df["cleaned_text"].apply(lambda x: any(k in x.lower() for k in keywords_critical))

alert_rate = (df["contains_keywords"].sum() / len(df)) * 100

 

########################################

# 3) FONCTION RUN_KPI_ANALYSIS

########################################

def run_kpi_analysis(df_kpi: pd.DataFrame):

    st.subheader("Analyse Globale des KPI (Ex-kpi_analysis)")

 

    df_kpi["week"] = df_kpi["created_at"].dt.strftime('%Y-%U')

    df_kpi["month"] = df_kpi["created_at"].dt.strftime('%Y-%m')

    df_kpi["quarter"] = df_kpi["created_at"].dt.to_period("Q").astype(str)

    df_kpi["year"] = df_kpi["created_at"].dt.year

 

    tweets_per_month_kpi = df_kpi.groupby("month")["id"].count()

    tweets_per_month_change_kpi = tweets_per_month_kpi.pct_change().fillna(0) * 100

 

    # Comptes Engie

    engie_accounts_kpi = ["engiepartfr", "engiepartsav", "engiegroup"]

    for acc in engie_accounts_kpi:

        if acc not in df_kpi.columns:

            df_kpi[acc] = df_kpi["cleaned_text"].apply(lambda x: acc in x.lower())

    mention_rate_kpi = (df_kpi[engie_accounts_kpi].sum(axis=1) > 0).mean() * 100

 

    # Mots-clés critiques

    keywords_critical_kpi = ["délai", "panne", "urgence", "scandale", "problème", "service",

                             "facture", "arnaque", "honteux"]

    if "contains_keywords" not in df_kpi.columns:

        df_kpi["contains_keywords"] = df_kpi["cleaned_text"].apply(lambda x: any(k in x.lower() for k in keywords_critical_kpi))

    alert_rate_kpi = (df_kpi["contains_keywords"].sum() / len(df_kpi)) * 100 if len(df_kpi) else 0

 

    # Score d’intensité

    if "intensity_score" not in df_kpi.columns:

        df_kpi["urgency_score"] = df_kpi["cleaned_text"].apply(lambda x: sum(3 for w in ["urgence", "immédiat", "catastrophe", "crise", "critique"] if w in x.lower()))

        df_kpi["complaint_score"] = df_kpi["cleaned_text"].apply(lambda x: sum(2 for w in ["panne", "scandale", "inadmissible", "grave", "honteux", "arnaque"] if w in x.lower()))

        df_kpi["total_complaint_score"] = df_kpi["urgency_score"] + df_kpi["complaint_score"]

        if df_kpi["total_complaint_score"].max() > 0:

            df_kpi["intensity_score"] = (df_kpi["total_complaint_score"] / df_kpi["total_complaint_score"].max()) * 100

        else:

            df_kpi["intensity_score"] = 0

    avg_intensity_score = df_kpi["intensity_score"].mean()

 

    # Taux de réponse

    if "engie_replied" not in df_kpi.columns:

        response_keywords = ["merci", "réponse", "contact", "désolé", "envoyez-nous un message", "notre service client"]

        df_kpi["engie_replied"] = df_kpi["cleaned_text"].apply(lambda x: any(word in x.lower() for word in response_keywords))

    response_rate_kpi = df_kpi["engie_replied"].mean() * 100

 

    # Corrélation

    if df_kpi["engie_replied"].nunique() > 1 and df_kpi["intensity_score"].nunique() > 1:

        correlation_kpi = df_kpi["engie_replied"].corr(df_kpi["intensity_score"])

    else:

        correlation_kpi = 0

 

    peak_month_kpi = tweets_per_month_kpi.idxmax()

    peak_value_kpi = tweets_per_month_kpi.max()

 

    # Affichage

    st.write(f"- **Nombre total de tweets** : {len(df_kpi)}")

    st.write(f"- **Mois avec le plus de tweets** : {peak_month_kpi} ({peak_value_kpi} tweets)")

    st.write(f"- **Variation mensuelle moyenne** : {tweets_per_month_change_kpi.mean():.2f}%")

    st.write(f"- **Taux de mentions Engie** : {mention_rate_kpi:.2f}%")

    st.write(f"- **Taux d’alertes critiques** : {alert_rate_kpi:.2f}%")

    st.write(f"- **Score moyen d’intensité** : {avg_intensity_score:.2f}/100")

    st.write(f"- **Taux de réponse d’Engie** : {response_rate_kpi:.2f}%")

    st.write(f"- **Corrélation (Réponse vs. Intensité)** : {correlation_kpi:.2f}")

 

    # Graphiques avec Plotly

    st.markdown("**Volume des Tweets par Mois (KPI)**")

    fig1 = go.Figure(data=[go.Scatter(x=tweets_per_month_kpi.index, y=tweets_per_month_kpi.values, mode='lines+markers')])

    fig1.update_layout(title="Volume des Tweets par Mois", xaxis_title="Mois", yaxis_title="Nombre de Tweets")

    st.plotly_chart(fig1)

 

    

    fig2 = go.Figure(data=[go.Histogram(x=df_kpi["intensity_score"], nbinsx=20, histnorm="probability")])

    fig2.update_layout(title="Distribution du Score d’Intensité", xaxis_title="Score d’Intensité", yaxis_title="Fréquence")

    st.plotly_chart(fig2)



   

    text_data = " ".join(df_kpi["cleaned_text"].dropna())

    if len(text_data) > 0:

        wc = WordCloud(width=800, height=400, background_color="white").generate(text_data)

        fig_wc = go.Figure(go.Image(z=np.array(wc.to_array())))

        fig_wc.update_layout(title="Nuage de Mots")

        st.plotly_chart(fig_wc)

    else:

        st.info("Pas de texte pour générer un nuage de mots.")

    df['trimestre'] = df['created_at'].dt.to_period('Q').astype(str)  # Convertir Period en str

    tweets_par_trimestre = df.groupby('trimestre').size().reset_index(name='nombre_tweets')

 

    fig = px.line(

        tweets_par_trimestre,

        x='trimestre',

        y='nombre_tweets',

        title="Évolution des Tweets (Trimestrielle)",

        markers=True

    )  

 

    st.plotly_chart(fig)

    st.write("---")

########################################
# 4) CODE CLASSEMENT RECLAMATION + SENTIMENT
########################################
cache = {}

def train_local_classifier(df_train):
    X = df_train["cleaned_text"]
    y = df_train["complaint_category"]
    vectorizer = TfidfVectorizer(max_features=1000)
    X_vec = vectorizer.fit_transform(X)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_vec, y)
    return vectorizer, classifier

def classify_local(text, vectorizer, classifier):
    X_vec = vectorizer.transform([text])
    pred = classifier.predict(X_vec)[0]
    prob = classifier.predict_proba(X_vec).max()
    score = int(prob * 100)
    auto_reply = f"Réponse générée localement pour la catégorie {pred}"
    return pred, score, auto_reply


def classify_complaint_and_sentiment(text, vectorizer=None, classifier=None, use_local=False):
    """
    1) Classifie la réclamation (catégorie + inconfort).
    2) Analyse le sentiment (Positif / Neutre / Négatif).
    """
    if text in cache:
        # Already processed
        return cache[text]

    # Default: external classification (Ollama)
    category, discomfort_score, raw_response, auto_reply = ("Autre",50,"","Nous analysons...")

    # If local is enabled & we have a local classifier
    if use_local and (vectorizer is not None) and (classifier is not None):
        try:
            cat, sc, rep = classify_local(text, vectorizer, classifier)
            category = cat
            discomfort_score = sc
            raw_response = f"Classification locale: {cat}, Score: {sc}"
            auto_reply = rep
        except:
            pass

    # If local classification didn't set raw_response => fallback to Ollama
    if raw_response=="" or raw_response.startswith("Nous analysons..."):
        # Make sure we didn't skip external classification:
        model_name = "llama3.2:latest"
        prompt = f"""Classifie la réclamation suivante et génère une réponse adaptée :

**Catégories possibles** :
- Facturation (montant erroné, prélèvement injustifié)
- Pannes et urgences (coupure gaz, problème électricité, eau chaude)
- Service client injoignable (relances, absence de réponse)
- Application Engie (bug, indisponibilité)
- Délai d’intervention (retard de prise en charge)
- Autre (si aucune catégorie ne convient)

**Format de réponse attendu** :
Catégorie : <NomCatégorie>
Score d'inconfort : <Valeur entre 0 et 100>
Réponse : <Message adapté>

**Tweet** : {text}

Réponds en respectant ce format strictement.
"""
        command = f'ollama run {shlex.quote(model_name)} {shlex.quote(prompt)}'
        try:
            res = subprocess.run(command, shell=True, check=True, 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            lines = res.stdout.strip().split("\n")
            cat = "Autre"
            if len(lines)>0 and "Catégorie" in lines[0]:
                cat = lines[0].replace("Catégorie :","").strip()
            try:
                disc = int(lines[1].replace("Score d'inconfort :","").strip())
            except:
                disc=50
            rep = "Nous analysons votre situation."
            if len(lines)>2 and "Réponse :" in lines[2]:
                rep = lines[2].replace("Réponse :","").strip()

            category, discomfort_score, raw_response, auto_reply = (cat, disc, res.stdout.strip(), rep)
        except:
            pass

    # Next => sentiment
    sentiment, sentiment_score, raw_sent = analyze_sentiment_ollama(text)

    out = (category, discomfort_score, raw_response, auto_reply, sentiment, sentiment_score, raw_sent)
    cache[text] = out
    return out


def analyze_sentiment_ollama(text):
    """
    Analyse le sentiment du tweet (Positif, Neutre, Négatif) via Ollama
    """
    if not text or text.strip()=="":
        return ("Neutre", 50, "Pas de texte")

    model_name = "llama3.2:latest"
    prompt = f"Analyse le sentiment du tweet suivant : {text}.\nRéponds uniquement par Positif, Neutre ou Négatif."
    command = f'ollama run {shlex.quote(model_name)} {shlex.quote(prompt)}'
    try:
        result = subprocess.run(command, shell=True, check=True, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        raw = result.stdout.strip()
        if raw in ["Positif","Neutre","Négatif"]:
            sent = raw
        else:
            sent = "Neutre"
        sscore = {"Positif":100,"Neutre":50,"Négatif":0}.get(sent,50)
        return (sent, sscore, raw)
    except:
        return ("Neutre",50,"Erreur Sentiment")


#######################################
# 5) BARRE LATÉRALE FILTRES
#######################################
st.sidebar.title("Filtres Avancés")

min_date = df["created_at"].min().date()
max_date = df["created_at"].max().date()
start_date = st.sidebar.date_input("Date de début", min_date)
end_date = st.sidebar.date_input("Date de fin", max_date)

# Filtre complaint_category
if "complaint_category" in df.columns:
    all_categories = df["complaint_category"].dropna().unique().tolist()
    selected_categories = st.sidebar.multiselect("Catégories :", all_categories, default=all_categories)
else:
    selected_categories = []

# Filtre sentiment
if "sentiment" in df.columns:
    all_sentiments = df["sentiment"].dropna().unique().tolist()
    selected_sentiments = st.sidebar.multiselect("Sentiments :", all_sentiments, default=all_sentiments)
else:
    selected_sentiments = []

st.sidebar.write("**Filtres sur comptes Engie :**")
selected_accounts = []
for acc in engie_accounts:
    if st.sidebar.checkbox(acc, value=False):
        selected_accounts.append(acc)

filter_keywords = st.sidebar.checkbox("Uniquement tweets cont. mots-clés critiques ?", value=False)

if "discomfort_score" in df.columns:
    smin = int(df["discomfort_score"].min())
    smax = int(df["discomfort_score"].max())
    min_sc, max_sc = st.sidebar.slider("Score d’inconfort", 0, 100, (smin, smax))
else:
    min_sc, max_sc = (0, 100)

#######################################
# 6) APPLICATION DES FILTRES
#######################################
df_filtered = df.copy()
df_filtered = df_filtered[df_filtered["created_at"].between(pd.to_datetime(start_date), pd.to_datetime(end_date))]

if selected_categories and "complaint_category" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["complaint_category"].isin(selected_categories)]

if selected_sentiments and "sentiment" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["sentiment"].isin(selected_sentiments)]

if selected_accounts:
    mask = False
    for a in selected_accounts:
        mask = mask | (df_filtered[a] == True)
    df_filtered = df_filtered[mask]

if filter_keywords:
    df_filtered = df_filtered[df_filtered["contains_keywords"] == True]

if "discomfort_score" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["discomfort_score"] >= min_sc) &
        (df_filtered["discomfort_score"] <= max_sc)
    ]

#######################################
# 7) TABLEAU DE BORD + KPI
#######################################
st.title("Tableau de Bord - Engie (Avancé)")

with st.expander("Afficher l'Analyse Globale des KPI", expanded=False):
    run_kpi_analysis(df)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Tweets filtrés", len(df_filtered))
with col2:
    st.metric("Moy Tweets/mois", f"{mean_tweets_per_month:.2f}")
with col3:
    st.metric("Variation/mois", f"{var_tweets_per_month:.2f}%")
with col4:
    st.metric("Mentions Engie", f"{mention_rate:.2f}%")
with col5:
    st.metric("Alertes critiques", f"{alert_rate:.2f}%")

st.markdown("---")

# Graphique altair: complaint_category
if "complaint_category" in df_filtered.columns and len(df_filtered) > 0:
    st.subheader("Répartition des Catégories")
    cat_chart = (
        alt.Chart(df_filtered)
        .mark_bar()
        .encode(
            x=alt.X("count()", title="Nombre de Tweets"),
            y=alt.Y("complaint_category:N", sort="-x", title="Catégorie")
        )
        .interactive()
    )
    st.altair_chart(cat_chart, use_container_width=True)
else:
    st.info("Aucune catégorie ou aucune donnée filtrée.")

st.subheader("Évolution des Plaintes (par Mois)")
df_filtered["month_period"] = df_filtered["created_at"].dt.to_period("M").astype(str)
complaints_o_time = df_filtered.groupby(["month_period","complaint_category"]).size().reset_index(name="count")
if len(complaints_o_time) > 0:
    t_chart = (
        alt.Chart(complaints_o_time)
        .mark_line(point=True)
        .encode(
            x=alt.X("month_period:N", sort=alt.SortField(field="month_period", order="ascending")),
            y="count:Q", color="complaint_category:N",
            tooltip=["month_period","complaint_category","count"]
        )
        .interactive()
    )
    st.altair_chart(t_chart, use_container_width=True)
else:
    st.info("Pas de données temporelles filtrées.")

if "discomfort_score" in df_filtered.columns and len(df_filtered) > 0:
    st.subheader("Distribution du Score d’Inconfort")
    disco_chart = (
        alt.Chart(df_filtered)
        .mark_bar()
        .encode(
            alt.X("discomfort_score:Q", bin=alt.Bin(maxbins=20)),
            alt.Y("count()")
        )
        .interactive()
    )
    st.altair_chart(disco_chart, use_container_width=True)

# Corrélation sentiment vs discomfort
if "sentiment_score" in df_filtered.columns and "discomfort_score" in df_filtered.columns and len(df_filtered) > 0:
    st.subheader("Corrélation Sentiment vs. Inconfort")
    s_chart = (
        alt.Chart(df_filtered)
        .mark_circle(size=60)
        .encode(
            x="sentiment_score:Q", 
            y="discomfort_score:Q", 
            color="sentiment:N",
            tooltip=["cleaned_text","sentiment","sentiment_score","discomfort_score"]
        )
        .interactive()
    )
    st.altair_chart(s_chart, use_container_width=True)


########################################
# (8.BIS) AFFICHER LA CLASSIFICATION EXISTANTE
########################################
st.subheader("Afficher la Classification (Réclamation + Sentiment) Déjà Existante")
cols_class = ["cleaned_text","complaint_category","discomfort_score","auto_reply","sentiment","sentiment_score"]
existing_cols = [c for c in cols_class if c in df.columns]

if len(existing_cols)>1:
    st.write("Voici un aperçu de la classification déjà présente :")
    st.dataframe(df[existing_cols].head(20))
else:
    st.info("Aucune classification existante (complaint_category / sentiment).")

########################################
# 9) BOUTON "CLASSIFIER" TOUT LES TWEETS
########################################
st.subheader("Classification IA : Réclamation + Sentiment")

# By default, set it to False so we prefer Ollama unless user explicitly wants local.
use_local_checkbox = st.checkbox("Utiliser le classifieur local (si possible)", value=False)

if st.button("Lancer la Classification sur tous les Tweets"):
    df_temp = df.copy()
    vectorizer, classifier = None, None
    local_enabled = False

    # (A) If user checks the box AND complaint_category is present => local training
    if "complaint_category" in df_temp.columns and use_local_checkbox:
        st.write("Entraînement du classifieur local TF-IDF + LogisticRegression...")
        try:
            vectorizer, classifier = train_local_classifier(df_temp)
            local_enabled = True
            st.success("Classifieur local entraîné avec succès.")
        except Exception as e:
            st.warning(f"Impossible d'entraîner le classifieur local : {e}")
            local_enabled = False

    # (B) For each tweet => complaint + sentiment
    new_cat, new_disc, new_raw, new_ar, new_sent, new_sent_score, new_sent_raw = [], [], [], [], [], [], []

    for txt in df_temp["cleaned_text"]:
        c, d, raw_cmp, ar, s, sscore, raw_s = classify_complaint_and_sentiment(
            txt,
            vectorizer=vectorizer,
            classifier=classifier,
            use_local=local_enabled
        )
        new_cat.append(c)
        new_disc.append(d)
        new_raw.append(raw_cmp)
        new_ar.append(ar)
        new_sent.append(s)
        new_sent_score.append(sscore)
        new_sent_raw.append(raw_s)

    # (C) Add columns
    df_temp["complaint_category"] = new_cat
    df_temp["discomfort_score"] = new_disc
    df_temp["ollama_complaint_response"] = new_raw
    df_temp["auto_reply"] = new_ar
    df_temp["sentiment"] = new_sent
    df_temp["sentiment_score"] = new_sent_score
    df_temp["raw_sentiment_resp"] = new_sent_raw  # Optionnel

    # (D) Save
    df_temp.to_csv("cleaned_tweets_with_complaints_and_replies.csv", sep=";", index=False, encoding="utf-8")
    st.success("Classification terminée (Réclamation + Sentiment). CSV mis à jour.")
    st.write(df_temp.head(10))

st.write("---")
st.write("© 2025 - Dashboard Interactif Engie (Réclamation + Sentiment)")