import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import shlex
import numpy as np
import matplotlib
from wordcloud import WordCloud
import concurrent.futures
import sys
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

matplotlib.use("TkAgg")  # For Mac or certain environments

##############################################################################
# Global cache to avoid re-processing identical tweets
##############################################################################
cache = {}

##############################################################################
# 1) Local Classification (TF–IDF + LogisticRegression)
##############################################################################
def train_local_classifier(df):
    """
    Train a simple classifier (TF-IDF + LogisticRegression).
    Expects df to have 'cleaned_text' and 'complaint_category'.
    """
    print("🔹 Attempting local classification training on df with shape:", df.shape)
    
    X = df["cleaned_text"]
    y = df["complaint_category"]

    vectorizer = TfidfVectorizer(max_features=1000)
    print("   TF-IDF fitting ...")
    X_vec = vectorizer.fit_transform(X)

    print("   LogisticRegression training ...")
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_vec, y)

    print("✅ Local classifier trained successfully.")
    return vectorizer, classifier

def classify_local(text, vectorizer, classifier):
    """
    Classify tweet text locally and return: (category, discomfort_score, local_auto_reply)
    The 'discomfort_score' is mapped from the classifier's max probability * 100
    """
    X_vec = vectorizer.transform([text])
    pred = classifier.predict(X_vec)[0]
    prob = classifier.predict_proba(X_vec).max()
    score = int(prob * 100)
    auto_reply = f"Réponse générée localement pour la catégorie {pred}"
    return pred, score, auto_reply

##############################################################################
# 2) Ollama-based classification for complaint
##############################################################################
def classify_complaint_external(text):
    """
    Call Ollama to classify a tweet's complaint and generate an auto-reply.
    Returns (category, discomfort_score, raw_ollama_output, auto_reply).
    """
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
    cmd = f'ollama run {shlex.quote(model_name)} {shlex.quote(prompt)}'
    try:
        result = subprocess.run(
            cmd, shell=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        raw_output = result.stdout.strip()

        lines = raw_output.split("\n")
        # Attempt to parse lines
        if len(lines)>0 and "Catégorie :" in lines[0]:
            category = lines[0].replace("Catégorie :","").strip()
        else:
            category = "Autre"

        try:
            if len(lines)>1 and "Score d'inconfort :" in lines[1]:
                disc_score = int(lines[1].replace("Score d'inconfort :","").strip())
            else:
                disc_score = 50
        except:
            disc_score = 50

        auto_reply = "Nous analysons votre situation."
        if len(lines)>2 and "Réponse :" in lines[2]:
            auto_reply = lines[2].replace("Réponse :","").strip()

        return category, disc_score, raw_output, auto_reply

    except subprocess.CalledProcessError as e:
        print(f"⚠️ Ollama classification error: {e.stderr.strip()}")
        return ("Autre", 50, "Erreur classification Ollama", "Nous rencontrons un problème, merci de réessayer.")

##############################################################################
# 3) Ollama-based sentiment (Positif / Neutre / Négatif)
##############################################################################
def analyze_sentiment_ollama(text):
    """
    Returns (sentiment, sentiment_score, raw_ollama_sentiment_output).
    sentiment_score = 100 (Positif), 50 (Neutre), 0 (Négatif).
    """
    if not text or not text.strip():
        return ("Neutre", 50, "Pas de texte (vide)")

    model_name = "llama3.2:latest"
    prompt = f"Analyse le sentiment du tweet suivant : {text}.\nRéponds uniquement par Positif, Neutre ou Négatif."
    cmd = f'ollama run {shlex.quote(model_name)} {shlex.quote(prompt)}'
    try:
        out = subprocess.run(cmd, shell=True, check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        raw = out.stdout.strip()

        if raw in ["Positif","Neutre","Négatif"]:
            sentiment = raw
        else:
            sentiment = "Neutre"
        sc = {"Positif":100,"Neutre":50,"Négatif":0}.get(sentiment,50)
        return (sentiment, sc, raw)

    except subprocess.CalledProcessError as e:
        print(f"⚠️ Ollama sentiment error: {e.stderr.strip()}")
        return ("Neutre",50,"Erreur sentiment")

##############################################################################
# 4) Merge classification + sentiment
##############################################################################
def classify_and_sentiment(text, vectorizer=None, classifier=None, use_local=False):
    """
    Returns a tuple of 7 items:
     (complaint_category, discomfort_score, raw_complaint_resp, auto_reply,
      sentiment, sentiment_score, raw_sentiment_resp)
    """
    if not text.strip():
        return ("Autre",50,"Aucun texte","Nous analysons", "Neutre",50,"Pas de texte")

    # Check cache to avoid re-calling
    if text in cache:
        return cache[text]

    # 1) Complaint classification
    category, discomfort, raw_cmp, auto_r = ("Autre",50,"","Nous analysons")
    # => local or fallback
    if use_local and vectorizer and classifier:
        try:
            cat,score,rep = classify_local(text, vectorizer, classifier)
            category, discomfort = cat, score
            raw_cmp = f"Classification locale: {cat}, Score: {score}"
            auto_r = rep
        except Exception as e:
            print("Local classify error:", e)

    if raw_cmp=="":
        # fallback external
        cat, disc, raw_out, rep = classify_complaint_external(text)
        category, discomfort, raw_cmp, auto_r = cat, disc, raw_out, rep

    # 2) Sentiment analysis
    sent, sscore, raw_s = analyze_sentiment_ollama(text)

    merged = (category, discomfort, raw_cmp, auto_r, sent, sscore, raw_s)
    cache[text] = merged
    return merged

##############################################################################
# 5) MAIN PROCESS
##############################################################################
if __name__ == "__main__":
    print("=== Starting Merged Classification + Sentiment Script ===")

    # 5.1 Load CSV
    input_csv = "cleaned_tweets_engie.csv"
    print(f"Reading CSV from: {input_csv}")
    try:
        df = pd.read_csv(input_csv, sep=";", encoding="utf-8", parse_dates=["created_at"])
        df["created_at"] = df["created_at"].dt.tz_localize(None)
    except Exception as e:
        print("❌ Could not read input CSV. Error:", e)
        sys.exit(1)

    print("Number of tweets loaded:", len(df))

    # 5.2 Optional local training if 'complaint_category' exists
    use_local = False
    vectorizer, classifier = None, None

    if "complaint_category" in df.columns:
        try:
            print("🔹 Attempting local classifier training...")
            vectorizer, classifier = train_local_classifier(df)
            use_local = True
        except Exception as e:
            print(f"⚠️ No local classifier trained: {e}")
            use_local = False
    else:
        print("No 'complaint_category' column found - skipping local training.")
        use_local = False

    # 5.3 Classify + sentiment in parallel
    texts = df["cleaned_text"].tolist()
    results = [None]*len(texts)
    print("ThreadPool concurrency starts with max_workers=8.")
    print("Processing tweets...")

    def process_one(i, txt):
        print(f"Processing tweet #{i} => {txt[:40]!r} ...")
        return i, classify_and_sentiment(txt, vectorizer, classifier, use_local)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(process_one,i,t): i for i,t in enumerate(texts)}
        for fut in concurrent.futures.as_completed(future_map):
            idx, merged_out = fut.result()
            results[idx] = merged_out

    print("All tweets processed. Unpacking results...")

    # 5.4 Unpack
    # each merged_out = (category, discomfort, raw_cmp, auto_r, sentiment, sscore, raw_s)
    cat_list, disc_list, raw_cmp_list, auto_r_list, sent_list, sscore_list, raw_s_list = zip(*results)

    df["complaint_category"] = cat_list
    df["discomfort_score"] = disc_list
    df["ollama_complaint_response"] = raw_cmp_list
    df["auto_reply"] = auto_r_list
    df["sentiment"] = sent_list
    df["sentiment_score"] = sscore_list
    df["ollama_sentiment_response"] = raw_s_list

    # 5.5 Save final CSV
    out_csv = "cleaned_tweets_with_complaints_and_sentiment1.csv"
    try:
        df.to_csv(out_csv, sep=";", encoding="utf-8", index=False)
        print(f"✅ Saved final CSV to: {out_csv}")
    except Exception as e:
        print("❌ Error saving final CSV:", e)

    # 5.6 Basic Visualizations
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from wordcloud import WordCloud

    print("Creating basic visualizations...")

    # Distribution of complaint_category
    if "complaint_category" in df.columns:
        plt.figure(figsize=(10,5))
        sns.countplot(x="complaint_category", data=df, 
                      order=df["complaint_category"].value_counts().index, 
                      palette="Set2")
        plt.title("Répartition des Types de Réclamations")
        plt.xticks(rotation=15)
        plt.xlabel("Catégorie")
        plt.ylabel("Nombre de Tweets")
        plt.show()

    # Distribution of discomfort_score
    if "discomfort_score" in df.columns:
        plt.figure(figsize=(10,5))
        sns.histplot(df["discomfort_score"], bins=20, kde=True, color="red")
        plt.title("Distribution des Scores d’Inconfort")
        plt.xlabel("Score d'Inconfort (0-100)")
        plt.ylabel("Nombre de Tweets")
        plt.show()

    # Sentiment count
    if "sentiment" in df.columns:
        plt.figure(figsize=(8,5))
        sns.countplot(x="sentiment", data=df,
                      order=["Positif","Neutre","Négatif"],
                      palette=["green","gray","red"])
        plt.title("Répartition du Sentiment (Ollama)")
        plt.xlabel("Sentiment")
        plt.ylabel("Nombre de Tweets")
        plt.show()

    # Scatter: discomfort vs sentiment_score
    if "discomfort_score" in df.columns and "sentiment_score" in df.columns:
        plt.figure(figsize=(10,5))
        sns.scatterplot(x="discomfort_score", y="sentiment_score", hue="sentiment",
                        data=df, palette=["green","gray","red"])
        plt.title("Relation Score d’Inconfort vs. Score de Sentiment")
        plt.xlabel("Score d’Inconfort (0-100)")
        plt.ylabel("Score de Sentiment (0=Nég,50=Neutre,100=Pos)")
        plt.grid()
        plt.show()

    # Word cloud for each sentiment if you want
    sents = ["Positif","Neutre","Négatif"]
    for s in sents:
        subset = df[df["sentiment"]==s]["cleaned_text"].dropna()
        if len(subset)>0:
            text_data = " ".join(subset)
            wc = WordCloud(width=800, height=400, background_color="white").generate(text_data)
            plt.figure(figsize=(10,5))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Nuage de Mots - Sentiment {s}")
            plt.show()

    print("✅ Visualizations done. End of script.")

    # Optional: Flask endpoint
    from flask import Flask, request, jsonify
    app = Flask(__name__)

    @app.route("/classify", methods=["POST"])
    def classify_endpoint():
        data = request.get_json()
        tweet_text = data.get("text","")
        out = classify_and_sentiment(tweet_text, vectorizer, classifier, use_local)
        (compl_cat, disc_score, raw_cmp, auto_r, sentiment, sscore, raw_sent) = out
        return jsonify({
            "complaint_category": compl_cat,
            "discomfort_score": disc_score,
            "ollama_complaint_response": raw_cmp,
            "auto_reply": auto_r,
            "sentiment": sentiment,
            "sentiment_score": sscore,
            "ollama_sentiment_response": raw_sent
        })

    # Uncomment below if you want to run Flask server on port 5000:
    # app.run(debug=True)
