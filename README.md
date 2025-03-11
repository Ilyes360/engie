# 🚀 Analyse des Tweets Clients d'Engie et Paramétrage d'Agents IA

## 📌 Introduction
Ce projet fait partie d'un hackathon visant à analyser les tweets adressés au service client d'Engie. L'objectif est de :
- Nettoyer les données textuelles 🧹
- Extraire des indicateurs clés de performance (KPI)
- Effectuer une analyse de sentiment des tweets
- Catégoriser automatiquement les réclamations avec un agent IA (Mistral ou Gemini)
- Visualiser les résultats dans un tableau de bord interactif (Streamlit ou Power BI)
---

## 🛠 Installation et Prérequis
### 📌 Prérequis
- Python 3.8+
---


---

## 🔍 Nettoyage des Tweets
- Supprime les emojis et les caractères spéciaux
- Retire les liens URL
- Normalise et nettoie le texte
- Convertit la colonne `created_at` au bon format de date
- Ajoute des colonnes : heure du tweet, longueur du texte et présence du mot "ENGIE"

---

## 🤖 Classification et Analyse de Sentiment
- **Modèle IA Ollama** pour classifier les réclamations et analyser les sentiments
- **Modèle local (TF-IDF + Regression Logistique)** pour classifier les tweets si activé
- **Mesure du score d'inconfort** des utilisateurs
- **Calcul du sentiment (Positif, Neutre, Négatif)**


---

## 📊 Tableau de Bord Interactif
- L'affichage des KPI (nombre de tweets, mentions Engie, alertes critiques...)
- La visualisation de l'évolution des réclamations
- L'analyse de sentiment
- L'interaction avec les données filtrées


---

## 📈 KPI Suivis
- **Nombre de tweets** par période
- **Fréquence des mentions** d'Engie
- **Détection de mots-clés critiques** ("panne", "urgence", "problème"...)
- **Score d'inconfort** des utilisateurs
- **Analyse de sentiment** (Positif / Neutre / Négatif)

---
