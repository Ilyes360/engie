
# Nettoyage des Données de Tweets

Ce fichier décrit les étapes de nettoyage des données de tweets afin de les préparer pour une analyse de sentiment ou toute autre analyse.

## Étapes de nettoyage

Les étapes suivantes ont été suivies pour préparer les données des tweets :

### 1. Suppression des URLs
Les URLs dans les tweets peuvent perturber l'analyse, donc elles ont été supprimées en utilisant une expression régulière pour capturer et supprimer tous les liens commençant par `http` ou `https`.

```python
df['full_text'] = df['full_text'].str.replace(r'http\S+', '', regex=True)

