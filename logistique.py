import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from nltk.tokenize import word_tokenize
import string
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

import warnings

# Ignorer les FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

from IPython.display import HTML
import webbrowser
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


nltk.download('punkt')
nltk.download('stopwords')

print()

################ PRETRAITEMENT DES DONNEES ###############################

nlp = spacy.load('en_core_web_sm')

def afficherTop():    
    print("\n")
    print("jeu: ",df['Jeu'].iloc[0])
    print("player: ",df['Player'].iloc[0])
    print("role: ",df['Role'].iloc[0])
    print("evaluation: ",df['evaluation'].iloc[0])
    print("response:\n",df['Response'].iloc[0])
    print("\n")

#charge le dataframe
df = pd.read_csv('conversation_data.csv', sep=',')

 #afficherTop()
 
# Compter les réponses avec évaluation TRUE et FALSE
true_count = df['evaluation'].sum()
false_count = len(df) - true_count

# Afficher les informations avec un print
print("\nÉvaluation des réponses\n")
print(f"Nombre de réponses avec évaluation TRUE: {true_count}")
print(f"Nombre de réponses avec évaluation FALSE: {false_count}\n")



##########catégorisation de 'role_joueur'
"""
en gros si on a villageois et loups dans role
elle va creer une colone role_villageois et une colone role_loups avec des valeurs binaire
et supprimer la colone role avec des valeurs textuelles
"""
df = pd.get_dummies(df, columns=['Role'])




############binarisation de 'mensonge'
#df['evaluation'] = df['evaluation'].map({"True": 1, "False": 0})

print(df['evaluation'].iloc[0])

############transforme le 'nom_joueur' en id 

#mapping entre chaque nom de joueur et un id
player_id_mapping = {player: i for i, player in enumerate(df['Player'].unique())}

#change le nom par un id
df['id'] = df['Player'].map(player_id_mapping)
df.drop('Player', axis=1, inplace=True)

print("\nFin categorisation\n")

############traitement de 'phrase'
"""
- on met en minuscule
- enleve la ponctuation
- on tokenise la phrase en mots (un token = un mot)
- enleve les stop words (les mots de liaisons)
- lematisation :on reduit les mots a leur forme de racine (normalise les tokens)
    ex: mangeront, mangames -> manger
"""


nlp = spacy.load('fr_core_news_sm')

def process_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nlp(text)
    tokens = [token.lemma_ for token in tokens if not token.is_stop]
    return ' '.join(tokens)

df['Response'] = df['Response'].apply(process_text)

print(df.iloc[0])
print()


stop_words = set(stopwords.words('english'))

def remove_stop_words(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(filtered_tokens)

df['Response'] = df['Response'].apply(remove_stop_words)


df.head()


nlp = spacy.load("en_core_web_sm")

def lemmatize(text):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text

print("\n")
print(df.iloc[0])
print("\n")

df['Response'] = df['Response'].apply(lemmatize)



print("\nFin pretraitement: minuscule, ponctuation, stop words, lemmatisation\n")

############## VECTORISATION ###########################

X_train, X_test, y_train, y_test = train_test_split(df[['Response', 'Jeu', 'id', 'Role_Docteur', 'Role_Voyante', 'Role_Villageois', 'Role_Loups-Garous']], df['evaluation'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train['Response'])
X_test_tfidf = vectorizer.transform(X_test['Response'])

print("\nFin Vectorisation tfidf\n")

#################### ENTRAINEMENT DU CLASSIFIEUR ##################

clf = LogisticRegression(random_state=42)
clf.fit(X_train_tfidf, y_train)


y_pred = clf.predict(X_test_tfidf)

print("\nFin Entrainement du classifieur\n")


################# EVALUATION ############################

# Évaluation du classifieur
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :")
print(conf_matrix)
print()

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Affichage des métriques
print(f"Précision : {precision:.4f}")
print(f"Rappel : {recall:.4f}")
print(f"Score F1 : {f1:.4f}")


print("\nFin evaluation \n")

############### LIME ###########################

################ Explication des poids des mots avec LimeTextExplainer

text_explainer = LimeTextExplainer()

def predict_proba_wrapper(texts):
    text_features = vectorizer.transform(texts)
    return clf.predict_proba(text_features)

word_weights = {}

for i in range(len(X_test)):
    text = X_test['Response'].iloc[i]
    sample_text = X_test_tfidf[i] 
    text_exp = text_explainer.explain_instance(text, predict_proba_wrapper, num_features=20)
    explanation_list = text_exp.as_list()
    for word, weight in explanation_list:
        if word in word_weights:
            word_weights[word].append(weight)
        else:
            word_weights[word] = [weight]

word_avg_weights = {word: np.mean(weights) for word, weights in word_weights.items()}

sorted_words = sorted(word_avg_weights.items(), key=lambda x: x[1])

top_negative_words = sorted_words[:10]
top_positive_words = sorted_words[-10:]

first_response_text = X_test['Response'].iloc[0]
first_response_exp = text_explainer.explain_instance(first_response_text, predict_proba_wrapper, num_features=10)
first_response_exp_html = first_response_exp.as_html()

temp_file_path = "lime_explanations.html"
with open(temp_file_path, "w") as f:
    f.write("<h2>Top 10 mots avec les poids les plus négatifs</h2>")
    f.write("<ul>")
    for word, weight in top_negative_words:
        f.write(f"<li>{word}: {weight:.4f}</li>")
    f.write("</ul>")
    f.write("<h2>Top 10 mots avec les poids les plus positifs</h2>")
    f.write("<ul>")
    for word, weight in top_positive_words:
        f.write(f"<li>{word}: {weight:.4f}</li>")
    f.write("</ul>")
    f.write("<h2>Explication pour la première réponse</h2>")
    f.write(first_response_exp_html)

webbrowser.open(temp_file_path)

print("\nFin evaluation Lime\n")


################ Explication des caractéristiques non textuelles avec LimeTabularExplainer

"""

clf.fit(X_train[['id', 'Role_Villageois', 'Role_Loups-Garous']], y_train)


numeric_explainer = LimeTabularExplainer(X_train[['id', 'Role_Villageois', 'Role_Loups-Garous']].values,
                                          feature_names=['id', 'Role_villageois', 'Role_Loups-Garous'])


for i in range(len(X_test)):
    # Récupération de l'échantillon i
    sample_numeric = X_test[['id', 'Role_Villageois', 'Role_Loups-Garous']].iloc[i]  # Caractéristiques non textuelles
    # Explication avec LIME pour les caractéristiques non textuelles
    numeric_exp = numeric_explainer.explain_instance(sample_numeric, clf.predict_proba, num_features=10)
    print(f"Explication pour l'échantillon {i + 1} - Caractéristiques non textuelles :")
    print(numeric_exp.as_list())
    print()
    
print("\nFin expliccation caractéristiques\n")
"""

"""
#################### TEST 





logreg = LogisticRegression(random_state=16)


logreg.fit(X_train_tfidf, y_train)

y_pred = logreg.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)


################ Evaluation sur de nouvelle donnée ##################################


new_data = ["Ceci est une phrase suspecte.", "Je dis toujours la vérité."]
new_data_tfidf = vectorizer.transform(new_data)
predictions = log_reg.predict(new_data_tfidf)
print("Prédictions pour les nouvelles données:")
for sentence, prediction in zip(new_data, predictions):
    print(f"Phrase: {sentence} - Prédiction: {'Vrai' if prediction == 1 else 'Faux'}")
"""
