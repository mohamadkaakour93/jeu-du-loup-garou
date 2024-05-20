# Jeu de Données Loups-Garous
Ce dépôt contient un jeu de données généré pour simuler des interactions dans le jeu de rôle "Loups-Garous". Les données sont annotées manuellement et peuvent être utilisées pour entraîner des modèles de détection de mensonge ou pour des analyses linguistiques. 
Instructions pour Télécharger les Données

Vous pouvez télécharger les données en cliquant sur le lien suivant :conversation_data.csv.
Structure des Données

Le fichier CSV contient les colonnes suivantes :

    Jeu : Le numéro du jeu auquel le message appartient.
    Nom du Joueur : Le nom du joueur qui a envoyé le message.
    Role : Le rôle attribué au joueur qui a envoyé le message (par exemple, Loup-Garou, Villageois, Voyante, etc.).
    Message : Le texte du message échangé entre les joueurs.
    Etiquette Vrai/Faux : Une étiquette indiquant si le message est vrai ou faux.

Exemples d'Utilisation
Python

Voici comment charger les données dans Python à l'aide de la bibliothèque pandas : 
#import de la bibliothèque pandas
import pandas as pd

# Charger le fichier CSV
df = pd.read_csv("conversation_data.csv")

# Afficher les premières lignes du DataFrame
print(df.head())


