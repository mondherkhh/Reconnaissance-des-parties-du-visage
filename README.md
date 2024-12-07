# Importer les packages nécessaires
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

# Construire l'analyseur d'arguments et analyser les arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="chemin vers le prédicteur de repère facial")
ap.add_argument("-i", "--image", required=True,
                help="chemin vers l'image d'entrée")
args = vars(ap.parse_args())

# Initialiser le détecteur de visage de dlib (basé sur HOG) puis créer
# le prédicteur de repère facial
detecteur = dlib.get_frontal_face_detector()
predicteur = dlib.shape_predictor(args["shape_predictor"])

# Charger l'image d'entrée, redimensionnez-la et convertissez-la en niveaux de gris
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Détecter les visages dans l'image en niveaux de gris
rects = detecteur(gris, 1)

# Boucle sur les détections de visage
for (i, rect) in enumerate(rects):
    # Déterminer les repères faciaux pour la région du visage, puis
    # convertir les coordonnées (x, y) du point de repère en un tableau NumPy
    forme = predicteur(gris, rect)
    forme = face_utils.shape_to_np(forme)

    # Boucle sur les parties du visage individuellement
    for (nom, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        # Cloner l'image originale pour pouvoir dessiner dessus, puis
        # afficher le nom de la partie du visage sur l'image
        clone = image.copy()
        cv2.putText(clone, nom, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        # Boucle sur le sous-ensemble de repères faciaux, en dessinant la
        # partie spécifique du visage
        for (x, y) in forme[i:j]:
            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

        # Extraire le ROI de la région du visage sous forme d'image séparée
        (x, y, w, h) = cv2.boundingRect(np.array([forme[i:j]]))
        roi = image[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

        # Montrer la partie particulière du visage
        cv2.imshow("ROI", roi)
        cv2.imshow("Image", clone)
        cv2.waitKey(0)

    # Visualiser tous les repères du visage avec une superposition transparente
    sortie = face_utils.visualize_facial_landmarks(image, forme)
    cv2.imshow("Image", sortie)
    cv2.waitKey(0)
