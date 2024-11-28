# Reconnaissance-des-parties-du-visage
La reconnaissance faciale est importante pour la sécurité moderne. Au lieu d'utiliser des mots de passe, elle détecte les visages et les compare entre eux.
# importer les packages nécessaires
depuis  imutils  importer  face_utils
importer  numpy  en tant  que np
importer  argparse
importer  imutils
importer  dlib
importer  cv2

# construire l'analyseur d'arguments et analyser les arguments
ap  =  argparse .ArgumentParser ( )
ap . add_argument ( "-p" ,  "--shape-predictor" ,  requis = True ,
	aide = "chemin vers le prédicteur de repère facial" )
ap . add_argument ( "-i" ,  "--image" ,  requis = True ,
	aide = "chemin vers l'image d'entrée" )
args  =  vars ( ap . parse_args ())

# initialiser le détecteur de visage de dlib (basé sur HOG) puis créer
# le prédicteur de repère facial
détecteur  =  dlib . get_frontal_face_detector ()
prédicteur  =  dlib . shape_predictor ( args [ "shape_predictor" ])

# chargez l'image d'entrée, redimensionnez-la et convertissez-la en niveaux de gris
image  =  cv2 .imread ( args [ "image" ] )
image  =  imutils . resize ( image ,  largeur = 500 )
gris  =  cv2 . cvtColor ( image ,  cv2 . COLOR_BGR2GRAY )

# détecter les visages dans l'image en niveaux de gris
rects  =  détecteur ( gris ,  1 )

# boucle sur les détections de visage
pour  ( i ,  rect )  dans  enumerate ( rects ):
	# déterminer les repères faciaux pour la région du visage, puis
	# convertir les coordonnées (x, y) du point de repère en un tableau NumPy
	forme  =  prédicteur ( gris ,  rect )
	forme  =  face_utils . shape_to_np ( forme )

	# boucle sur les parties du visage individuellement
	pour  ( nom ,  ( i ,  j ))  dans  face_utils . FACIAL_LANDMARKS_IDXS . items ():
		# cloner l'image originale pour pouvoir dessiner dessus, puis
		# affiche le nom de la partie du visage sur l'image
		clone  =  image .copie ( )
		cv2 . putText ( clone ,  nom ,  ( 10 ,  30 ),  cv2 . FONT_HERSHEY_SIMPLEX ,
			0,7 ,  ( 0 ,  0 ,  255 ),  2 )

		# boucle sur le sous-ensemble de repères faciaux, en dessinant le
		# partie spécifique du visage
		pour  ( x ,  y )  de  forme [ i : j ] :
			cv2 . cercle ( clone ,  ( x ,  y ),  1 ,  ( 0 ,  0 ,  255 ),  - 1 )

		# extraire le ROI de la région du visage sous forme d'image séparée
		( x ,  y ,  l ,  h )  =  cv2 . boundingRect ( np . array ( [ forme [ i : j ]]))
		roi  =  image [ y : y  +  h ,  x : x  +  w ]
		roi  =  imutils . resize ( roi ,  largeur = 250 ,  inter = cv2 . INTER_CUBIC )

		# montre la partie particulière du visage
		cv2 . imshow ( "ROI" ,  roi )
		cv2 . imshow ( "Image" ,  clone )
		cv2 . waitKey ( 0 )

	# visualiser tous les repères du visage avec une superposition transparente
	sortie  =  face_utils . visualize_facial_landmarks ( image ,  forme )
	cv2 . imshow ( "Image" ,  sortie )
	cv2 . waitKey ( 0 )
