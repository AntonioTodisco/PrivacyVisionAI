import os
import cv2
import numpy as np
from ultralytics import YOLO
from util import disegna_bounding_box


# Carica il modello
model = YOLO('model.pt')


def Face_Detection(image,img_path):

    #IDENTIFICATORE UNIVOCO DI UN VOLTO
    face_id = 0

    #CONTERRA UNA LISTA CON I VALORI DEI BOUNDING BOX RILEVATI E IL FACE_ID
    faces = []

    # Rileva i volti nell'immagine
    output = model(image)
    
    
    for result in output:
        # Applica la sfocatura ai bounding box rilevati
        for bbox in result.boxes:
            left, top, right, bottom = map(int, bbox.xyxy[0])  # Coordinate bounding box
            
            score_face = bbox.conf[0].item()  #score

            face_id += 1

            print(f"Bounding box Face: face_id={face_id}, left={left}, top={top}, right={right}, bottom={bottom}, score={score_face}")

            # Espandi il bounding box
            top = max(0, top - 20)
            right = min(image.shape[1], right + 20)
            bottom = min(image.shape[0], bottom + 20)
            left = max(0, left - 20)

            # Estrai il volto dall'immagine
            face_image = image[top:bottom, left:right]

            # Crea una maschera vuota della stessa dimensione del volto
            mask = np.zeros_like(face_image, dtype=np.uint8)

            # Disegna un'ellisse bianca sulla maschera
            center = (mask.shape[1] // 2, mask.shape[0] // 2)
            axes = (mask.shape[1] // 2, mask.shape[0] // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)

            # Applica la sfocatura all'intera regione del volto
            blurred_face = cv2.GaussianBlur(face_image, (151, 151), 70)

            # Combina la regione sfocata con la maschera
            face_with_blurred = np.where(mask == 255, blurred_face, face_image)

            # Sostituisci il volto originale con quello sfocato
            image[top:bottom, left:right] = face_with_blurred

            #inserisci informazioni volto rilevate
            faces.append([face_id,left, top, right, bottom,score_face])


    
    #SALVATAGGIO RISULTATI DELLA PREDICT DEL FACE DETECTION
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    #base_directory = os.path.join('result_main_predict', base_name)
    base_directory = 'result_main_predict'
    output_path = os.path.join(base_directory,base_name+'.txt')

    # Salvare i risultati delle predizioni in un file di testo
    if faces:
        with open(output_path, 'a') as f:

            for info in faces:
                face_id,left, top, right, bottom,score_face = info 

                x_center = (left + right) / 2 / image.shape[1]
                y_center = (top + bottom) / 2 / image.shape[0]
                width = (right - left) / image.shape[1]
                height = (bottom - top) / image.shape[0]

                f.write(f"100 {x_center} {y_center} {width} {height}\n")

    else:
        print("Nessuna volto rilevato, NESSUN BOUNDING-BOX SALVATO.")
        return image, []

   

    return image, faces