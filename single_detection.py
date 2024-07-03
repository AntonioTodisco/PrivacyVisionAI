import os
import cv2
from ultralytics import YOLO
from util import *


# Modello per rilevare i veicoli
model_vehicle = YOLO('yolov8n.pt')

# Modello per rilevare le targhe dei veicoli
model_license_plate = YOLO('license_plate_detector.pt')

# Modello per rilevare i volti
model = YOLO('model.pt')


def veicolo_detection(img_path):

    img = cv2.imread(img_path)

    if img is None:
        print(f"Errore nel caricamento dell'immagine: {img_path}")
        return None, None, None

    results_vehicle = model_vehicle(img)
    vehicles = [2, 3, 5, 7]
    info_veicoli = []
    id_vehicle = 0
    veicoli_rilevati = False

    #output_path = os.path.join('result_predict', os.path.basename(img_path) + '_.txt')
    output_path = modifica_estensione(img_path)
    
    for result in results_vehicle:
        for box in result.boxes:
            x_min_vehicle, y_min_vehicle, x_max_vehicle, y_max_vehicle = map(int, box.xyxy[0])
            score = box.conf[0].item()
            class_id = box.cls[0].item()

            if int(class_id) in vehicles:
                veicoli_rilevati = True
                id_vehicle += 1
                info_veicoli.append([id_vehicle, x_min_vehicle, y_min_vehicle, x_max_vehicle, y_max_vehicle, score, class_id])
                #print(f"Veicolo rilevato: id_veicolo:{id_vehicle} bbox veicolo: x_min={x_min_vehicle}, y_min={y_min_vehicle}, x_max={x_max_vehicle}, y_max={y_max_vehicle}, score={score}, class_id={class_id}")

    if veicoli_rilevati:
        with open(output_path, 'w') as f:
            for info in info_veicoli:
                id_vehicle, x_min_vehicle, y_min_vehicle, x_max_vehicle, y_max_vehicle, score, class_id = info
                x_center = (x_min_vehicle + x_max_vehicle) / 2 / img.shape[1]
                y_center = (y_min_vehicle + y_max_vehicle) / 2 / img.shape[0]
                width = (x_max_vehicle - x_min_vehicle) / img.shape[1]
                height = (y_max_vehicle - y_min_vehicle) / img.shape[0]
                f.write(f"{class_id} {x_center} {y_center} {width} {height} {score}\n")
        #print(f"File salvato in: {output_path}")
    else:
       
        print("Nessun veicolo rilevato, NESSUN BOUNDING-BOX SALVATO.")
        return img, [], False
        

    return img, info_veicoli, veicoli_rilevati




def targa_detection2(img_path):
    info_finali = []
    blurred_targhe_image = None

    targa_rilevata = False

    img = cv2.imread(img_path)
    results_plate = model_license_plate(img)
    
    for result in results_plate:
        for license_plate in result.boxes:
            x_min_targa, y_min_targa, x_max_targa, y_max_targa = map(int, license_plate.xyxy[0])
            score_targa = license_plate.conf[0].item()

            print(f"Informazioni su targhe rilevate: bbox veicolo: x_min={x_min_targa}, y_min={y_min_targa}, x_max={x_max_targa}, y_max={y_max_targa}, score={score_targa}")

            license_plate_crop = img[y_min_targa:y_max_targa, x_min_targa:x_max_targa]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

            license_plate_text, license_plate_text_score = read_license_plate2(license_plate_crop_gray)

            if license_plate_text == '':
                license_plate_text = 'Testo targa non disponibile'


            license_plate_INFO = [x_min_targa, y_min_targa, x_max_targa, y_max_targa, score_targa, license_plate_text]
            info_finali.append(license_plate_INFO)

            blurred_targhe_image = BlurredTarga(img, x_min_targa, y_min_targa, x_max_targa, y_max_targa)

            targa_rilevata = True


    # SALVATAGGIO RISULTATI PREDICT TARGHE
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    # Aggiungi .txt come nuova estensione
    output_path = os.path.join('result_predict_targhe', base_name + '.txt')
    
    # Salvare i risultati delle predizioni in un file di testo
    if info_finali:
        with open(output_path, 'w') as f:
            for info in info_finali:
                x_min_targa, y_min_targa, x_max_targa, y_max_targa, score_targa, license_plate_text = info
                x_center = (x_min_targa + x_max_targa) / 2 / img.shape[1]
                y_center = (y_min_targa + y_max_targa) / 2 / img.shape[0]
                width = (x_max_targa - x_min_targa) / img.shape[1]
                height = (y_max_targa - y_min_targa) / img.shape[0]
                f.write(f"0 {x_center} {y_center} {width} {height}\n")
        print(f"File salvato in: {output_path}")
    else:
        #file_vuoto_path = os.path.join('result_predict_targhe', 'NO_TARGHE_PREDICT_FILE_VUOTO' + os.path.basename(img_path) + '_.txt')
        #with open(file_vuoto_path, 'w') as f:
           # f.write(f"NESSUN TARGA RILEVATA\n")
        print("Nessuna targa rilevato, NESSUN BOUNDING-BOX SALVATO.")
        return  [], blurred_targhe_image, False

    return info_finali, blurred_targhe_image, targa_rilevata





def Face_Detection2(img_path):

    # Carica l'immagine utilizzando OpenCV
    image = cv2.imread(img_path) 

    #IDENTIFICATORE UNIVOCO DI UN VOLTO
    face_id = 0

    #CONTERRA UNA LISTA CON I VALORI DEI BOUNDING BOX RILEVATI E IL FACE_ID
    faces = []

    # Rileva i volti nell'immagine
    output = model(image)
    
    volto_rilevato = False

    for result in output:
        # Applica la sfocatura ai bounding box rilevati
        for bbox in result.boxes:
            left, top, right, bottom = map(int, bbox.xyxy[0])  # Coordinate bounding box
            
            score_face = bbox.conf[0].item()  #score

            face_id += 1

            volto_rilevato = True
            #print(f"Bounding box Face: face_id={face_id}, left={left}, top={top}, right={right}, bottom={bottom}, score={score_face}")

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
    
    
    # SALVATAGGIO RISULTATI PREDICT FACES
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    # Aggiungi .txt come nuova estensione
    output_path = os.path.join('result_predict_faces', base_name + '.txt')
    
    # Salvare i risultati delle predizioni in un file di testo
    if faces:
        with open(output_path, 'w') as f:

            for info in faces:
                face_id,left, top, right, bottom,score_face = info 

                x_center = (left + right) / 2 / image.shape[1]
                y_center = (top + bottom) / 2 / image.shape[0]
                width = (right - left) / image.shape[1]
                height = (bottom - top) / image.shape[0]

                f.write(f"0 {x_center} {y_center} {width} {height}\n")
        #print(f"File salvato in: {output_path}")
    else:
        #se non sono stati rilevati volti return false
        print("Nessuna volto rilevato, NESSUN BOUNDING-BOX SALVATO.")
        return image, [], False

       
    return image, faces, volto_rilevato