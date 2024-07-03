import os
from ultralytics import YOLO
import cv2
from util import *
from main_face_detection import Face_Detection
from OCR import *


#modello per rilevare i veicoli
model_vehicle = YOLO('yolov8n.pt')

#modello per rilevare le targhe dei veicoli
model_license_plate = YOLO('license_plate_detector.pt')


def process_image(img):
    
    #percorso dell'immagine
    img_path = os.path.basename(img)

    # Carica l'immagine
    img = cv2.imread(img)
    
    #crea una copia dell'immagine per disegnare i bounding-box
    img_bbox = img.copy()
    
    #UTILIZZO MODELLO PER RILEVAMENTO VEICOLI
    results_vehicle = model_vehicle(img)

    # variabile che indica i veicoli da dover identificare (2=auto , 3=moto , 5=bus , 7=camion) 
    vehicles = [2, 3, 5, 7]  
    #variabile che contiene le informazioni dei veicoli rilevati: [xmin,ymin,xmax,ymax,score,class_id]
    info_veicoli = []
    #id univoco veicolo
    id_vehicle = 0
    # Controlla se sono stati rilevati veicoli
    veicoli_rilevati = False

    # Rimuovi l'estensione attuale
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    
    base_directory = 'result_main_predict'
    # Aggiungi .txt come nuova estensione
    output_path = os.path.join(base_directory,base_name + '.txt')

    #percorso per la cartella contenente le immagini con i bbox rilevati disegnati
    output_img_bbox = os.path.join('output_immagini', 'image_bbox_detection')
    os.makedirs(output_img_bbox, exist_ok=True)

    # Estrai i bounding box dei veicoli, lo score e la class_id e aggiungili a info_veicoli
    for result in results_vehicle:
        for box in result.boxes:

            x_min_vehicle, y_min_vehicle, x_max_vehicle, y_max_vehicle = map(int, box.xyxy[0]) #coordinate bounding-box

            score = box.conf[0].item()  #score

            class_id = box.cls[0].item()  #class_id 

            if int(class_id) in vehicles:  # Classi YOLOv8 per auto, moto, bus, camion: class_id 
                
                veicoli_rilevati = True

                id_vehicle+=1 #incrementa id_vehicle univoco

                # Aggiungi le informazioni del veicolo a info_veicoli
                info_veicoli.append([id_vehicle,x_min_vehicle, y_min_vehicle, x_max_vehicle, y_max_vehicle, score, class_id])
                print(f"Veicolo rilevato: id_veicolo:{id_vehicle} bbox veicolo: x_min={x_min_vehicle}, y_min={y_min_vehicle}, x_max={x_max_vehicle}, y_max={y_max_vehicle}, score={score}, class_id={class_id}")


                #DISEGNA IL RETTANGOLO DEL BOUNDING-BOX ATTORNO AL VEICOLO 
                img_bbox = disegna_bounding_box(img_bbox,x_min_vehicle, y_min_vehicle, x_max_vehicle, y_max_vehicle,color=(0,255,0),thickness=2)
                 
    
    #salvataggio dell'immagine con i bounding-box disegnati
    path_img_bbox= os.path.join(output_img_bbox, f'bbox_{img_path}')
    cv2.imwrite(path_img_bbox,img_bbox)           


    #SALVATAGGIO DEI RISULTATI DELLA PREDICT DEL PRIMO MODELLO 
    #result.save_txt(output_path, save_conf=True)

    # Se sono stati rilevati veicoli, salva le informazioni in un file di testo
    if veicoli_rilevati:
        with open(output_path, 'w') as f:
            for info in info_veicoli:
                id_vehicle, x_min_vehicle, y_min_vehicle, x_max_vehicle, y_max_vehicle, score, class_id = info
                # Scrivi le informazioni nel formato YOLO: <class_id> <x_center> <y_center> <width> <height> <score>
                x_center = (x_min_vehicle + x_max_vehicle) / 2 / img.shape[1]
                y_center = (y_min_vehicle + y_max_vehicle) / 2 / img.shape[0]
                width = (x_max_vehicle - x_min_vehicle) / img.shape[1]
                height = (y_max_vehicle - y_min_vehicle) / img.shape[0]
                f.write(f"{class_id} {x_center} {y_center} {width} {height} {score}\n")
        print(f"File salvato in: {output_path}")
    else:
        print("Nessun veicolo rilevato, nessun file creato.")


    # Stampa le informazioni sui veicoli rilevati
    print("Informazioni sui veicoli rilevati:", info_veicoli)




    #CONTERRA' LE INFORMAZIONI DEI VEICOLI CON LE TARGHE RILEVATE: info_finali = [car_info, license_plate_INFO] 
    info_finali = []

    #CONTERRA' LE INFORMAZIONI DI OGNI TARGA RILEVATA: 
    license_plate_INFO=[]

    #CONTERRA' L'IMMAGINE CON LE TARGHE CENSURATE
    blurred_targhe_image = None


    #UTILIZZO MODELLO PER RILEVAMENTO TARGHE
    results_plate = model_license_plate(img)

    i=0
    # Estrai i bounding box delle targhe e leggi il testo della targa: per ogni targa rilevata
    for result in results_plate:
        for license_plate in result.boxes:
            x_min_targa, y_min_targa, x_max_targa, y_max_targa = map(int, license_plate.xyxy[0])
            score_targa = license_plate.conf[0].item()  # score
                
            #Stampa le informazioni sulle targhe rilevati
            print(f"Informazioni su targhe rilevate: bbox veicolo: x_min={x_min_targa}, y_min={y_min_targa}, x_max={x_max_targa}, y_max={y_max_targa}, score={score_targa}")

            # Passa i bounding box della targa e le info dei veicoli alla funzione get_car. 
            # car_info: CONTERRA' LE INFORMAZIONI DEI VEICOLI SU CUI SONO STATE RILEVATE LE TARGHE
            car_info = get_car([x_min_targa, y_min_targa, x_max_targa, y_max_targa, score_targa], info_veicoli)
            
            if car_info != (-1, -1, -1, -1, -1, -1, -1):
                car_id, xcar1, ycar1, xcar2, ycar2,score, class_id = car_info
                print(f"Targa trovata nel veicolo ID: {car_id}, bbox veicolo: x_min={xcar1}, y_min={ycar1}, x_max={xcar2}, y_max={ycar2}, score={score}, class_id={class_id}")

                # Crop license plate: ora possiamo ritagliare il box della targa
                license_plate_crop = img[y_min_targa:y_max_targa, x_min_targa:x_max_targa]

                #DISEGNA IL RETTANGOLO DEL BOUNDING-BOX ATTORNO ALLA TARGA 
                img_bbox = disegna_bounding_box(img_bbox,x_min_targa, y_min_targa, x_max_targa, y_max_targa,color=(0,0,255),thickness=2)
                
                i+=1
                #DOWNLOAD IMMAGINE cropped
                output_dir = os.path.join('output_immagini', 'targhe_cropped')
                os.makedirs(output_dir, exist_ok=True)
                path_img_cropped= os.path.join(output_dir, f'cropped_{i}{img_path}')

                cv2.imwrite(path_img_cropped,license_plate_crop)
                license_plate_text = OCR(path_img_cropped)
                
                #os.remove(path_img_cropped)

                # METODO PER LEGGERE IL NUMERO DI TARGA: in util.py
                #license_plate_text, license_plate_text_score = read_license_plate2(license_plate_crop)

                # Mappa delle classi alfanumeriche
                alphanumeric_mapping = {
                    '0': 999, '1': 1, '2': 222, '3': 333, '4': 4,
                    '5': 555, '6': 6, '7': 777, '8': 8, '9': 9,
                    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14,
                    'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
                    'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24,
                    'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
                    'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34,
                    'Z': 35
                }
            
                # Scrivi le predizioni nel file di testo
                with open(output_path, 'a') as f:
                    for char in license_plate_text:
                        if char in alphanumeric_mapping:
                            f.write(f"{alphanumeric_mapping[char]}\n")
                        else:
                            print('non è una cifra alfanumerica')
                            

                if license_plate_text == '':
                    license_plate_text = 'Testo targa non disponibile'

                #infromazioni della targa
                license_plate_INFO = [x_min_targa, y_min_targa, x_max_targa, y_max_targa,score_targa,license_plate_text]

                #INSERISCE NELLA LISTA INFO_FINALI: TUTTE LE INFORMAZIONI DELL'AUTO E IL VALORE DELLA TARGA SE LO TROVA
                info_finali.append([car_info,license_plate_INFO])

                # Stampa le informazioni finali
                for car_info, license_plate_text in info_finali:
                    car_id, xcar1, ycar1, xcar2, ycar2 ,score, class_id= car_info
                    print(f"INFOMAZIONI FINALI: Veicolo ID: {car_id}, bbox veicolo: x_min={xcar1}, y_min={ycar1}, x_max={xcar2}, y_max={ycar2}, score={score}, class_id={class_id}, Targa: {license_plate_text}")
                
                #APPLICAZIONE BLURRED SOLO ALLE TARGHE DEI VEICOLI RILEVATI
                blurred_targhe_image = BlurredTarga(img,x_min_targa,y_min_targa,x_max_targa,y_max_targa)
            

            else:
                print("Nessun veicolo trovato per questa targa")
                
                # Crop license plate: ora possiamo ritagliare il box della targa
                license_plate_crop = img[y_min_targa:y_max_targa, x_min_targa:x_max_targa]

                #DISEGNA IL RETTANGOLO DEL BOUNDING-BOX ATTORNO ALLA TARGA 
                img_bbox = disegna_bounding_box(img_bbox,x_min_targa, y_min_targa, x_max_targa, y_max_targa,color=(0,0,255),thickness=2)

                i+=1
                #DOWNLOAD IMMAGINE cropped
                output_dir = os.path.join('output_immagini', 'targhe_cropped')
                os.makedirs(output_dir, exist_ok=True)
                path_img_cropped= os.path.join(output_dir, f'cropped_{i}{img_path}')
                cv2.imwrite(path_img_cropped,license_plate_crop)
                license_plate_text = OCR(path_img_cropped)
                
                


                # METODO PER LEGGERE IL NUMERO DI TARGA: in util.py
                #license_plate_text, license_plate_text_score = read_license_plate2(license_plate_crop)

                # Mappa delle classi alfanumeriche
                alphanumeric_mapping = {
                    '0': 999, '1': 1, '2': 222, '3': 333, '4': 4,
                    '5': 555, '6': 6, '7': 777, '8': 8, '9': 9,
                    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14,
                    'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
                    'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24,
                    'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
                    'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34,
                    'Z': 35
                }
            
                #SCRIVI LE CLASSI DEI CARATTERI DELLA TARGA NEL FILE.TXT
                with open(output_path, 'a') as f:
                    for char in license_plate_text:
                        if char in alphanumeric_mapping:
                            f.write(f"{alphanumeric_mapping[char]}\n")
                        else:
                            print('non è una cifra alfanumerica')


                if license_plate_text == '':
                    license_plate_text = 'Testo targa non disponibile'

                #infromazioni della targa
                license_plate_INFO = [x_min_targa, y_min_targa, x_max_targa, y_max_targa,score_targa,license_plate_text]

                #INSERISCE NELLA LISTA INFO_FINALI: TUTTE LE INFORMAZIONI DELL'AUTO E IL VALORE DELLA TARGA SE LO TROVA
                info_finali.append([car_info,license_plate_INFO])


                # Stampa le informazioni finali
                for car_info, license_plate_text in info_finali:
                    car_id, xcar1, ycar1, xcar2, ycar2 ,score, class_id= car_info
                    print(f"INFOMAZIONI FINALI: Veicolo ID: {car_id}, bbox veicolo: x_min={xcar1}, y_min={ycar1}, x_max={xcar2}, y_max={ycar2}, score={score}, class_id={class_id}, Targa: {license_plate_text}")

                #APPLICAZIONE BLURRED SOLO ALLE TARGHE DEI VEICOLI RILEVATI
                blurred_targhe_image = BlurredTarga(img,x_min_targa,y_min_targa,x_max_targa,y_max_targa)
            
    
    #salvataggio dell'immagine con i bounding-box disegnati
    path_img_bbox= os.path.join(output_img_bbox, f'bbox_{img_path}')
    cv2.imwrite(path_img_bbox,img_bbox) 
    

    #SALVATAGGIO TARGHE DETECTION INFO
    output_path = os.path.join(base_directory, base_name + '.txt')
    #La funzione save_txt di un oggetto result di YOLOv8 salva i risultati delle predizioni in un file di testo
    result.save_txt(output_path, save_conf=True)




    #FACE DETECTION: CENSURA VOLTI NELL'IMMAGINE
    #se sono state censurate targhe, passa al face_detection l'immagine con le targhe censurate ed il percorso dell'immagine per salvare i risultati della predict
    if blurred_targhe_image is not None:   
        image_final, faces_info = Face_Detection(blurred_targhe_image, img_path)

    #altrimenti passa al face_detection l'immagine principale
    else:                                   
        image_final, faces_info = Face_Detection(img, img_path)


    if faces_info:
        for face_id, left, top, right, bottom, score_face in faces_info:
            img_bbox = disegna_bounding_box(img_bbox, left, top, right, bottom, color=(255, 0, 0), thickness=2)

        path_img_bbox = os.path.join(output_img_bbox, f'bbox_{img_path}')
        cv2.imwrite(path_img_bbox, img_bbox)


    #SALVATAGGIO INFORMAZIONI VEICOLI CSV
    cartella_veicoli = os.path.join('final_info_csv',base_name )
    output_path_csv_veicoli = os.path.join(cartella_veicoli, 'info_veicoli&targhe_'+base_name+'_.csv')
    salva_info_veicoli_csv(info_finali,output_path_csv_veicoli,'output_immagini',img_path, image_final)


    #SALVATAGGIO INFORMAZIONI FACES CSV
    cartella_faces = os.path.join('final_info_csv',base_name )
    output_path_csv_faces = os.path.join(cartella_faces, 'info_faces_'+base_name+'_.csv') 
    salva_info_FACES_csv(faces_info,output_path_csv_faces,'output_immagini',img_path,image_final)



    # Mostra l'immagine risultante
    #cv2.imshow('Blurred Image Final', image_final)
    #cv2.waitKey(2000)
    #cv2.destroyAllWindows()
    #exit()




#process_image('foto8.png')


