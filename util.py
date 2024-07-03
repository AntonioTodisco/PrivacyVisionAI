import csv
import os
import string
import cv2
import easyocr
import numpy as np



# Initialize the OCR reader
reader = easyocr.Reader(['en','it'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

# Input: targa e info dei veicoli rilevati. Return: info del veicolo che ha quella targa
def get_car(license_plate, info_veicoli):

    # Estrai le coordinate della targa
    x_min_targa, y_min_targa, x_max_targa, y_max_targa,score_targa = license_plate

    for veicolo in info_veicoli:
        id_vehicle, x_min_vehicle, y_min_vehicle, x_max_vehicle, y_max_vehicle, score, class_id = veicolo

        # Controlla se la targa è all'interno del bounding box del veicolo
        if (x_min_targa >= x_min_vehicle and y_min_targa >= y_min_vehicle and
            x_max_targa <= x_max_vehicle and y_max_targa <= y_max_vehicle):
            return id_vehicle,x_min_vehicle, y_min_vehicle, x_max_vehicle, y_max_vehicle,score, class_id

    # Se non viene trovato nessun veicolo contenente la targa restitusce valori negativi
    return -1, -1, -1, -1, -1, -1, -1



#Funzione per applicare il blurred e mostrare l'immagine della targa
def BlurredTarga(img,x_min_targa,y_min_targa,x_max_targa,y_max_targa):
    
    # Disegna un rettangolo intorno al bounding box
    cv2.rectangle(img, (x_min_targa, y_min_targa), (x_max_targa, y_max_targa), (0, 255, 0), 2)

    # Estrai la regione della targa
    license_plate = img[y_min_targa:y_max_targa, x_min_targa:x_max_targa]

    # Applica la sfocatura al rettangolo
    blurred_rectangle = cv2.GaussianBlur(license_plate, (555, 555), 0)
                
    # Sovrapponi il rettangolo sfocato sull'immagine principale
    img[y_min_targa:y_max_targa, x_min_targa:x_max_targa] = blurred_rectangle

    return img


def ViewDownloadImageBlurred(img,img_path):
    # Mostra l'immagine risultante
    cv2.imshow('Blurred Image Final', img)
    cv2.waitKey(5000)

    var = input("Premi 'd' download immagine o 'q' per uscire: ")
    if var == 'q':
        cv2.destroyAllWindows()
        exit(0)
    elif var == 'd':
        # Salva l'immagine risultante
        result_path = "blurred_final_"+img_path
        cv2.imwrite(result_path, img)

    return None




def read_license_plate(license_plate_crop):
    # Usa EasyOCR per leggere il testo
    results = reader.readtext(license_plate_crop)
    text = ''
    score = ''


    # Prende solo il risultato con il punteggio di confidenza più alto
    best_result = max(results, key=lambda x: x[2])
    text = best_result[1].upper().replace(' ', '')
    text = text.replace('.', '').replace('-', '')   # Rimuove i caratteri . e -
    score = best_result[2]

    # Controlla se il testo è conforme al formato della targa
    if license_complies_format(text):
        return format_license(text), score
    else:
        return text, score



# Controlla se il testo della targa è del formato richiesto
# Input: testo della targa. Return: vero se è conforme al formato, falso altrimenti
def license_complies_format(text):
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False

# Formatta la targa secondo i dizionari definiti  
# Input: testo della targa. Return: targa formattata
def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int, 4: dict_char_to_int}
    
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_





def read_license_plate2(image):
    # Usa EasyOCR per leggere il testo
    results = reader.readtext(image)

    # Inizializza il testo della targa e il punteggio di confidenza
    license_plate_text = ""
    license_plate_text_score = 0.0
    
    # Se ci sono risultati
    if results:
        # Prendi il risultato con il punteggio di confidenza più alto
        best_result = max(results, key=lambda x: x[2])

        license_plate_text = best_result[1].upper().replace(' ', '')  # Converte il testo in maiuscolo e rimuove gli spazi
        license_plate_text = license_plate_text.replace('.', '').replace('-', '')   # Rimuove i caratteri . e -

        license_plate_text_score = best_result[2]
    
    return license_plate_text, license_plate_text_score




def salva_info_veicoli_csv(info_finali, nome_file_csv, directory_img, img_path, img):

    # Assicurati che la directory per il file CSV esista: se non esiste, creala
    directory_csv = os.path.dirname(nome_file_csv)
    if not os.path.exists(directory_csv):
        os.makedirs(directory_csv)

    # Assicurati che la directory delle immagini esista: se non esiste la directory, la crea
    if not os.path.exists(directory_img):
        os.makedirs(directory_img)
    
    #DOWNLOAD IMMAGINE OUTPUT: censurata
    output_img_censored = os.path.join(directory_img, 'image_censored')
    os.makedirs(output_img_censored, exist_ok=True)
    percorso_immagine = os.path.join(output_img_censored, f'censored_{img_path}')
    cv2.imwrite(percorso_immagine, img)

    # Definisci i campi del CSV
    campi = ['id_veicolo', 'bounding-box_veicolo', 'class_id_veicolo', 'score_veicolo', 'license_plate_text', 'bounding-box_license-plate', 'score_license_plate', 'percorso_immagine_censurata']
    
    # Apri il file CSV per scrittura
    with open(nome_file_csv, mode='w', newline='') as file_csv:
        writer = csv.writer(file_csv)
        
        # Scrivi i campi nel file CSV
        writer.writerow(campi)
        
        # Variabile per tracciare se l'immagine è stata salvata
        immagine_salvata = False
        
        
        # Scrivi le informazioni finali nel file CSV
        for car_info, license_plate in info_finali:
            car_id, xcar1, ycar1, xcar2, ycar2, score_veicolo, class_id = car_info
            x_min_targa, y_min_targa, x_max_targa, y_max_targa, score_targa, license_plate_text = license_plate
            
            # Se l'immagine non è stata ancora salvata, salva il percorso
            if not immagine_salvata:
                percorso = percorso_immagine
                immagine_salvata = True
            else:
                percorso = ""
                
            # Crea la riga con le informazioni del veicolo e della targa
            riga = [
                car_id, 
                f"{xcar1}, {ycar1}, {xcar2}, {ycar2}", 
                class_id, 
                score_veicolo, 
                license_plate_text, 
                f"{x_min_targa}, {y_min_targa}, {x_max_targa}, {y_max_targa}", 
                score_targa,
                percorso
            ]
            
            # Scrivi la riga nel file CSV
            writer.writerow(riga)




def salva_info_FACES_csv(faces_info, nome_file,directory_img, img_path, img):

    # Assicurati che la directory per il file CSV esista: se non esiste, creala
    directory_csv = os.path.dirname(nome_file)
    if not os.path.exists(directory_csv):
        os.makedirs(directory_csv)

    # Assicurati che la directory delle immagini esista: se non esiste la directory, la crea
    if not os.path.exists(directory_img):
        os.makedirs(directory_img)

    #DOWNLOAD IMMAGINE OUTPUT: censurata
    output_img_censored = os.path.join(directory_img, 'image_censored')
    os.makedirs(output_img_censored, exist_ok=True)
    percorso_immagine = os.path.join(output_img_censored, f'censored_{img_path}')
    cv2.imwrite(percorso_immagine, img)

    # Definisci i campi del CSV
    campi = ['face_id', 'left', 'top', 'right', 'bottom', 'score_face', 'percorso_immagine_censurata']
    
    # Apri il file CSV per scrittura
    with open(nome_file, mode='w', newline='') as file_csv:
        writer = csv.writer(file_csv)
        
        # Scrivi i campi nel file CSV
        writer.writerow(campi)
        
        # Variabile per tracciare se l'immagine è stata salvata
        immagine_salvata = False

        # Scrivi le informazioni finali nel file CSV
        for face_info in faces_info:
            face_id, left, top, right, bottom, score_face = face_info
            
            # Se l'immagine non è stata ancora salvata, salva il percorso
            if not immagine_salvata:
                percorso = percorso_immagine
                immagine_salvata = True
            else:
                percorso = ""

            # Crea la riga con le informazioni del volto
            riga = [face_id, left, top, right, bottom, score_face, percorso]
            
            # Scrivi la riga nel file CSV
            writer.writerow(riga)




def modifica_estensione(img_path: str) -> str:
    # Rimuovi l'estensione attuale
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    # Aggiungi .txt come nuova estensione
    new_path = os.path.join('result_predict', base_name + '.txt')
    return new_path



def raddrizza_targa(img_targa):
    # Converti l'immagine in scala di grigi
    gray = cv2.cvtColor(img_targa, cv2.COLOR_BGR2GRAY)
    
    # Applica un'operazione di soglia per binarizzare l'immagine
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Trova i contorni nell'immagine binarizzata
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Trova il contorno più grande
    largest_contour = min(contours, key=cv2.contourArea)
    
    # Trova il rettangolo con la minima area che contiene il contorno più grande
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Calcola l'angolo di rotazione
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    
    # Ottieni le dimensioni dell'immagine
    (h, w) = img_targa.shape[:2]
    center = (w // 2, h // 2)
    
    # Calcola la matrice di rotazione
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Applica la rotazione
    raddrizzata = cv2.warpAffine(img_targa, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return raddrizzata



def disegna_bounding_box(image, x_min, y_min, x_max, y_max, color, thickness):
    """
    Disegna un rettangolo attorno all'oggetto identificato nell'immagine.

    Args:
    - image (numpy array): L'immagine su cui disegnare il rettangolo.
    - x_min (int): La coordinata x del vertice in alto a sinistra del bounding box.
    - y_min (int): La coordinata y del vertice in alto a sinistra del bounding box.
    - x_max (int): La coordinata x del vertice in basso a destra del bounding box.
    - y_max (int): La coordinata y del vertice in basso a destra del bounding box.
    - color (tuple): Il colore del rettangolo in formato BGR.
    - thickness (int): Lo spessore del rettangolo.

    Returns:
    - image_with_box (numpy array): L'immagine con il rettangolo disegnato.
    """
    # Disegna il rettangolo sull'immagine
    image_with_box = cv2.rectangle(image.copy(), (x_min, y_min), (x_max, y_max), color, thickness)
    return image_with_box