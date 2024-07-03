import os
from util import read_license_plate2
from OCR import OCR
from compute_IOU import *
from scipy.spatial.distance import hamming


def test_ocr():
    print('OCR TEST\n')

    distance_values = []
    hamming_distance_values = []

    # Lista dei file con estensione .txt nella directory result_predict
    predict_files = [file for file in os.listdir('result_predict_ocr') if file.lower().endswith('.txt')]
    label_files = [file for file in os.listdir('labels_ocr') if file.lower().endswith('.txt')]
    
    # Verifica che ci siano file predetti e etichettati
    if not predict_files:
        print("Nessun file predetto trovato nella directory 'result_predict_ocr'.")
        exit()

    if not label_files:
        print("Nessun file etichettato trovato nella directory 'labels_ocr'.")
        exit()

    # Itera sui file predetti e relativi file etichettati
    for predict_file in predict_files:
        for label_file in label_files:
            #se il nome del predict_file è uguale al nome del label_file, procede con il processo per il calcolo dell'IOU
            if predict_file == label_file:

                print(f'PREDICT_FILE: {predict_file}')
                print(f'LABEL_FILE: {label_file}')

                # Leggi le classi dei caratteri dai file
                class_list_predict = read_class_char(os.path.join('result_predict_ocr', predict_file))
                class_list_labels = read_class_char(os.path.join('labels_ocr', label_file))
                
                distance = damerau_levenshtein_distance(class_list_predict, class_list_labels)
                # Calcola la distanza di Damerau-Levenshtein tra interi
                print(f'Distanza di Damerau-Levenshtein: {distance}\n')
                # Aggiungi la distanza alla lista delle distanze
                distance_values.append(distance)
                
                
                # Trova la lunghezza massima tra le due stringhe
                max_length = max(len(class_list_predict), len(class_list_labels))

                # Allinea le due stringhe aggiungendo zeri alla fine della più corta
                # Aggiungi zeri alla fine della lista più corta
                padded_list1 = class_list_predict + [0] * (max_length - len(class_list_predict))
                padded_list2 = class_list_labels + [0] * (max_length - len(class_list_labels))
                # Calcola la distanza di Hamming
                hamming_distance = hamming(list(padded_list1), list(padded_list2))
                print("Distanza di Hamming:", hamming_distance)
                hamming_distance_values.append(hamming_distance)




    # Calcola la media delle distanze
    if distance_values:
        mean_distance = sum(distance_values) / len(distance_values)
    else:
        mean_distance = 0
    
        # Calcola la media delle distanze
    if hamming_distance_values:
        mean_distance_HAMMING = sum(hamming_distance_values) / len(hamming_distance_values)
    else:
        mean_distance_HAMMING = 0

    print("--- RESULT SET ---")
    print(f'Media della distanza di Damerau-Levenshtein: {mean_distance}')
    print(f'Media della distanza di HAMMING: {mean_distance_HAMMING}')
                




#CALCOLA LA DISTANZA TRA DUE STRINGHE:
# Questa distanza è una misura di similarità tra due stringhe, definita come il numero minimo di operazioni necessarie per trasformare una stringa nell'altra. 
# Le operazioni consentite includono:
# Inserimento di un carattere.
# Cancellazione di un carattere.
# Sostituzione di un carattere.
# Trasposizione di due caratteri adiacenti.      

# VALORE DI RITORNO:

# Il VALORE MINIMO della distanza di Damerau-Levenshtein è 0. 
# Questo accade quando le due stringhe sono identiche, cioè non sono necessarie operazioni per trasformare una stringa nell'altra.  
#  
# Il VALORE MASSIMO della distanza di Damerau-Levenshtein tra due stringhe è la lunghezza della stringa più lunga tra le due. 
# Questo accade quando ogni carattere della stringa più lunga deve essere cambiato o rimosso, oppure ogni carattere della stringa più corta deve essere aggiunto.     
def damerau_levenshtein_distance(arr1, arr2):
    d = {}
    lenarr1 = len(arr1)
    lenarr2 = len(arr2)
    
    for i in range(-1, lenarr1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, lenarr2 + 1):
        d[(-1, j)] = j + 1

    for i in range(lenarr1):
        for j in range(lenarr2):
            cost = 0 if arr1[i] == arr2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,  # deletion
                d[(i, j - 1)] + 1,  # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            if i > 0 and j > 0 and arr1[i] == arr2[j - 1] and arr1[i - 1] == arr2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition

    return d[lenarr1 - 1, lenarr2 - 1]





# Itera su tutte le immagini nella directory del testset
def iteration_testset():
    if not os.path.exists('result_predict_ocr'):
        os.makedirs('result_predict_ocr')

    i = 0
    for img_name in os.listdir('testset_ocr'):
        i += 1
        print(f"Immagine: {i}")
        img_path = os.path.join('testset_ocr', img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Nell'if {i}-> Elaborazione immagine: {img_path}")
        
            # OCR
            #license_plate_text, confidences = read_license_plate2(img_path)
            license_plate_text, confidences = OCR(img_path)
            
            print(f'--testo targa: {license_plate_text}')
            # Salva le predizioni in un file di testo
            base_name = os.path.splitext(img_name)[0]
            txt_path = os.path.join('result_predict_ocr', base_name + '.txt')
            
            # Mappa delle classi alfanumeriche
            alphanumeric_mapping = {
                '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14,
                'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
                'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24,
                'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
                'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34,
                'Z': 35
            }
            
            # Scrivi le predizioni nel file di testo
            with open(txt_path, 'w') as f:
                for char in license_plate_text:
                    if char in alphanumeric_mapping:
                        f.write(f"{alphanumeric_mapping[char]}\n")
                    else:
                        print('non è una cifra alfanumerica')

        else:
            print(f"Immagine: {i} non è un'estensione png, jpg, jpeg\n")





#ITERA SU OGNI IMMAGINE DEL TESTSET
#iteration_testset()

#ESEGUE IL TEST DOPO LA PREDICT SU TUTTE LE IMMAGINI 
test_ocr()