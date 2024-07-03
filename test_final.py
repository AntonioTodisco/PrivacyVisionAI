from main import *
from OCR import *
from compute_IOU import *
from test_ocr import damerau_levenshtein_distance
import os
from scipy.spatial.distance import hamming

def test_final():
    print('--- FINAL TEST ---')

    iou_values = []
    iou_values_faces = []
    iou_values_targhe = []
    distance_values = []
    distance_values_hamming = []

    correct_classifications = 0
    incorrect_classifications = 0


    predict_files = get_txt_files_in_directory('result_main_predict')
    label_files = get_txt_files_in_directory('labels_total')

    # Verifica che ci siano file predetti e etichettati
    if not predict_files:
        print("Nessun file predetto trovato nella directory 'result_main_predict'.")
        exit()

    if not label_files:
        print("Nessun file etichettato trovato nella directory 'labels_total'.")
        exit()

    # Itera sui file predetti e relativi file etichettati
    for predict_file in predict_files:
        for label_file in label_files:
            # Se il nome del predict_file è uguale al nome del label_file, procede con il processo per il calcolo dell'IOU
            if os.path.basename(predict_file) == os.path.basename(label_file):

                print(f'PREDICT_FILE: {predict_file}')
                print(f'LABEL_FILE: {label_file}')

                # Leggi i bounding box dai file
                box_predict = read_bounding_boxes_total(predict_file)
                box_labels = read_bounding_boxes_total(label_file)

                # Ordina i bounding box in base alla coordinata 'left', ORDINAMENTO CRESCENTE
                box_predict = sorted(box_predict, key=lambda x: x['left'])
                box_labels = sorted(box_labels, key=lambda x: x['left'])


                # Leggi le classi dei caratteri dai file
                class_list_predict = read_class_char(predict_file)
                class_list_labels = read_class_char(label_file)
                print(f'class_list_predict: {class_list_predict}')
                print(f'class_list_labels: {class_list_labels}')               


                # Filtra le classi per rimuovere 0, 2, 3, 5, 7, 100
                classes_to_remove = {0, 2, 3, 5, 7, 100}
                filtered_class_list_predict_OCR = [cls for cls in class_list_predict if cls not in classes_to_remove]
                filtered_class_list_labels_OCR = [cls for cls in class_list_labels if cls not in classes_to_remove]

                print(f'filtered_class_list_predict_OCR: {filtered_class_list_predict_OCR}')
                print(f'filtered_class_list_labels_OCR: {filtered_class_list_labels_OCR}')
                
                #se almeno una delle due liste delle classi dei caratteri esiste: calcola la distanza
                if len(filtered_class_list_predict_OCR) > 0 or len(filtered_class_list_labels_OCR) > 0:

                    # CALCOLO DISTANCA DI DAMEREU-LEVENSHTEIN
                    distance = damerau_levenshtein_distance(filtered_class_list_predict_OCR, filtered_class_list_labels_OCR)
                    print(f'Distanza di Damerau-Levenshtein: {distance}\n')
                    # Aggiungi la distanza alla lista delle distanze
                    distance_values.append(distance)

                    #CALCOLO DISTANZA DI HAMMING
                    # Trova la lunghezza massima tra le due stringhe
                    max_length = max(len(class_list_predict), len(class_list_labels))
                    # Allinea le due stringhe aggiungendo zeri alla fine della più corta
                    # Aggiungi zeri alla fine della lista più corta
                    padded_list1 = class_list_predict + [0] * (max_length - len(class_list_predict))
                    padded_list2 = class_list_labels + [0] * (max_length - len(class_list_labels))
                    # Calcola la distanza di Hamming
                    hamming_distance = hamming(list(padded_list1), list(padded_list2))
                    print("Distanza di Hamming:", hamming_distance)
                    distance_values_hamming.append(hamming_distance)

                
                # Mantieni solo le classi {0, 2, 3, 5, 7, 100}
                classes_of_interest = {0, 2, 3, 5, 7, 100}
                class_list_predict = [cls for cls in class_list_predict if cls in classes_of_interest]
                class_list_labels = [cls for cls in class_list_labels if cls in classes_of_interest]

                j=0
                k=0
                # Calcola l'IoU per ciascun bounding box
                for i in range(len(box_predict)):
                    
                    if class_list_predict[i]==2 or class_list_predict[i]==3 or class_list_predict[i]==5 or class_list_predict[i]==7:
                        # EFFETTUO IL CONTROLLO DEL CLASS_ID PER TESTARE LA CLASSIFICAZIONE DEL MODELLO
                        if class_list_predict[i] == class_list_labels[i]:
                            print("Classificazione OK ")
                            correct_classifications += 1
                        else:
                            print("Classificazione ERRATA ")
                            incorrect_classifications += 1

                        # CALCOLO INTERSECTION OVER UNION PER I BBOX DEI VEICOLI
                        iou = compute_iou(box_predict[i], box_labels[i])
                        iou_values.append(iou)
                        print(f'IoU bbox veicolo{i + 1}: {iou}\n')

                    elif class_list_predict[i] == 0:
                        j+=1
                        # CALCOLO INTERSECTION OVER UNION PER I BBOX DELLE TARGHE
                        iou_targhe = compute_iou(box_predict[i], box_labels[i])
                        iou_values_targhe.append(iou_targhe)
                        print(f'IoU bbox targa{j}: {iou_targhe}\n')

                    elif class_list_predict[i] == 100:
                        k+=1
                        # CALCOLO INTERSECTION OVER UNION PER I BBOX DEI VOLTI
                        iou_faces = compute_iou(box_predict[i], box_labels[i])
                        iou_values_faces.append(iou_faces)
                        print(f'IoU bbox volto{k}: {iou_faces}\n')
                    



    # CLASSIFICAZIONI CORRETTE
    total_classifications = correct_classifications + incorrect_classifications
    if total_classifications > 0:
        correct_percentage = (correct_classifications / total_classifications) * 100
    else:
        correct_percentage = 0.0

    print('--- RESULT TEST ---\n')
    print(f'Totale Classificazioni: {total_classifications}')
    print(f'Classificazioni Corrette: {correct_classifications}')
    print(f'Classificazioni Errate: {incorrect_classifications}')
    print(f'Percentuale Classificazioni Corrette: {correct_percentage:.2f}%\n')  

    # MEDIA DEGLI IOU veicoli
    if len(iou_values) > 0:
        mean_iou = sum(iou_values) / len(iou_values)
    else:
        mean_iou = 0.0
    print(f'Mean IoU veicoli: {mean_iou}') 

    # MEDIA DEGLI IOU targhe
    if len(iou_values_targhe) > 0:
        mean_iou_targhe = sum(iou_values_targhe) / len(iou_values_targhe)
    else:
        mean_iou_targhe = 0.0
    print(f'Mean IoU targhe: {mean_iou_targhe}') 

    # MEDIA DEGLI IOU volti
    if len(iou_values_faces) > 0:
        mean_iou_faces = sum(iou_values_faces) / len(iou_values_faces)
    else:
        mean_iou_faces = 0.0
    print(f'Mean IoU volti: {mean_iou_faces}') 

    # MEDIA DISTANZE OCR
    if distance_values:
        mean_distance = sum(distance_values) / len(distance_values)
    else:
        mean_distance = 0
    print(f'Media della distanza di Damerau-Levenshtein: {mean_distance}')


    # MEDIA DISTANZE OCR
    if distance_values_hamming:
        mean_distance_hamming = sum(distance_values_hamming) / len(distance_values_hamming)
    else:
        mean_distance_hamming = 0
    print(f'Media della distanza di Hamming: {mean_distance_hamming}')




#recupera tutti i file txt da una directory/sottodirectory
def get_txt_files_in_directory(directory):
    txt_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files



# Itera su tutte le immagini nella directory del testset
def iteration_testset():
    i=0
    for img_name in os.listdir('testset_total'): 
        i+=1
        print(f"Immagine: {i}")
        img_path = os.path.join('testset_total', img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):

            process_image(img_path)

        else:
            print(f'immagine: {i} non è un estensione png, jpg, jpeg\n')



#ITERA NELLA CARTELLA DEL TESTSET PER ESEGUIRE LA PREDICT
#iteration_testset()


#ESEGUE IL TEST SUI RISULTATI
test_final()
