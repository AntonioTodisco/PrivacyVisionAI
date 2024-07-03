from compute_IOU import *
import os
from single_detection import targa_detection2



def test_targa_detection():
    print('TARGA DETECTION TEST\n')

    iou_values = []

    # Lista dei file con estensione .txt nella directory result_predict
    predict_files = [file for file in os.listdir('result_predict_targhe') if file.lower().endswith('.txt')]
    label_files = [file for file in os.listdir('labels_targhe') if file.lower().endswith('.txt')]
    
    # Verifica che ci siano file predetti e etichettati
    if not predict_files:
        print("Nessun file predetto trovato nella directory 'result_predict_targhe'.")
        exit()

    if not label_files:
        print("Nessun file etichettato trovato nella directory 'labels_targhe'.")
        exit()

    # Itera sui file predetti e relativi file etichettati
    for predict_file in predict_files:
        for label_file in label_files:
            #se il nome del predict_file è uguale al nome del label_file, procede con il processo per il calcolo dell'IOU
            if predict_file == label_file:

                print(f'PREDICT_FILE: {predict_file}')
                print(f'LABEL_FILE: {label_file}')

                # Leggi i bounding box dai file
                box_predict = read_bounding_boxes_TargheAndFaces(os.path.join('result_predict_targhe', predict_file))
                box_labels = read_bounding_boxes_TargheAndFaces(os.path.join('labels_targhe', label_file))
               
                # Ordina i bounding box in base alla coordinata 'left', ORDINAMENTO CRESCENTE
                box_predict = sorted(box_predict, key=lambda x: x['left'])
                box_labels = sorted(box_labels, key=lambda x: x['left'])
                
                # Controllo che il numero di bounding box predetti e etichettati sia lo stesso
                if len(box_predict) != len(box_labels):
                    print(f"Il numero di bounding box predetti nel file '{predict_file}' non corrisponde al numero di bounding box etichettati nel file '{label_file}'.\n")
                    
                    print("DELETE FILE CHE HANNO DIFFERENTI BOUNDING-BOX")
                    os.remove('result_predict_targhe/' + predict_file)
                    os.remove('labels_targhe/' + label_file)
                    # Rimuovi l'estensione attuale
                    base_name = os.path.splitext(os.path.basename(label_file))[0]
                    # Aggiungi .jpg come nuova estensione
                    new_path = os.path.join('testset_targhe', base_name + '.jpg')
                    os.remove(new_path)
                    continue

                # Calcola l'IoU per ciascun bounding box
                for i in range(len(box_predict)):
                    iou = compute_iou(box_predict[i], box_labels[i])
                    iou_values.append(iou)
                    print(f'IoU bbox{i+1}: {iou}\n')
    # Calcolo della media dell'IoU
    if len(iou_values) > 0:
        mean_iou = sum(iou_values) / len(iou_values)
    else:
        mean_iou = 0.0

    #MEDIA DEGLI IOU
    print('--- RESULT TEST ---')
    print(f'Mean IoU: {mean_iou}') 


# Itera su tutte le immagini nella directory del testset
def iteration_testset():
    i = 0
    for img_name in os.listdir('testset_targhe'):
        i += 1
        print(f"Immagine: {i}")
        img_path = os.path.join('testset_targhe', img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Nell'if {i}-> Elaborazione immagine: {img_path}")
        
            # TARGHE DETECTION
            info_finali, blurred_targhe_image, targa_rilevata = targa_detection2(img_path)

            if targa_rilevata==False:
                # Rimuovi l'estensione attuale
                base_name = os.path.splitext(os.path.basename(img_name))[0]

                #rimuove  file label
                labels_txt = os.path.join('labels_targhe', base_name + '.txt') 
                if os.path.exists(labels_txt):
                    os.remove(labels_txt)

                #rimuove immagine
                if os.path.exists(img_path):
                    os.remove(img_path)

        else:
            print(f'immagine: {i} non è un estensione png, jpg, jpeg\n')


#ITERA SU OGNI IMMAGINE DEL TESTSET
#iteration_testset()


# DOPO AVER FATTO LE PREDICT, PASSA AL TEST
test_targa_detection()

