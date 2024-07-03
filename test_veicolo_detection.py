import os
from single_detection import veicolo_detection
from compute_IOU import *



def test_veicolo_detection():
    print('VEHICLES TEST\n')

    correct_classifications = 0
    incorrect_classifications = 0

    iou_values = []

    # Lista dei file con estensione .txt nella directory result_predict
    predict_files = [file for file in os.listdir('result_predict') if file.lower().endswith('.txt')]
    label_files = [file for file in os.listdir('labels_veicoli') if file.lower().endswith('.txt')]
    print(f'PREDICT_FILES: {predict_files}')
    print(f'LABEL_FILES: {label_files}')
    # Verifica che ci siano file predetti e etichettati
    if not predict_files:
        print("Nessun file predetto trovato nella directory 'result_predict'.")
        exit()

    if not label_files:
        print("Nessun file etichettato trovato nella directory 'labels'.")
        exit()


    # Itera sui file predetti e relativi file etichettati
    for predict_file in predict_files:
  
        for label_file in label_files:
            #se il nome del predict_file è uguale al nome del label_file, procede con il processo per il calcolo dell'IOU
            if predict_file == label_file:

                print(f'PREDICT_FILE: {predict_file}')
                print(f'LABEL_FILE: {label_file}')
               
                # Leggi i bounding box dai file
                box_predict, class_id_predict = read_bounding_boxes(os.path.join('result_predict', predict_file))
                box_labels, class_id_labels = read_bounding_boxes(os.path.join('labels_veicoli', label_file))

                # Ordina i bounding box in base alla coordinata 'left', ORDINAMENTO CRESCENTE
                box_predict = sorted(box_predict, key=lambda x: x['left'])
                box_labels = sorted(box_labels, key=lambda x: x['left'])

                #cancella i file che hanno class_id=0
                if class_id_labels == 0:
                        print("DELETE CLASSE_ID=0\n")
                        os.remove('result_predict/'+predict_file)
                        os.remove('labels_veicoli/'+label_file)

                        # Rimuovi l'estensione attuale
                        base_name = os.path.splitext(os.path.basename(label_file))[0]
                        # Aggiungi .jpg come nuova estensione
                        new_path = os.path.join( base_name + '.jpg')
                        os.remove('testset_veicoli/'+new_path)

                        continue
                

                # Controllo che il numero di bounding box predetti e etichettati sia lo stesso
                if len(box_predict) != len(box_labels):
                    print(f"Il numero di bounding box predetti nel file '{predict_file}' non corrisponde al numero di bounding box etichettati nel file '{label_file}'.\n")
                    
                    print("DELETE FILE CHE HANNO DIFFERENTI BOUNDING-BOX")
                    os.remove('result_predict/'+predict_file)
                    os.remove('labels_veicoli/'+label_file)
                    # Rimuovi l'estensione attuale
                    base_name = os.path.splitext(os.path.basename(label_file))[0]
                    # Aggiungi .jpg come nuova estensione
                    new_path = os.path.join( base_name + '.jpg')
                    os.remove('testset_veicoli/'+new_path)

                    continue
                
                delete_file = False

                # Calcola l'IoU per ciascun bounding box
                for i in range(len(box_predict)):
                    #EFFETTUO IL CONTROLLO DEL CLASS_ID PER TESTARE LA CLASSIFICAZIONE DEL MODELLO
                    if (class_id_predict == class_id_labels) or (class_id_predict==5 and class_id_labels==1) or (class_id_predict==3 and class_id_labels==4) or (class_id_predict==7 and class_id_labels==2):
                        print("Classificazione OK ")
                        correct_classifications+=1
                    else:
                        print("Classificazione ERRATA ")
                        incorrect_classifications+=1

                    iou = compute_iou(box_predict[i], box_labels[i])
                    if iou < 0.77:
                        delete_file = True
                        break
                    else:
                        iou_values.append(iou)
                        print(f'IoU bbox{i+1}: {iou}\n')
                
                if delete_file:
                    print("-----DELETE FILE CHE HANNO -0.77------")
                    os.remove('result_predict/' + predict_file)
                    os.remove('labels_veicoli/' + label_file)
                    # Rimuovi l'estensione attuale
                    base_name = os.path.splitext(os.path.basename(label_file))[0]
                    # Aggiungi .jpg come nuova estensione
                    new_path = os.path.join('testset_veicoli', base_name + '.jpg')
                    os.remove(new_path)
                    if os.path.exists(new_path):
                        os.remove(new_path)

    # Calcola la percentuale di classificazioni corrette
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

    # Calcolo della media dell'IoU
    if len(iou_values) > 0:
        mean_iou = sum(iou_values) / len(iou_values)
    else:
        mean_iou = 0.0

    print(f'Mean IoU: {mean_iou}')          



# Itera su tutte le immagini nella directory del testset
def iteration_testset():
    i=0
    for img_name in os.listdir('testset_veicoli'): 
        i+=1
        print(f"Immagine: {i}")
        img_path = os.path.join('testset_veicoli', img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            #print(f"Nell'if {i}-> Elaborazione immagine: {img_path}")

            #VEICOLI DETECTION
            img, info_veicoli, veicoli_rilevati = veicolo_detection(img_path)

            if veicoli_rilevati == False:
                # Rimuovi l'estensione attuale
                base_name = os.path.splitext(os.path.basename(img_name))[0]

                #rimuove  file label
                labels_txt = os.path.join('labels_veicoli', base_name + '.txt') 
                if os.path.exists(labels_txt):
                    os.remove(labels_txt)

                #rimuove immagine
                if os.path.exists(img_path):
                    os.remove(img_path)

        else:
            print(f'immagine: {i} non è un estensione png, jpg, jpeg\n')

#ITERA NELLA CARTELLA DEL TESTSET
#iteration_testset()

#DOPO AVER FATTO LE PREDICT DEL TESTSET, PASSA AL TEST
test_veicolo_detection()



