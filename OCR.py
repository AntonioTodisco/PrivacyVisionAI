import os
from inference_sdk import InferenceHTTPClient
import cv2



# Inizializza il client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="FQID0OG2nOkDbvOTV5UC"
)


def OCR(img_cropped):
    
   # Ottieni il percorso assoluto del file
    absolute_path = os.path.abspath(img_cropped)

    # Assicurati che il file esista
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"Il file {absolute_path} non esiste.")

    # Esegui l'inferenza
    result = CLIENT.infer(absolute_path, model_id="alphanumeric-character-detection/1")

    # Stampa i risultati dell'inferenza
    #print("---RISULTATI DEL OCR---")
    #print(result)

    # Estrai le predizioni
    predictions = result['predictions']

    # Ordina le predizioni per la coordinata x
    sorted_predictions = sorted(predictions, key=lambda p: p['x'])

    # Recupera il testo della targa
    license_plate_text = ''.join([p['class'] for p in sorted_predictions])

    #Lista di confidence di ogni carattere della targa
    #confidences = [p['confidence'] for p in sorted_predictions]

    # Stampa il testo della targa e le confidence
    print("Testo della targa:", license_plate_text)
    #print("Confidence:", confidences)

    return license_plate_text



