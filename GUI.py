import gradio as gr
import main  # Assumi che main.py contenga la funzione process_image
import tempfile
import shutil
import os
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path


# Funzione per estrarre il testo della targa dal file CSV
def get_license_plate_text(image_name):
    csv_dir = os.path.join('final_info_csv', image_name)
    csv_filename = f"info_veicoli&targhe_{image_name}_.csv"
    csv_path = os.path.join(csv_dir, csv_filename)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        license_plate_text = "\n".join(df['license_plate_text'].dropna().astype(str).tolist())
        return license_plate_text

    return ""


# Funzione per processare l'immagine singola
def process_single_image(data):
    # Converti l'array numpy in un'immagine PIL
    input_image = Image.fromarray(data.astype('uint8'), 'RGB')

    # Salviamo temporaneamente l'immagine nel formato desiderato (JPEG, PNG, ecc.)
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "input_image.jpg")
    input_image.save(temp_path)

    # Chiamiamo la funzione definita in main.py per processare l'immagine
    main.process_image(temp_path)

    # Recuperiamo il nome dell'immagine processata basata sul nome dell'immagine temporanea
    output_image_folder = os.path.join('output_immagini', 'image_censored')
    temp_image_name = os.path.basename(temp_path)
    for filename in os.listdir(output_image_folder):
        if filename.startswith(f"censored_{temp_image_name}"):
            output_image_path = os.path.join(output_image_folder, filename)
            processed_image = Image.open(output_image_path)
            license_plate_text = get_license_plate_text(temp_image_name.split('.')[0])
            shutil.rmtree(temp_dir)  # Elimina la cartella temporanea
            return np.array(processed_image), license_plate_text  # Restituisce l'immagine come array numpy e il testo della targa

    # Se non trova l'immagine processata, elimina comunque la cartella temporanea
    shutil.rmtree(temp_dir)
    return None, ""


# Funzione per processare una cartella di immagini
def process_folder(filepaths):
    processed_images = []
    detected_texts = []

    # Estensioni accettate per le immagini
    allowed_extensions = {'.jpeg', '.jpg', '.png'}

    for input_image_path in filepaths:
        # Verifica l'estensione del file
        if Path(input_image_path).suffix.lower() in allowed_extensions:
            # Chiamiamo la funzione definita in main.py per processare l'immagine
            main.process_image(input_image_path)

            # Recuperiamo il nome dell'immagine processata
            output_image_folder = os.path.join('output_immagini', 'image_censored')
            temp_image_name = os.path.basename(input_image_path)
            for filename in os.listdir(output_image_folder):
                if filename.startswith(f"censored_{temp_image_name}"):
                    output_file = Path(os.path.join(output_image_folder, filename))
                    processed_images.append(output_file)
                    license_plate_text = get_license_plate_text(temp_image_name.split('.')[0])
                    detected_texts.append(license_plate_text)
                    break

    # Restituisce la lista dei file delle immagini processate e i testi rilevati
    return processed_images, "\n".join(detected_texts)


def compgetFolderUploader():
    folder_uploader = gr.File(
        file_count="directory",
        type="filepath",
        label="Seleziona la cartella contenente le immagini da processare",
        show_label=True,
        interactive=True
    )
    return folder_uploader


def compgetGenerateButton():
    generate_button = gr.Button(
        value="Processa Immagini",
        variant="primary"
    )
    return generate_button


def compgetFolderImages():
    images = gr.Gallery(
        label="Immagini processate",
        show_label=True,
        interactive=False,
    )
    return images


def compgetClearButton():
    # Pulsante per cancellare l'immagine caricata
    clear_button = gr.Button(
        value="Clear"
    )
    return clear_button


def compgetTextBoxSingle():
    text_box = gr.Textbox(
        label="Testo della targa rilevata",
        show_label=True,
        interactive=False
    )
    return text_box

def compgetTextBoxFolder():
    text_box = gr.Textbox(
        label="Testo delle targhe rilevate",
        show_label=True,
        interactive=False
    )
    return text_box

# Costruzione dell'interfaccia utente
def buildGUI():
    with gr.Blocks(title="PrivacyVisionAI") as demo:
        # Titolo
        gr.Markdown('''
            # PrivacyVisionAI
        ''')

        # Descrizione
        gr.Markdown('''
            Software di Censura Automatica e Protezione della Privacy con Intelligenza Artificiale. Carica un'immagine singola o una cartella contenente immagini per applicare la censura automatica su contenuti sensibili: targhe e volti.
        ''')

        with gr.Tabs():
            with gr.TabItem("Processa Immagine Singola"):
                with gr.Column():
                    # Componente per caricare un'immagine singola.
                    single_image_uploader = gr.Image(
                        label="Carica un'immagine singola da processare",
                        type="numpy",
                        interactive=True
                    )

                    # Pulsante per processare un'immagine singola.
                    process_single_button = gr.Button(
                        value="Processa Immagine Singola",
                        variant="primary"
                    )

                    clear_single_button = compgetClearButton()
                    text_box_single = compgetTextBoxSingle()

                    # Definizione dell'ascoltatore di eventi per l'elaborazione di un'immagine singola.
                    process_single_button.click(
                        fn=process_single_image,
                        inputs=[single_image_uploader],
                        outputs=[gr.Image(label="Immagine processata"), text_box_single],
                    )

                    # Ascoltatore per il pulsante di cancellazione
                    clear_single_button.click(
                        fn=lambda: (None, ""),
                        outputs=[single_image_uploader, text_box_single]
                    )

                    gr.Row(process_single_button, clear_single_button)

            with gr.TabItem("Processa Cartella"):
                with gr.Column():
                    # Componente per caricare la cartella dell'utente.
                    folder_uploader = compgetFolderUploader()

                    # Pulsante per processare la cartella.
                    processa_button = compgetGenerateButton()

                    clear_folder_button = compgetClearButton()

                    text_box_folder = compgetTextBoxFolder()


                    # Definizione dell'ascoltatore di eventi per l'elaborazione della cartella.
                    processa_button.click(
                        fn=process_folder,
                        inputs=folder_uploader,
                        outputs=[compgetFolderImages(), text_box_folder],
                        scroll_to_output=True,
                        show_progress='full'
                    )

                    # Ascoltatore per il pulsante di cancellazione
                    clear_folder_button.click(
                        fn=lambda: (None, ""),
                        outputs=[folder_uploader, text_box_folder]
                    )

                    gr.Row(processa_button, clear_folder_button)

    return demo


# Avvio dell'interfaccia
demo = buildGUI()
demo.launch(share=True)
