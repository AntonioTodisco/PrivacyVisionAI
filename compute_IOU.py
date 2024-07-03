
#FUNZIONE PER Intersection over Union
def compute_iou(box1, box2):
    """
    Calcola l'Intersection over Union (IoU) tra due bounding box.
    
    Args:
        box1 (dict): Coordinate del primo bounding box, con chiavi 'left', 'top', 'right', 'bottom'.
        box2 (dict): Coordinate del secondo bounding box, con chiavi 'left', 'top', 'right', 'bottom'.
        
    Returns:
        float: L'IoU tra i due bounding box.
        0 significa che i due bounding box non si sovrappongono affatto.
        1 significa che i due bounding box sono esattamente sovrapposti.
        Un valore intermedio tra 0 e 1 rappresenta il grado di sovrapposizione tra i due bounding box. 
        Più il valore è vicino a 1, maggiore è la sovrapposizione.
    """
    
    # Calcola la larghezza e l'altezza dell'area di intersezione
    intersection_width = min(box1['right'], box2['right']) - max(box1['left'], box2['left'])
    intersection_height = min(box1['bottom'], box2['bottom']) - max(box1['top'], box2['top'])
    
    # Se non c'è intersezione, l'area di intersezione è zero
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0
    
    # Calcola l'area di intersezione
    intersection_area = intersection_width * intersection_height

    # Calcola l'area di ciascun box
    box1_area = (box1['right'] - box1['left']) * (box1['bottom'] - box1['top'])
    box2_area = (box2['right'] - box2['left']) * (box2['bottom'] - box2['top'])
    
    # Calcola l'area di unione
    union_area = box1_area + box2_area - intersection_area

    # Calcola l'IoU
    iou = intersection_area / union_area
    
    return iou 



#FUNZIONE PER RECUPERARE I VALORI DEL BOUNDING-BOX DA UN FILE.TXT. RESTITUISCE UN DIZIONARIO CON I VALORI
def read_bounding_boxes(file_path):
    """
    Legge i bounding box da un file e li converte nel formato richiesto.
    
    Args:
        file_path (str): Percorso del file contenente i bounding box.
        
    Returns:
        list of dict: Una lista di bounding box con chiavi 'left', 'top', 'right', 'bottom'.
    """
    boxes = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            values = line.strip().split()
            class_id =float(values[0])
            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])
            # Convert from (x_center, y_center, width, height) to (left, top, right, bottom)
            left = x_center - width / 2
            top = y_center - height / 2
            right = x_center + width / 2
            bottom = y_center + height / 2
            boxes.append({'left': left, 'top': top, 'right': right, 'bottom': bottom})
    return boxes, class_id



#FUNZIONE PER RECUPERARE I VALORI DEL BOUNDING-BOX DA UN FILE.TXT. RESTITUISCE UN DIZIONARIO CON I VALORI
def read_bounding_boxes_TargheAndFaces(file_path):
    """
    Legge i bounding box da un file e li converte nel formato richiesto.
    
    Args:
        file_path (str): Percorso del file contenente i bounding box.
        
    Returns:
        list of dict: Una lista di bounding box con chiavi 'left', 'top', 'right', 'bottom'.
    """
    boxes = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            values = line.strip().split()
            #class_id = float(values[0])
            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])

            # Convert from (x_center, y_center, width, height) to (left, top, right, bottom)
            left = x_center - width / 2
            top = y_center - height / 2
            right = x_center + width / 2
            bottom = y_center + height / 2
            
            boxes.append({'left': left, 'top': top, 'right': right, 'bottom': bottom})
    return boxes


#funzione che va a leggere la classe dei caratteri e li restituisce in un dizionario di valori
def read_class_char(file_path):
    class_id_list = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            values = line.strip().split()
            class_id = float(values[0])
            
            class_id_list.append(class_id)
    return class_id_list





def read_bounding_boxes_total(filename):
    bounding_boxes = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split()
            if len(values) <5:
                continue  # Skip invalid lines: classi targas
            try:
                class_id = float(values[0])
                x_center = float(values[1])
                y_center = float(values[2])
                width = float(values[3])
                height = float(values[4])

                # Convert from (x_center, y_center, width, height) to (left, top, right, bottom)
                left = x_center - width / 2
                top = y_center - height / 2
                right = x_center + width / 2
                bottom = y_center + height / 2

                bounding_boxes.append({'class_id':class_id ,'left': left, 'top': top, 'right': right, 'bottom': bottom})
            except ValueError:
                continue  # Skip lines that can't be parsed as numbers
    return bounding_boxes
