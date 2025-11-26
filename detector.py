"""
Módulo de Detecção de Veículos
Utiliza YOLO para detectar carros, motos, caminhões e ônibus
"""
from ultralytics import YOLO


class VehicleDetector:
    """Detector de veículos usando YOLO"""
    
    # Classes COCO para veículos
    VEHICLE_CLASSES = {
        2: 'car',           # carro
        3: 'motorcycle',    # moto
        5: 'bus',           # ônibus
        7: 'truck'          # caminhão
    }
    
    def __init__(self, model_path='yolov8n.pt', confidence=0.5):
        """
        Inicializa o detector
        
        Args:
            model_path: Caminho para o modelo YOLO (padrão: yolov8n.pt)
            confidence: Threshold de confiança mínima (0.0 - 1.0)
        """
        print(f"🔄 Carregando modelo YOLO: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        print("✅ Modelo carregado com sucesso!")
    
    def detect_vehicles(self, frame):
        """
        Detecta veículos no frame
        
        Args:
            frame: Imagem/frame para análise
            
        Returns:
            Lista de dicionários com informações dos veículos detectados
            [{'class': 'car', 'coords': [x1, y1, x2, y2], 'confidence': 0.85}, ...]
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        vehicles = []
        
        for detection in results[0].boxes:
            class_id = int(detection.cls[0])
            
            # Verifica se é um veículo
            if class_id in self.VEHICLE_CLASSES:
                coords = detection.xyxy[0].cpu().numpy().tolist()
                confidence = float(detection.conf[0])
                
                vehicles.append({
                    'class': self.VEHICLE_CLASSES[class_id],
                    'coords': coords,  # [x1, y1, x2, y2]
                    'confidence': confidence
                })
        
        return vehicles
    
    def check_spot_occupied(self, vehicles, spot_coords):
        """
        Verifica se uma vaga está ocupada
        
        Args:
            vehicles: Lista de veículos detectados
            spot_coords: Coordenadas da vaga [x1, y1, x2, y2]
            
        Returns:
            True se ocupada, False se livre
        """
        sx1, sy1, sx2, sy2 = spot_coords
        
        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = vehicle['coords']
            
            # Verifica sobreposição (overlap)
            if self._is_overlapping(vx1, vy1, vx2, vy2, sx1, sy1, sx2, sy2):
                return True
        
        return False
    
    @staticmethod
    def _is_overlapping(x1, y1, x2, y2, sx1, sy1, sx2, sy2):
        """
        Verifica se dois retângulos se sobrepõem
        
        Args:
            x1, y1, x2, y2: Coordenadas do veículo
            sx1, sy1, sx2, sy2: Coordenadas da vaga
            
        Returns:
            True se há sobreposição
        """
        # Retângulos NÃO se sobrepõem se:
        # - Um está completamente à esquerda do outro
        # - Um está completamente acima do outro
        return not (x2 < sx1 or x1 > sx2 or y2 < sy1 or y1 > sy2)
    
    def get_annotated_frame(self, frame):
        """
        Retorna frame com anotações do YOLO (todas as detecções)
        
        Args:
            frame: Frame original
            
        Returns:
            Frame anotado com todas as detecções
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        return results[0].plot()
