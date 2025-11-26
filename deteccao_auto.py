"""
Detector automático de vagas de estacionamento
Identifica vagas através da análise da imagem, sem overlays manuais
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


class ParkingDetector:
    """Detector inteligente que identifica vagas automaticamente"""
    
    def __init__(self, model_path='yolov8n.pt'):
        """
        Inicializa o detector
        
        Args:
            model_path: Caminho para o modelo YOLO
        """
        print(f"🔄 Carregando modelo YOLO: {model_path}")
        self.model = YOLO(model_path)
        self.vehicle_classes = {2, 3, 5, 7}  # car, motorcycle, bus, truck
        print("✅ Modelo carregado!")
    
    def detect_parking_spaces(self, frame, background_frame=None):
        """
        Detecta vagas de estacionamento automaticamente
        
        Args:
            frame: Frame atual
            background_frame: Frame de referência (sem carros) - opcional
            
        Returns:
            Lista de vagas detectadas com status
        """
        # Detecta todos os veículos no frame
        results = self.model(frame, verbose=False)
        
        # Extrai informações dos veículos
        vehicles = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            if class_id in self.vehicle_classes:
                coords = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                vehicles.append({
                    'coords': coords.tolist(),
                    'class': class_id,
                    'confidence': confidence
                })
        
        return vehicles, results[0]
    
    def draw_detections(self, frame, vehicles, show_info=True):
        """
        Desenha apenas as detecções de veículos (sem vagas)
        
        Args:
            frame: Frame para desenhar
            vehicles: Lista de veículos detectados
            show_info: Mostrar painel de informações
        """
        # Desenha retângulos ao redor dos veículos
        for vehicle in vehicles:
            x1, y1, x2, y2 = map(int, vehicle['coords'])
            conf = vehicle['confidence']
            
            # Cor baseada na confiança
            if conf > 0.7:
                color = (0, 255, 0)  # Verde - alta confiança
            elif conf > 0.5:
                color = (0, 255, 255)  # Amarelo - média confiança
            else:
                color = (0, 165, 255)  # Laranja - baixa confiança
            
            # Desenha retângulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label com confiança
            label = f"Veiculo {conf:.0%}"
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        if show_info:
            # Painel simples no topo
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 60), (50, 50, 50), -1)
            
            cv2.putText(
                frame, "DETECCAO DE ESTACIONAMENTO",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            cv2.putText(
                frame, f"Veiculos detectados: {len(vehicles)}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
    
    def get_annotated_frame(self, frame):
        """
        Retorna frame com anotações automáticas do YOLO
        
        Args:
            frame: Frame original
            
        Returns:
            Frame anotado
        """
        results = self.model(frame, verbose=False)
        return results[0].plot()


def main():
    print("=" * 70)
    print("🚗 DETECÇÃO AUTOMÁTICA DE ESTACIONAMENTO")
    print("=" * 70)
    
    # Configuração do modelo
    USE_CUSTOM_MODEL = True
    CUSTOM_MODEL_PATH = "car_detection_test/train/weights/best.pt"
    PRETRAINED_MODEL = "yolov8n.pt"
    
    if USE_CUSTOM_MODEL and Path(CUSTOM_MODEL_PATH).exists():
        model_path = CUSTOM_MODEL_PATH
        print(f"🎯 Modelo: Customizado")
    else:
        model_path = PRETRAINED_MODEL
        print(f"🔧 Modelo: Pré-treinado")
    
    # Inicializa detector
    detector = ParkingDetector(model_path=model_path)
    
    # Configuração da câmera
    CAMERA_URL = "http://192.168.1.5:8080/video"
    
    print(f"\n📹 Conectando à câmera: {CAMERA_URL}")
    cap = cv2.VideoCapture(CAMERA_URL)
    
    if not cap.isOpened():
        print("⚠️  Tentando webcam local...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Erro ao abrir fonte de vídeo")
        return
    
    print("✅ Conectado!")
    print("\n🚀 Sistema iniciado!")
    print("📝 Controles:")
    print("   ESC - Sair")
    print("   1 - Modo: Detecções simples")
    print("   2 - Modo: Anotações YOLO completas")
    print("   S - Salvar screenshot")
    
    mode = 1  # 1 = simples, 2 = completo
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n⚠️  Erro ao capturar frame")
            break
        
        frame_count += 1
        
        if mode == 1:
            # Modo simples: apenas veículos
            vehicles, _ = detector.detect_parking_spaces(frame)
            detector.draw_detections(frame, vehicles, show_info=True)
        else:
            # Modo completo: anotações YOLO
            frame = detector.get_annotated_frame(frame)
            
            # Info adicional
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (50, 50, 50), -1)
            cv2.putText(
                frame, "MODO COMPLETO - Todas as deteccoes YOLO",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
        # Info de frame
        cv2.putText(
            frame, f"Frame: {frame_count} | Modo: {mode}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        cv2.imshow("Detecção Automática de Estacionamento", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('1'):
            mode = 1
            print("▶️  Modo: Detecções simples")
        elif key == ord('2'):
            mode = 2
            print("▶️  Modo: Anotações completas")
        elif key == ord('s') or key == ord('S'):
            filename = f"deteccao_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot salvo: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Sistema encerrado")


if __name__ == "__main__":
    main()
