"""
Sistema Híbrido SIMPLIFICADO - Sem Threading
Versão mais estável para câmeras que apresentam problemas com threads
"""
import cv2
import numpy as np
from pathlib import Path
from tensorflow import keras
from detector import VehicleDetector
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class SistemaVagasSimples:
    """Sistema híbrido simplificado sem threading"""
    
    def __init__(self, yolo_model_path='yolov8n.pt', cnn_model_path='modelos/vaga_detector_best.h5'):
        print("🔄 Inicializando sistema...")
        
        # YOLO
        print("  Carregando YOLO...")
        self.vehicle_detector = VehicleDetector(model_path=yolo_model_path)
        
        # CNN
        if Path(cnn_model_path).exists():
            print(f"  Carregando CNN...")
            self.cnn_model = keras.models.load_model(cnn_model_path)
            self.use_cnn = True
            print("  ✅ Modo: YOLO + CNN")
        else:
            print("  ⚠️  CNN não encontrada, apenas YOLO")
            self.cnn_model = None
            self.use_cnn = False
        
        self.parking_spots = []
    
    def carregar_vagas_json(self, json_path='vagas.json'):
        """Carrega vagas de arquivo JSON"""
        try:
            with open(json_path, 'r') as f:
                config = json.load(f)
                self.parking_spots = config['vagas']
            print(f"✅ {len(self.parking_spots)} vagas carregadas")
        except FileNotFoundError:
            print(f"⚠️  Arquivo {json_path} não encontrado")
    
    def classificar_vaga_cnn(self, frame, coords):
        """Classifica vaga usando CNN"""
        x1, y1, x2, y2 = coords
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return False, 0.0
        
        # Preprocessar
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_resized.reshape(1, 48, 48, 1).astype('float32') / 255
        
        # Prever
        prediction = self.cnn_model.predict(roi_normalized, verbose=0)
        is_occupied = prediction[0][1] > 0.5
        confidence = float(prediction[0][1] if is_occupied else prediction[0][0])
        
        return is_occupied, confidence
    
    def processar_frame(self, frame, metodo='hibrido'):
        """Processa frame"""
        vagas_status = []
        vehicles = self.vehicle_detector.detect_vehicles(frame)
        
        for spot in self.parking_spots:
            coords = spot['coords']
            
            if metodo == 'hibrido':
                # YOLO primeiro
                occupied_yolo = self.vehicle_detector.check_spot_occupied(coords, vehicles)
                # CNN para confirmar
                occupied_cnn, conf = self.classificar_vaga_cnn(frame, coords) if self.use_cnn else (False, 0)
                # Decisão final: se um dos dois detectar, considera ocupado
                is_occupied = occupied_yolo or occupied_cnn
                confidence = conf
            elif metodo == 'cnn':
                is_occupied, confidence = self.classificar_vaga_cnn(frame, coords) if self.use_cnn else (False, 0)
            else:  # yolo
                is_occupied = self.vehicle_detector.check_spot_occupied(coords, vehicles)
                confidence = 1.0 if is_occupied else 0.0
            
            vagas_status.append({
                'id': spot['id'],
                'coords': coords,
                'occupied': is_occupied,
                'confidence': confidence
            })
        
        return vagas_status, vehicles
    
    def desenhar_resultados(self, frame, vagas_status, show_vehicles=False, vehicles=None):
        """Desenha resultados no frame"""
        livres = sum(1 for v in vagas_status if not v['occupied'])
        ocupadas = len(vagas_status) - livres
        
        # Desenhar vagas
        for vaga in vagas_status:
            x1, y1, x2, y2 = vaga['coords']
            color = (0, 0, 255) if vaga['occupied'] else (0, 255, 0)  # Vermelho/Verde
            
            # Retângulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"#{vaga['id']} {'OCUPADA' if vaga['occupied'] else 'LIVRE'}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Desenhar veículos
        if show_vehicles and vehicles:
            for vehicle in vehicles:
                x1, y1, x2, y2 = map(int, vehicle['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{vehicle['class']} {vehicle['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Painel de info
        cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Total: {len(vagas_status)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Livres: {livres}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Ocupadas: {ocupadas}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def main():
    print("=" * 70)
    print("🚗 SISTEMA HÍBRIDO SIMPLIFICADO - SEM DELAY")
    print("=" * 70)
    
    # Inicializa
    sistema = SistemaVagasSimples(
        yolo_model_path='car_detection_test/train/weights/best.pt',
        cnn_model_path='modelos/vaga_detector_best.h5'
    )
    
    sistema.carregar_vagas_json('vagas.json')
    
    if len(sistema.parking_spots) == 0:
        print("\n⚠️  Nenhuma vaga configurada!")
        return
    
    # Câmera
    CAMERA_URLS = [
        "http://10.118.3.92:8080/video",
        "https://10.118.3.92:8080/video",
    ]
    
    cap = None
    for url in CAMERA_URLS:
        print(f"\n🔗 Tentando: {url}")
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo
        
        if cap.isOpened():
            print(f"✅ Conectado!")
            break
        cap.release()
    
    if cap is None or not cap.isOpened():
        print("\n⚠️  Câmera IP não disponível, usando webcam...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ Erro ao abrir vídeo")
        return
    
    print("\n🚀 Sistema rodando!")
    print("📝 Controles:")
    print("   ESC - Sair")
    print("   1 - Modo Híbrido")
    print("   2 - Modo CNN")
    print("   3 - Modo YOLO")
    print("   V - Toggle veículos")
    print("   S - Salvar screenshot")
    
    metodo = 'hibrido'
    show_vehicles = False
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Erro ao ler frame")
            break
        
        frame_count += 1
        
        # Processar (pula frames alternados para velocidade)
        if frame_count % 2 == 0:
            vagas_status, vehicles = sistema.processar_frame(frame, metodo=metodo)
        
        # Desenhar
        if 'vagas_status' in locals():
            sistema.desenhar_resultados(frame, vagas_status, show_vehicles, vehicles)
        
        cv2.imshow("Sistema Híbrido - Detecção de Vagas", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('1'):
            metodo = 'hibrido'
            print("▶️  Modo: Híbrido")
        elif key == ord('2'):
            metodo = 'cnn'
            print("▶️  Modo: CNN")
        elif key == ord('3'):
            metodo = 'yolo'
            print("▶️  Modo: YOLO")
        elif key == ord('v') or key == ord('V'):
            show_vehicles = not show_vehicles
            print(f"📦 Veículos: {'VISÍVEIS' if show_vehicles else 'OCULTOS'}")
        elif key == ord('s') or key == ord('S'):
            filename = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot salvo: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Sistema encerrado")


if __name__ == "__main__":
    main()
