"""
Sistema Híbrido de Detecção de Vagas
Combina YOLO (detecção de veículos) + CNN (classificação vazia/ocupada)
"""
import cv2
import numpy as np
from pathlib import Path
from tensorflow import keras
from detector import VehicleDetector
import threading
import queue
import time


class VideoCapture:
    """Classe otimizada para captura de vídeo sem delay"""
    
    def __init__(self, src, use_thread=True):
        self.cap = cv2.VideoCapture(src)
        self.use_thread = use_thread
        
        # Configurações para reduzir latência
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo
        
        if use_thread:
            self.q = queue.Queue(maxsize=1)
            self.stopped = False
            self.error = None
            
            # Thread para captura contínua
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()
    
    def _reader(self):
        """Thread que lê frames continuamente"""
        while not self.stopped:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                
                # Descarta frame antigo se já tem um novo
                if not self.q.empty():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                
                self.q.put(frame)
            except Exception as e:
                self.error = str(e)
                self.stopped = True
                break
            
            time.sleep(0.001)  # Pequena pausa para reduzir CPU
    
    def read(self):
        """Retorna o frame mais recente"""
        if self.use_thread:
            if self.error:
                print(f"⚠️  Erro na thread de captura: {self.error}")
                return False, None
            if self.q.empty():
                return False, None
            return True, self.q.get()
        else:
            # Modo direto sem thread (mais estável para algumas câmeras)
            return self.cap.read()
    
    def isOpened(self):
        return self.cap.isOpened()
    
    def release(self):
        if self.use_thread:
            self.stopped = True
            if self.thread.is_alive():
                self.thread.join(timeout=1)
        self.cap.release()


class SistemaVagasHibrido:
    """
    Sistema que combina:
    1. YOLO para detectar veículos
    2. CNN para classificar cada vaga (vazia/ocupada)
    """
    
    def __init__(self, yolo_model_path='yolov8n.pt', cnn_model_path='modelos/vaga_detector_best.h5'):
        """
        Inicializa o sistema híbrido
        
        Args:
            yolo_model_path: Caminho do modelo YOLO
            cnn_model_path: Caminho do modelo CNN
        """
        print("🔄 Inicializando sistema híbrido...")
        
        # YOLO para detecção de veículos
        print("  Carregando YOLO...")
        self.vehicle_detector = VehicleDetector(model_path=yolo_model_path)
        
        # CNN para classificação de vagas
        if Path(cnn_model_path).exists():
            print(f"  Carregando CNN de: {cnn_model_path}")
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning)
            self.cnn_model = keras.models.load_model(cnn_model_path)
            self.use_cnn = True
            print("  ✅ Modo: YOLO + CNN (híbrido)")
        else:
            print(f"  ⚠️  CNN não encontrada: {cnn_model_path}")
            print("  ✅ Modo: Apenas YOLO")
            self.cnn_model = None
            self.use_cnn = False
        
        self.parking_spots = []
    
    def adicionar_vaga(self, coords, spot_id=None):
        """
        Adiciona uma vaga para monitoramento
        
        Args:
            coords: [x1, y1, x2, y2]
            spot_id: ID da vaga
        """
        self.parking_spots.append({
            'id': spot_id or len(self.parking_spots) + 1,
            'coords': coords
        })
    
    def carregar_vagas_json(self, json_path='vagas.json'):
        """Carrega vagas de arquivo JSON"""
        import json
        try:
            with open(json_path, 'r') as f:
                config = json.load(f)
                self.parking_spots = config['vagas']
            print(f"✅ {len(self.parking_spots)} vagas carregadas")
        except FileNotFoundError:
            print(f"⚠️  Arquivo {json_path} não encontrado")
    
    def classificar_vaga_cnn(self, frame, coords):
        """
        Classifica uma vaga usando CNN
        
        Args:
            frame: Frame completo
            coords: Coordenadas da vaga [x1, y1, x2, y2]
            
        Returns:
            (is_occupied, confidence)
        """
        x1, y1, x2, y2 = map(int, coords)
        
        # Extrai região da vaga
        vaga_img = frame[y1:y2, x1:x2]
        
        if vaga_img.size == 0:
            return False, 0.0
        
        # Preprocessa para CNN
        vaga_gray = cv2.cvtColor(vaga_img, cv2.COLOR_BGR2GRAY)
        vaga_resized = cv2.resize(vaga_gray, (48, 48))
        vaga_normalized = vaga_resized.reshape(1, 48, 48, 1).astype('float32') / 255
        
        # Predição
        prediction = self.cnn_model.predict(vaga_normalized, verbose=0)[0]
        
        # 0 = vazia, 1 = ocupada
        is_occupied = prediction[1] > prediction[0]
        confidence = float(prediction[1] if is_occupied else prediction[0])
        
        return is_occupied, confidence
    
    def classificar_vaga_yolo(self, vehicles, coords):
        """
        Classifica vaga usando detecção YOLO
        
        Args:
            vehicles: Lista de veículos detectados
            coords: Coordenadas da vaga
            
        Returns:
            is_occupied
        """
        return self.vehicle_detector.check_spot_occupied(vehicles, coords)
    
    def processar_frame(self, frame, metodo='hibrido'):
        """
        Processa frame e detecta status das vagas
        
        Args:
            frame: Frame da câmera
            metodo: 'hibrido', 'cnn', ou 'yolo'
            
        Returns:
            vagas_status: Lista com status de cada vaga
        """
        vagas_status = []
        
        # Detecta veículos com YOLO
        vehicles = self.vehicle_detector.detect_vehicles(frame)
        
        # Analisa cada vaga
        for spot in self.parking_spots:
            coords = spot['coords']
            spot_id = spot['id']
            
            if metodo == 'cnn' and self.use_cnn:
                # Apenas CNN
                is_occupied, confidence = self.classificar_vaga_cnn(frame, coords)
                metodo_usado = 'CNN'
                
            elif metodo == 'yolo':
                # Apenas YOLO
                is_occupied = self.classificar_vaga_yolo(vehicles, coords)
                confidence = 1.0 if is_occupied else 0.0
                metodo_usado = 'YOLO'
                
            else:  # híbrido (padrão)
                # Combina YOLO + CNN
                yolo_occupied = self.classificar_vaga_yolo(vehicles, coords)
                
                if self.use_cnn:
                    cnn_occupied, cnn_conf = self.classificar_vaga_cnn(frame, coords)
                    
                    # Decisão final: se um dos dois detectar ocupação
                    is_occupied = yolo_occupied or (cnn_occupied and cnn_conf > 0.6)
                    confidence = cnn_conf
                    metodo_usado = 'Híbrido'
                else:
                    is_occupied = yolo_occupied
                    confidence = 1.0
                    metodo_usado = 'YOLO'
            
            vagas_status.append({
                'id': spot_id,
                'coords': coords,
                'occupied': is_occupied,
                'confidence': confidence,
                'method': metodo_usado
            })
        
        return vagas_status, vehicles
    
    def desenhar_resultados(self, frame, vagas_status, show_vehicles=False, vehicles=None):
        """
        Desenha resultados no frame
        
        Args:
            frame: Frame para desenhar
            vagas_status: Status das vagas
            show_vehicles: Mostrar veículos detectados
            vehicles: Lista de veículos (opcional)
        """
        occupied_count = sum(1 for v in vagas_status if v['occupied'])
        free_count = len(vagas_status) - occupied_count
        
        # Desenha cada vaga
        for vaga in vagas_status:
            x1, y1, x2, y2 = map(int, vaga['coords'])
            is_occupied = vaga['occupied']
            confidence = vaga['confidence']
            
            # Cor
            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            
            # Retângulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label
            status = "OCUPADA" if is_occupied else "LIVRE"
            label = f"#{vaga['id']} {status} ({confidence:.0%})"
            
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        # Desenha veículos se solicitado
        if show_vehicles and vehicles:
            for vehicle in vehicles:
                x1, y1, x2, y2 = map(int, vehicle['coords'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        # Painel de info
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 80), (50, 50, 50), -1)
        
        cv2.putText(
            frame, "SISTEMA HIBRIDO - YOLO + CNN",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        cv2.putText(
            frame, f"Vagas Livres: {free_count} | Ocupadas: {occupied_count} | Total: {len(vagas_status)}",
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )


def main():
    """Demonstração do sistema híbrido"""
    print("=" * 70)
    print("🚗 SISTEMA HÍBRIDO DE DETECÇÃO DE VAGAS")
    print("=" * 70)
    
    # Inicializa sistema
    sistema = SistemaVagasHibrido(
        yolo_model_path='car_detection_test/train/weights/best.pt',
        cnn_model_path='modelos/vaga_detector_best.h5'
    )
    
    # Carrega vagas
    sistema.carregar_vagas_json('vagas.json')
    
    if len(sistema.parking_spots) == 0:
        print("\n⚠️  Nenhuma vaga configurada!")
        print("Adicione vagas manualmente ou crie vagas.json")
        return
    
    # Câmera - Tente HTTPS primeiro, depois HTTP, depois webcam
    CAMERA_URLS = [
        "https://10.120.0.17:8080/video",  # Câmera IP HTTPS
        "http://10.120.0.17:8080/video",   # Câmera IP HTTP
    ]
    
    # Tente primeiro com thread (mais rápido mas pode dar erro em algumas câmeras)
    USE_THREAD = True
    
    cap = None
    for url in CAMERA_URLS:
        print(f"🔗 Tentando: {url}")
        try:
            cap = VideoCapture(url, use_thread=USE_THREAD)
            if cap.isOpened():
                print(f"✅ Conectado: {url}")
                # Aguarda 1 segundo para ver se thread funciona
                time.sleep(1)
                ret, test_frame = cap.read()
                if ret:
                    break
                else:
                    print("⚠️  Thread com erro, tentando modo direto...")
                    cap.release()
                    cap = VideoCapture(url, use_thread=False)
                    if cap.isOpened():
                        print(f"✅ Conectado em modo direto: {url}")
                        break
            cap.release()
        except Exception as e:
            print(f"❌ Erro: {e}")
            if cap:
                cap.release()
    
    if cap is None or not cap.isOpened():
        print("⚠️  Câmera IP não disponível, usando webcam...")
        cap = VideoCapture(0, use_thread=False)  # Webcam geralmente funciona melhor sem thread
    
    if not cap.isOpened():
        print("❌ Erro ao abrir vídeo")
        return
    
    print("\n🚀 Sistema rodando!")
    print("📝 Controles:")
    print("   ESC - Sair")
    print("   1 - Modo Híbrido (YOLO + CNN)")
    print("   2 - Modo CNN apenas")
    print("   3 - Modo YOLO apenas")
    print("   V - Mostrar/ocultar veículos")
    print("   + - Processar todos os frames (mais lento)")
    print("   - - Pular frames (mais rápido, menos preciso)")
    print("\n⏳ Aguardando frames da câmera...")
    
    metodo = 'hibrido'
    show_vehicles = False
    skip_frames = 1  # Processa 1 a cada N frames (1 = todos, 2 = metade, etc)
    frame_count = 0
    error_count = 0
    max_errors = 10
    
    while True:
        try:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                error_count += 1
                print(f"⚠️  Erro ao ler frame ({error_count}/{max_errors})")
                
                if error_count >= max_errors:
                    print("❌ Muitos erros consecutivos, encerrando...")
                    break
                
                time.sleep(0.1)
                continue
            
            # Reset error counter on success
            error_count = 0
            frame_count += 1
            
            # Debug primeira leitura
            if frame_count == 1:
                print(f"✅ Primeiro frame recebido! Resolução: {frame.shape[1]}x{frame.shape[0]}")
        except Exception as e:
            print(f"❌ Exceção ao ler frame: {e}")
            error_count += 1
            if error_count >= max_errors:
                break
            continue
        
        frame_count += 1
        
        # Pular frames para aumentar velocidade
        if frame_count % skip_frames != 0:
            # Ainda desenha o último resultado mesmo pulando frames
            if 'last_vagas_status' in locals() and 'last_vehicles' in locals():
                sistema.desenhar_resultados(frame, last_vagas_status, show_vehicles, last_vehicles)
            
            try:
                cv2.imshow("Sistema Híbrido - Detecção de Vagas", frame)
            except Exception as e:
                print(f"❌ Erro ao mostrar frame: {e}")
                break
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\n🛑 ESC pressionado, encerrando...")
                break
            elif key == ord('+') or key == ord('='):
                skip_frames = max(1, skip_frames - 1)
                print(f"⚡ Skip frames: {skip_frames} (processando {100//skip_frames}% dos frames)")
            elif key == ord('-') or key == ord('_'):
                skip_frames = min(5, skip_frames + 1)
                print(f"⚡ Skip frames: {skip_frames} (processando {100//skip_frames}% dos frames)")
            
            continue
        
        # Processa
        try:
            vagas_status, vehicles = sistema.processar_frame(frame, metodo=metodo)
            last_vagas_status = vagas_status
            last_vehicles = vehicles
            
            # Desenha
            sistema.desenhar_resultados(frame, vagas_status, show_vehicles, vehicles)
        except Exception as e:
            print(f"⚠️  Erro ao processar frame: {e}")
            # Continua com último resultado
            if 'last_vagas_status' in locals() and 'last_vehicles' in locals():
                sistema.desenhar_resultados(frame, last_vagas_status, show_vehicles, last_vehicles)
        
        try:
            cv2.imshow("Sistema Híbrido - Detecção de Vagas", frame)
        except Exception as e:
            print(f"❌ Erro ao mostrar frame: {e}")
            break
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\n🛑 ESC pressionado, encerrando...")
            break
        elif key == ord('1'):
            metodo = 'hibrido'
            print("▶️  Modo: Híbrido (YOLO + CNN)")
        elif key == ord('2'):
            metodo = 'cnn'
            print("▶️  Modo: CNN")
        elif key == ord('3'):
            metodo = 'yolo'
            print("▶️  Modo: YOLO")
        elif key == ord('v') or key == ord('V'):
            show_vehicles = not show_vehicles
            print(f"📦 Veículos: {'VISÍVEIS' if show_vehicles else 'OCULTOS'}")
        elif key == ord('+') or key == ord('='):
            skip_frames = max(1, skip_frames - 1)
            print(f"⚡ Skip frames: {skip_frames} (processando {100//skip_frames}% dos frames)")
        elif key == ord('-') or key == ord('_'):
            skip_frames = min(5, skip_frames + 1)
            print(f"⚡ Skip frames: {skip_frames} (processando {100//skip_frames}% dos frames)")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Sistema encerrado")


if __name__ == "__main__":
    main()
