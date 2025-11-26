"""
Sistema de Detecção de Vagas de Estacionamento
Detecta veículos e verifica disponibilidade de vagas em tempo real
"""
import cv2
import json
from pathlib import Path
from detector import VehicleDetector
from utils.drawing import draw_parking_spot, draw_info_panel

# ========== CONFIGURAÇÃO DO MODELO ==========
# Escolha qual modelo usar:
USE_CUSTOM_MODEL = True  # True = modelo treinado | False = modelo pré-treinado

# Modelo customizado (treinado com Stanford Car Dataset)
CUSTOM_MODEL_PATH = "car_detection_test/train/weights/best.pt"

# Modelo pré-treinado YOLO (genérico)
PRETRAINED_MODEL = "yolov8n.pt"
# ===========================================


def main():
    # Carrega configuração das vagas
    try:
        with open('vagas.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            parking_spots = config['vagas']
    except FileNotFoundError:
        print("⚠️  Arquivo vagas.json não encontrado!")
        print("📝 Criando configuração padrão...")
        parking_spots = [
            {"id": 1, "coords": [100, 200, 200, 300]},
            {"id": 2, "coords": [220, 200, 320, 300]},
            {"id": 3, "coords": [340, 200, 440, 300]}
        ]
    
    # Seleciona modelo
    if USE_CUSTOM_MODEL and Path(CUSTOM_MODEL_PATH).exists():
        model_path = CUSTOM_MODEL_PATH
        print(f"🎯 Usando modelo customizado: {model_path}")
        print("   (196 classes de carros específicos)")
    else:
        model_path = PRETRAINED_MODEL
        print(f"🔧 Usando modelo pré-treinado: {model_path}")
        if USE_CUSTOM_MODEL:
            print("⚠️  Modelo customizado não encontrado, usando pré-treinado")
    
    # Inicializa detector
    detector = VehicleDetector(model_path=model_path)
    
    # ========== CONFIGURAÇÃO DA FONTE DE VÍDEO ==========
    # Escolha UMA das opções abaixo:
    
    # Opção 1: Webcam local
    # cap = cv2.VideoCapture(0)
    
    # Opção 2: Arquivo de vídeo
    # cap = cv2.VideoCapture('caminho/do/video.mp4')
    
    # Opção 3: Câmera IP / Stream MJPEG
    VIDEO_URL = "http://192.168.1.5:8080/video"
    cap = cv2.VideoCapture(VIDEO_URL)
    
    # Se HTTPS não funcionar, tente HTTP
    if not cap.isOpened():
        VIDEO_URL_HTTP = VIDEO_URL.replace('https://', 'http://')
        print(f"⚠️  Tentando HTTP: {VIDEO_URL_HTTP}")
        cap = cv2.VideoCapture(VIDEO_URL_HTTP)
    # ===================================================
    
    if not cap.isOpened():
        print("❌ Erro ao abrir câmera/vídeo")
        return
    
    print("🚀 Sistema iniciado! Pressione ESC para sair")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Fim do vídeo ou erro na captura")
            break
        
        # Detecta veículos
        vehicles = detector.detect_vehicles(frame)
        
        # Verifica ocupação das vagas
        occupied_count = 0
        for spot in parking_spots:
            spot_coords = spot['coords']
            is_occupied = detector.check_spot_occupied(vehicles, spot_coords)
            
            if is_occupied:
                occupied_count += 1
            
            # Desenha vaga no frame
            draw_parking_spot(frame, spot_coords, is_occupied, spot['id'])
        
        # Desenha painel de informações
        total_spots = len(parking_spots)
        free_spots = total_spots - occupied_count
        draw_info_panel(frame, total_spots, free_spots, occupied_count)
        
        # Mostra resultado
        cv2.imshow("Detecção de Vagas - Estacionamento Vision", frame)
        
        # ESC para sair
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Sistema encerrado")


if __name__ == "__main__":
    main()
