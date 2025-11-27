# 🚗 Sistema Inteligente de Detecção de Vagas de Estacionamento

Sistema completo de monitoramento de vagas usando **YOLO + CNN** com suporte para câmera IP do celular.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)
[![Acurácia](https://img.shields.io/badge/Acur%C3%A1cia-97.4%25-brightgreen.svg)]()

---

## 📋 Características Principais

| Funcionalidade           | Descrição                                    |
| ------------------------ | -------------------------------------------- |
| 🎯 **Detecção Híbrida**  | YOLO detecta veículos + CNN classifica vagas |
| 📱 **Câmera do Celular** | Use seu smartphone como câmera IP            |
| 🎨 **Marcação Visual**   | Interface interativa para definir vagas      |
| ⚡ **Tempo Real**        | Processamento otimizado com baixa latência   |
| 🎛️ **3 Modos**           | Híbrido, CNN-only, YOLO-only                 |
| 📊 **Alta Precisão**     | CNN com 97.40% de acurácia                   |
| 💻 **Multiplataforma**   | Windows, Linux, macOS                        |

---

## 📱 Parte 1: Configurar Câmera IP no Celular

### 🔹 Passo 1.1: Instalar App

**📱 Android - IP Webcam (Recomendado)**

1. Abra a **Google Play Store**
2. Busque: **"IP Webcam"** (Pavel Khlebovich)
3. Clique em **Instalar**

**🍎 iPhone - IP Camera**

1. Abra a **App Store**
2. Busque: **"IP Camera - WiFi Home Security"**
3. Clique em **Obter**

### 🔹 Passo 1.2: Configurar o App

**No celular:**

1. **Abra o app** instalado
2. Role a tela para baixo até o fim
3. Ajuste as configurações (opcional):
   - **Resolução de vídeo:** 720p (recomendado para velocidade)
   - **Qualidade:** 80%
   - **FPS:** 15-20 fps
   - **Encoder:** H.264 ou MJPEG
4. Toque em **"Start Server"** ou **"Iniciar Servidor"**

**O app mostrará algo assim:**

```
==============================
Seu endereço IP é:
http://192.168.1.5:8080
==============================
```

✅ **IMPORTANTE:** Anote este IP completo! Você vai precisar dele.

### 🔹 Passo 1.3: Testar Conexão

**No computador:**

1. Certifique-se que **celular e computador estão na mesma rede WiFi**
2. Abra um navegador (Chrome, Firefox, Edge)
3. Digite o endereço completo que apareceu no app:
   ```
   http://192.168.1.5:8080
   ```
4. Pressione **Enter**

**✅ Se funcionar:** Você verá a imagem da câmera do celular no navegador!

**❌ Se não funcionar:** Vá para [Solução de Problemas](#-solução-de-problemas)

### 🔹 Passo 1.4: Descobrir URL do Stream de Vídeo

A URL completa do vídeo depende do app:

| App                         | URL do Vídeo               |
| --------------------------- | -------------------------- |
| **IP Webcam** (Android)     | `http://IP:8080/video`     |
| **IP Camera** (iOS/Android) | `http://IP:8080/video`     |
| **iVCam**                   | `http://IP:8080/mjpegfeed` |
| **DroidCam**                | `http://IP:4747/video`     |

**Exemplo completo:**

```
http://192.168.1.5:8080/video
```

---

## 💻 Parte 2: Instalar o Sistema

### 🔹 Passo 2.1: Verificar Requisitos

**Requisitos de Sistema:**

- ✅ **Python 3.8 ou superior** (testado no 3.13)
- ✅ **4GB de RAM** mínimo (8GB recomendado)
- ✅ **2GB de espaço em disco**
- ✅ **Windows 10/11, Linux ou macOS**
- ✅ **Conexão WiFi**

**Verificar se Python está instalado:**

```bash
python --version
```

Se aparecer algo como `Python 3.13.x`, está OK! ✅

**❌ Se não tiver Python:** Baixe em [python.org](https://www.python.org/downloads/)

### 🔹 Passo 2.2: Baixar o Projeto

**Opção 1 - Git (se tiver instalado):**

```bash
git clone https://github.com/seu-usuario/estacionamento-vision.git
cd estacionamento-vision
```

**Opção 2 - Download direto:**

1. Baixe o arquivo ZIP do projeto
2. Extraia para uma pasta (ex: `C:\projetos\estacionamento-vision`)
3. Abra o terminal nesta pasta

### 🔹 Passo 2.3: Criar Ambiente Virtual

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# Windows PowerShell:
.\venv\Scripts\activate

# Windows CMD:
venv\Scripts\activate.bat

# Linux/Mac:
source venv/bin/activate
```

✅ Você verá `(venv)` antes do caminho no terminal.

### 🔹 Passo 2.4: Instalar Bibliotecas

```bash
pip install opencv-python ultralytics tensorflow scikit-learn matplotlib kagglehub numpy
```

**Tempo estimado:** 2-5 minutos (depende da internet)

**Bibliotecas instaladas:**

| Biblioteca      | Função                      |
| --------------- | --------------------------- |
| `opencv-python` | Processar imagens e vídeo   |
| `ultralytics`   | YOLO para detectar veículos |
| `tensorflow`    | Rede neural CNN             |
| `scikit-learn`  | Dividir dados e métricas    |
| `matplotlib`    | Gerar gráficos              |
| `kagglehub`     | Baixar datasets             |
| `numpy`         | Computação numérica         |

### 🔹 Passo 2.5: Baixar Modelo Pré-treinado (Opcional)

O projeto já vem com modelo CNN treinado (97.4% acurácia), mas você pode baixar o dataset e retreinar:

```bash
# Download dataset (1.726 imagens, ~5MB)
python baixar_dataset_vagas.py

# Treinar modelo CNN (~3 minutos)
python treinar_cnn_vagas.py
```

✅ **Se pular este passo:** O sistema usa o modelo já incluído em `modelos/vaga_detector_best.h5`

---

## ⚙️ Parte 3: Configurar o Sistema com Sua Câmera

### 🔹 Passo 3.1: Atualizar IP da Câmera

Você precisa editar **4 arquivos** e colocar o IP do seu celular:

**📝 Arquivo 1: `sistema_vagas_cnn.py`**

Abra o arquivo e encontre a linha ~323:

```python
CAMERA_URLS = [
    "http://192.168.1.5:8080/video",  # ← MUDE AQUI PARA SEU IP
    "https://192.168.1.5:8080/video",
]
```

**📝 Arquivo 2: `sistema_vagas_simples.py`**

Linha ~121:

```python
CAMERA_URLS = [
    "http://192.168.1.5:8080/video",  # ← MUDE AQUI PARA SEU IP
    "https://192.168.1.5:8080/video",
]
```

**📝 Arquivo 3: `capturar_frame_camera.py`**

Linha ~8:

```python
CAMERA_URL = "http://192.168.1.5:8080/video"  # ← MUDE AQUI PARA SEU IP
```

**📝 Arquivo 4: `camera_ip.py`**

Procure a linha com `CAMERA_URL` e mude:

```python
CAMERA_URL = "http://192.168.1.5:8080/video"  # ← MUDE AQUI PARA SEU IP
```

**💡 Exemplo prático:**

Se o IP do seu celular for `http://192.168.0.10:8080`, mude para:

```python
"http://192.168.0.10:8080/video"
```

### 🔹 Passo 3.2: Testar Conexão com a Câmera

Teste se o sistema consegue conectar:

```bash
python capturar_frame_camera.py
```

**✅ Saída esperada:**

```
📷 CAPTURA DE FRAME DA CÂMERA IP
🔗 Conectando em: http://192.168.0.10:8080/video
✅ Conectado!
📸 Capturando frame...
✅ Frame salvo: camera_frame.jpg
   Resolução: 1920x1080
```

Um arquivo `camera_frame.jpg` será criado com a imagem da câmera.

**❌ Se der erro:** Confira o IP e vá para [Solução de Problemas](#-solução-de-problemas)

---

## 🎯 Parte 4: Marcar as Vagas

### 🔹 Passo 4.1: Executar Ferramenta de Marcação

```bash
python marcar_vagas.py camera_frame.jpg
```

Uma janela abrirá com a imagem da sua câmera.

### 🔹 Passo 4.2: Desenhar Vagas

**Como marcar:**

1. **Abra o arquivo vagas.json**
2. **Altere os valores das posições x,y**
3. **confira as posições**

**Repita** para cada vaga que você quer monitorar.

**Dicas importantes:**

- ✅ Desenhe retângulos que cobrem toda a vaga (com margem pequena)
- ✅ Não sobreponha retângulos
- ✅ Marque apenas vagas claramente visíveis
- ❌ Evite marcar vagas muito pequenas na imagem

### 🔹 Passo 4.4: Verificar Arquivo vagas.json

Abra `vagas.json` e confira se as vagas foram salvas:

```json
{
  "vagas": [
    {
      "id": 1,
      "coords": [100, 200, 300, 400],
      "descricao": "Vaga 1"
    },
    {
      "id": 2,
      "coords": [320, 200, 520, 400],
      "descricao": "Vaga 2"
    }
  ]
}
```

**Formato das coordenadas:** `[x1, y1, x2, y2]`

- `(x1, y1)` = canto superior esquerdo
- `(x2, y2)` = canto inferior direito

---

## 🚀 Parte 5: Executar o Sistema

### 🔹 Passo 5.1: Escolher Qual Sistema Usar

Você tem 2 opções:

| Sistema                      | Quando usar                                |
| ---------------------------- | ------------------------------------------ |
| **sistema_vagas_cnn.py**     | Câmera IP estável, quer máxima velocidade  |
| **sistema_vagas_simples.py** | Câmera com problemas, quer estabilidade ⭐ |

**Recomendado para iniciantes:** `sistema_vagas_simples.py`

### 🔹 Passo 5.2: Executar Sistema

**Opção 1 - Sistema Completo (com threading):**

```bash
python sistema_vagas_cnn.py
```

**Opção 2 - Sistema Simplificado (mais estável) ⭐:**

```bash
python sistema_vagas_simples.py
```

### 🔹 Passo 5.3: Aguardar Inicialização

Você verá algo assim:

```
======================================================================
🚗 SISTEMA HÍBRIDO DE DETECÇÃO DE VAGAS
======================================================================
🔄 Inicializando sistema híbrido...
  Carregando YOLO...
✅ Modelo carregado com sucesso!
  Carregando CNN...
✅ Modo: YOLO + CNN (híbrido)
✅ 6 vagas carregadas
🔗 Tentando: http://192.168.0.10:8080/video
✅ Conectado!
✅ Primeiro frame recebido! Resolução: 1920x1080

🚀 Sistema rodando!
```

### 🔹 Passo 5.4: Usar a Interface

Uma janela abrirá mostrando:

```
┌─────────────────────────────────────┐
│ Total: 6 vagas                      │
│ Livres: 4                           │
│ Ocupadas: 2                         │
└─────────────────────────────────────┘

Vaga #1 - LIVRE     ██████ Verde
Vaga #2 - OCUPADA   ██████ Vermelho
Vaga #3 - LIVRE     ██████ Verde
```

### 🔹 Passo 5.5: Controles durante Execução

| Tecla   | Função                                           |
| ------- | ------------------------------------------------ |
| **ESC** | Sair do sistema                                  |
| **1**   | Modo Híbrido (YOLO + CNN) ⭐ Recomendado         |
| **2**   | Modo CNN apenas                                  |
| **3**   | Modo YOLO apenas (mais rápido)                   |
| **V**   | Mostrar/ocultar veículos detectados              |
| **+**   | Processar mais frames (mais preciso, mais lento) |
| **-**   | Pular frames (mais rápido, menos preciso)        |
| **S**   | Salvar screenshot                                |

### 🔹 Passo 5.6: Entender os Modos

**🔷 Modo 1 - Híbrido (YOLO + CNN)**

- Usa YOLO para detectar veículos
- Usa CNN para confirmar se vaga está vazia/ocupada
- **Mais preciso** ✅
- Um pouco mais lento

**🔷 Modo 2 - Apenas CNN**

- Analisa imagem da vaga diretamente
- Bom para vagas sem movimento
- Velocidade média

**🔷 Modo 3 - Apenas YOLO**

- Apenas detecta veículos
- **Mais rápido** 🚀
- Menos preciso

---

## 📊 Informações Técnicas

### 🔹 Métricas do Modelo CNN

| Métrica              | Valor                                     |
| -------------------- | ----------------------------------------- |
| **Acurácia**         | **97.40%** ✅                             |
| Loss                 | 0.0532                                    |
| Dataset              | 1.726 imagens (908 vazias + 818 ocupadas) |
| Treino/Validação     | 1.380 / 346 imagens                       |
| Épocas treinadas     | 24 (com early stopping)                   |
| Tempo de treinamento | ~2-3 minutos                              |

**Arquitetura da CNN:**

- 3 blocos Conv2D (16 → 32 → 64 filtros)
- MaxPooling + Dropout entre cada bloco
- 2 camadas Dense (128 → 64 neurônios)
- Camada de saída: Softmax (2 classes)

### 🔹 Modelo YOLO

| Característica     | Detalhes                      |
| ------------------ | ----------------------------- |
| Versão             | YOLOv8n (nano)                |
| Classes detectadas | car, motorcycle, bus, truck   |
| FPS                | ~30 (varia conforme hardware) |
| Tamanho            | ~6MB                          |
| Confiança mínima   | 0.5 (50%)                     |

### 🔹 Estrutura de Arquivos

```
estacionamento-vision/
│
├── 📁 Sistema Principal
│   ├── sistema_vagas_cnn.py        # Sistema com threading ⚡
│   ├── sistema_vagas_simples.py    # Sistema estável ⭐
│   ├── main.py                      # Sistema YOLO tradicional
│   └── camera_ip.py                 # Teste de câmera
│
├── 📁 Ferramentas
│   ├── capturar_frame_camera.py    # Captura frame da câmera
│   ├── marcar_vagas.py             # Marcar vagas interativamente
│   ├── detector.py                 # Classe detector YOLO
│   └── utils/drawing.py            # Funções de desenho
│
├── 📁 Treinamento CNN
│   ├── baixar_dataset_vagas.py     # Download dataset Kaggle
│   ├── treinar_cnn_vagas.py        # Treinar modelo CNN
│   └── modelos/
│       ├── vaga_detector_best.h5   # Modelo CNN (97.4%) ✅
│       ├── vaga_detector_final.h5  # Modelo final
│       └── training_history.png    # Gráfico de treinamento
│
├── 📁 Treinamento YOLO (Opcional)
│   ├── preparar_dataset.py         # Prepara Stanford Cars
│   ├── treinar_modelo.py           # Treina YOLO customizado
│   ├── testar_modelo.py            # Testa modelo
│   └── GUIA_TREINAMENTO.md         # Guia completo
│
├── 📁 Datasets
│   ├── dataset_vagas/              # Vagas vazias/ocupadas
│   │   ├── empty/     (908 imgs)
│   │   └── occupied/  (818 imgs)
│   └── car_data/                   # Stanford Car Dataset
│       ├── train/     (8.144 imgs)
│       └── val/
│
└── 📁 Configuração
    ├── vagas.json                  # Suas vagas marcadas
    ├── camera_frame.jpg            # Frame capturado
    └── README.md                   # Este arquivo
```

---

## 🔧 Solução de Problemas

### ❌ Problema: Não conecta na câmera IP

**Verificações:**

1. **Celular e PC na mesma WiFi?**

   - No celular: Configurações → WiFi → Veja o nome da rede
   - No PC: Ícone WiFi → Veja o nome da rede
   - Devem ser iguais!

2. **IP está correto?**

   - Confira o IP que aparece no app do celular
   - Deve ser algo como `192.168.x.x:8080`
   - Números podem mudar quando reconecta no WiFi

3. **Servidor está rodando no celular?**

   - Abra o app no celular
   - Deve estar escrito "Server is running" ou "Servidor iniciado"
   - Se não, aperte "Start Server"

4. **Firewall bloqueando?**

   ```bash
   # Windows: Desabilitar temporariamente firewall
   # Painel de Controle → Sistema e Segurança → Firewall do Windows
   ```

5. **Testar no navegador primeiro:**
   - Abra Chrome/Firefox no PC
   - Digite: `http://SEU_IP:8080`
   - Deve mostrar a câmera

### ❌ Problema: Sistema muito lento / FPS baixo

**Soluções:**

1. **Use modo YOLO apenas (mais rápido):**

   - Pressione tecla **3** durante execução

2. **Reduza resolução no app da câmera:**

   - Configure para 720p ou 480p
   - Menos pixels = mais rápido

3. **Pule frames:**

   - Pressione tecla **-** várias vezes
   - Sistema processará menos frames

4. **Use sistema simplificado:**

   ```bash
   python sistema_vagas_simples.py
   ```

5. **Reduza número de vagas:**
   - Monitore apenas vagas importantes
   - Menos vagas = mais rápido

### ❌ Problema: Detecção imprecisa

**Soluções:**

1. **Use modo híbrido:**

   - Pressione tecla **1**
   - Combina YOLO + CNN (mais preciso)

2. **Remarque as vagas:**

   ```bash
   python marcar_vagas.py camera_frame.jpg
   ```

   - Desenhe retângulos maiores
   - Inclua margem nas bordas

3. **Melhore iluminação:**

   - Evite sombras fortes
   - Use iluminação uniforme
   - Evite contra-luz

4. **Ajuste ângulo da câmera:**
   - Prefira vista de cima (bird's eye)
   - Ângulo de 45° a 90°

### ❌ Problema: "ModuleNotFoundError"

```
ModuleNotFoundError: No module named 'cv2'
```

**Solução:**

```bash
# Certifique-se que está no ambiente virtual
.\venv\Scripts\activate

# Reinstale OpenCV
pip install opencv-python

# Verifique instalação
python -c "import cv2; print(cv2.__version__)"
```

### ❌ Problema: IP da câmera muda toda vez

**Solução:**

No roteador WiFi, configure IP fixo para o celular (DHCP reservation):

1. Acesse seu roteador (geralmente `192.168.1.1`)
2. Procure "DHCP" ou "Reserva de IP"
3. Adicione o MAC address do celular
4. Defina um IP fixo (ex: `192.168.1.100`)

Ou use nome do host (se suportado):

```python
CAMERA_URL = "http://nome-do-celular.local:8080/video"
```

---

## 💡 Dicas e Truques

### 🔹 Performance Máxima

1. **Conecte celular no carregador** - Vai usar muita bateria
2. **Desative protetor de tela** - No app, geralmente tem opção "Keep screen on"
3. **Use cabo de rede (Ethernet)** - Se possível, conecte PC via cabo
4. **Feche outros apps** - No celular e no PC
5. **Configure FPS para 15-20** - Mais que isso é desperdício

### 🔹 Melhor Qualidade de Detecção

1. **Boa iluminação** - Fundamental!
2. **Câmera fixa** - Use suporte/tripé
3. **Ângulo de cima** - Vista de pássaro é melhor
4. **Marque vagas com precisão** - Retângulos bem desenhados
5. **Use modo híbrido** - Tecla 1 durante execução

### 🔹 Economizar Bateria do Celular

1. **Reduza brilho da tela** - Ou desligue (se app permitir)
2. **Configure FPS baixo** - 10-15 fps é suficiente
3. **Use resolução baixa** - 480p ou 640p
4. **Desative GPS/Bluetooth** - Não são necessários
5. **Conecte no carregador!** - Sempre!

### 🔹 Usar Múltiplas Câmeras

Para monitorar estacionamento grande com várias câmeras:

1. **Configure cada celular** em um IP diferente
2. **Execute múltiplas instâncias** do sistema:

   ```bash
   # Terminal 1
   python sistema_vagas_cnn.py  # Câmera 1

   # Terminal 2
   python sistema_vagas_cnn.py  # Câmera 2
   ```

3. **Marque vagas diferentes** em cada (crie `vagas1.json`, `vagas2.json`, etc)

---

## 🎓 Retreinar Modelos (Avançado)

### 🔹 Retreinar CNN

Se quiser ajustar o modelo para suas condições específicas:

```bash
# 1. Baixar dataset original (1.726 imagens)
python baixar_dataset_vagas.py

# 2. Treinar modelo
python treinar_cnn_vagas.py
```

**Ajustar hiperparâmetros em `treinar_cnn_vagas.py`:**

```python
epochs = 50              # Mais épocas = mais tempo, pode melhorar
batch_size = 64          # Maior = mais RAM, mais rápido
learning_rate = 0.0005   # Menor = mais lento, mais estável
```

**Tempo esperado:** 2-5 minutos  
**Resultado esperado:** 95-98% acurácia

### 🔹 Treinar YOLO Customizado

Para detectar tipos específicos de carros:

```bash
# 1. Preparar Stanford Car Dataset (8.144 imagens, 196 classes)
python preparar_dataset.py

# 2. Treinar modelo YOLO
python treinar_modelo.py

# 3. Testar modelo
python testar_modelo.py
```

**Tempo esperado:**

- Fast (10 épocas): ~15 minutos
- Complete (100 épocas): ~2 horas

---

## 📚 Recursos Adicionais

### 🔗 Links Úteis

- **Datasets:**

  - [Parking Dataset (Kaggle)](https://www.kaggle.com/datasets/rizwanrizwannazir/parking) - 1.726 imagens
  - [Stanford Car Dataset](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder) - 8.144 imagens

- **Documentação:**

  - [OpenCV Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
  - [Ultralytics YOLO](https://docs.ultralytics.com/)
  - [TensorFlow](https://www.tensorflow.org/api_docs/python/tf)

- **Apps Câmera IP:**
  - [IP Webcam (Android)](https://play.google.com/store/apps/details?id=com.pas.webcam)
  - [iVCam (iOS/Android)](https://www.e2esoft.com/ivcam/)

### 📖 Arquivos de Documentação

- `README.md` - Este arquivo (guia completo)
- `GUIA_TREINAMENTO.md` - Guia detalhado YOLO
- `modelos/training_history.png` - Gráficos do treinamento CNN

### 🛠️ Tecnologias Usadas

| Tecnologia | Versão | Uso                      |
| ---------- | ------ | ------------------------ |
| Python     | 3.8+   | Linguagem principal      |
| OpenCV     | 4.x    | Processamento de imagens |
| YOLO       | v8     | Detecção de veículos     |
| TensorFlow | 2.x    | Rede neural CNN          |
| Keras      | 3.x    | API de alto nível        |
| NumPy      | 1.24+  | Computação numérica      |

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Áreas de melhoria:

- 🌐 Interface web (Flask/Django)
- 📱 App mobile nativo
- 🔔 Sistema de notificações
- 📊 Dashboard de estatísticas
- 🎥 Suporte para múltiplas câmeras simultâneas
- 🚗 Reconhecimento de placas (OCR)
- ☁️ Deploy em nuvem (AWS/Azure)

---

## 📄 Licença

MIT License - Livre para uso pessoal e comercial.

---

## ✨ Créditos

**Desenvolvido por:** Murilo Jean Claudio, Luiz Felipe, Herick Eduardo  
**Datasets:** Rizwan Nazir (Kaggle), Stanford University  
**Frameworks:** Ultralytics (YOLO), Google (TensorFlow)  
**Inspiração:** Monitoramento inteligente de estacionamentos

---

**Status do Projeto:** ✅ Pronto para uso | 🧠 CNN Treinado (97.4%) | 🚗 YOLO Configurado

**Última atualização:** 26/11/2025

---

## 🚀 Quick Start (Resumo de 5 passos)

```bash
# 1. Instale app "IP Webcam" no celular e inicie servidor

# 2. Clone projeto e instale dependências
git clone <repo>
cd estacionamento-vision
python -m venv venv
.\venv\Scripts\activate
pip install opencv-python ultralytics tensorflow scikit-learn matplotlib kagglehub

# 3. Configure IP nos arquivos (mude 192.168.1.5 para seu IP)
# Edite: sistema_vagas_cnn.py, capturar_frame_camera.py

# 4. Capture frame e marque vagas
python capturar_frame_camera.py
python marcar_vagas.py camera_frame.jpg
# (Desenhe retângulos, pressione 's' para salvar)

# 5. Execute o sistema
python sistema_vagas_simples.py
# (Pressione '1' para modo híbrido)
```

**Pronto! Sistema rodando em 5 minutos!** 🎉
0
