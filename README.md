# boxboxAI 🏎️🤖

Proyecto de simulación de carreras con agentes de **Aprendizaje por Refuerzo Profundo (Deep Reinforcement Learning - DRL)** basado en Gymnasium y Stable-Baselines3. El simulador permite entrenar y evaluar agentes inteligentes para optimizar estrategias de parada en boxes y rendimiento en carrera utilizando datos reales de Grandes Premios de Fórmula 1 extraídos vía FastF1.

---

## 📂 Estructura del Proyecto

El repositorio está estructurado de la siguiente manera para mantener un entorno limpio y profesional:

```text
boxboxAI/
├── tests/                       # Directorio de pruebas unitarias e integración
│   ├── test_gym_integration.py  # Pruebas del entorno Gymnasium
│   ├── test_integration.py      # Pruebas integrales de simulación de carrera
│   └── test_logger.py           # Pruebas del sistema de logging
├── .gitignore                   # Archivos y carpetas excluidos en control de versiones
├── bahrain_2024.json            # Datos históricos reales del GP de Bahréin 2024
├── domain_model.py              # Lógica y física del simulador (Car, Track, Race, etc.)
├── fetch_race_data.py           # Script para extraer telemetría real usando FastF1
├── race_config.py               # Configuración e instanciación de carreras desde datos JSON
├── race_gym_env.py              # Entorno Gymnasium personalizado para el agente DRL
├── README.md                    # Documentación principal del proyecto
├── requirements.txt             # Dependencias necesarias para ejecutar el simulador
├── simulation_logger.py         # Logger estructurado de telemetría y eventos
├── train_agent.py               # Script principal para entrenar y evaluar el agente PPO/A2C/DQN
└── [Excluidos]                  # Carpetas locales como venv/, .fastf1_cache/, logs/, etc.
```

---

## 🛠️ Instalación y Configuración

Sigue estos pasos para configurar el entorno de ejecución en tu máquina local:

### 1. Requisitos Previos
Asegúrate de tener instalado **Python 3.8** o superior en tu sistema.

### 2. Crear y Activar Entorno Virtual
Se recomienda utilizar un entorno virtual para aislar las dependencias:

```bash
# Crear entorno virtual
python -m venv venv

# Activar en Windows (PowerShell)
venv\Scripts\Activate.ps1

# Activar en Windows (CMD)
venv\Scripts\activate.bat

# Activar en macOS / Linux
source venv/bin/activate
```

### 3. Instalar Dependencias
Instala los paquetes requeridos especificados en `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🕹️ Uso del Simulador

### 1. Probar el Entorno Dinámico (Modo Humano/Demostración)
Puedes comprobar el funcionamiento del entorno Gymnasium con acciones automáticas iniciales ejecutando:

```bash
python train_agent.py test
```

### 2. Entrenar el Agente DRL
Para iniciar el entrenamiento del agente utilizando el algoritmo PPO (configurado por defecto con datos reales del GP de España 2024):

```bash
python train_agent.py
```
*Esto entrenará al agente por 200,000 pasos de tiempo, guardará el progreso en Tensorboard, generará una gráfica de rendimiento y almacenará el modelo final como `race_agent_ppo.zip`.*

### 3. Extraer Datos de Nuevos Grandes Premios
Puedes descargar datos reales de cualquier circuito utilizando `fetch_race_data.py`. FastF1 requiere una conexión a Internet la primera vez que se descarga un GP:

```bash
# Ejemplo: Extraer datos del GP de España de 2024
python fetch_race_data.py --year 2024 --gp "Spanish" --output spanish_gp_2024.json
```

---

## ⚙️ Características de la Simulación

*   **Modelo de Degradación de Neumáticos**: El desgaste y la pérdida de rendimiento se calculan dinámicamente mediante un modelo cuadrático adaptado a datos reales.
*   **Adelantamientos Probabilísticos**: Los adelantamientos se modelan considerando la brecha entre pilotos, delta de neumáticos, rendimiento del coche y la probabilidad de adelantamiento empírica del circuito.
*   **Gestión de Combustible**: Los coches son más rápidos al final de la carrera debido a la reducción del peso del combustible a lo largo de las vueltas.
*   **Toma de Decisiones DRL**: El espacio de observaciones proporciona datos completos (posición, desgaste del juego actual, tiempos de los rivales) para que el agente decida la estrategia óptima de parada en boxes (`SOFT`, `MEDIUM`, `HARD` o continuar en pista).

---

## 🧪 Pruebas Unitarias e Integración

Para validar que todos los componentes y el entorno de Gymnasium funcionen correctamente, ejecuta los tests utilizando pytest o unittest:

```bash
# Ejecutar todas las pruebas en el directorio tests/
python -m unittest discover -s tests
```

---

## 📈 Monitoreo y Visualización

*   **Tensorboard**: Puedes monitorizar métricas como recompensa promedio, pérdidas del modelo y progreso del aprendizaje con:
    ```bash
    tensorboard --logdir=race_tensorboard
    ```
*   **Gráfico de Progreso**: Tras el entrenamiento, se genera el archivo `training_progress.png` comparando el retorno obtenido y la posición promedio en carrera del agente a lo largo de los episodios.
