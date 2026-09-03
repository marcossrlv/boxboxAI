# boxboxAI 🏎️🤖

Proyecto de simulación de carreras y optimización estratégica de paradas en boxes con agentes de **Aprendizaje por Refuerzo Profundo (Deep Reinforcement Learning - DRL)** basado en [Gymnasium](https://gymnasium.farama.org/) y [Stable-Baselines3](https://stable-baselines3.readthedocs.io/). 

El simulador permite entrenar, evaluar y comparar agentes inteligentes frente a estrategias reales de Grandes Premios de Fórmula 1 y estrategias heurísticas tradicionales, calibrado con telemetría y datos históricos extraídos mediante [FastF1](https://docs.fastf1.dev/).

---

## 📂 Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

```text
boxboxAI/
├── data/                        # Datos JSON de Grandes Premios reales (telemetría, parrilla y estrategias)
│   ├── bahrain_2024.json        # Datos históricos del GP de Bahréin 2024
│   └── spanish_gp_2024.json     # Datos históricos del GP de España 2024
├── plots/                       # Gráficas generadas durante la evaluación comparativa
│   ├── average_positions.png    # Comparación de posiciones vuelta a vuelta entre estrategias
│   ├── cumulative_rewards.png   # Evolución de la recompensa acumulada por vuelta
│   ├── evaluation_comparison.png# Distribución de posiciones finales en estudio Monte Carlo
│   └── tire_degradation.png     # Ciclo de vida y degradación de los compuestos de neumáticos
├── tests/                       # Suite de pruebas unitarias e integración
│   ├── conftest.py              # Configuración de fixtures para pytest
│   ├── test_gym_integration.py  # Pruebas del entorno Gymnasium
│   ├── test_integration.py      # Pruebas integrales de la física y simulación de carrera
│   └── test_logger.py           # Pruebas del sistema de logging y telemetría
├── domain_model.py              # Lógica y física del simulador (Car, Track, Race, TireModel, FuelModel)
├── evaluate_agent.py            # Evaluación Monte Carlo y suite de comparación de estrategias
├── fetch_race_data.py           # Extracción y ajuste de telemetría real usando FastF1
├── race_agent_ppo.zip           # Checkpoint del modelo entrenado con PPO
├── race_config.py               # Configuración e instanciación de carreras desde archivos JSON
├── race_gym_env.py              # Entorno Gymnasium personalizado (espacio de estados y acciones)
├── race_tensorboard/            # Métricas y registros para visualización en TensorBoard
├── requirements.txt             # Dependencias necesarias del proyecto
├── simulation_logger.py         # Logger estructurado de telemetría y eventos en carrera
├── train_agent.py               # Script para entrenar el agente DRL (PPO) y monitorizar el progreso
├── training_progress.png        # Gráfica de evolución del entrenamiento (recompensa y posición)
└── README.md                    # Documentación principal del proyecto
```

---

## 🛠️ Instalación y configuración

Sigue estos pasos para configurar el entorno de ejecución en tu máquina local:

### 1. Requisitos previos
Asegúrate de contar con **Python 3.8** o superior instalado en tu sistema.

### 2. Crear y activar entorno virtual
Se recomienda utilizar un entorno virtual para aislar las dependencias:

```bash
# Crear entorno virtual
python -m venv venv

# Activar en macOS / Linux
source venv/bin/activate

# Activar en Windows (PowerShell)
venv\Scripts\Activate.ps1

# Activar en Windows (CMD)
venv\Scripts\activate.bat
```

### 3. Instalar dependencias
Instala los paquetes requeridos especificados en `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🕹️ Uso del simulador

### 1. Probar el entorno
Para comprobar el correcto funcionamiento del entorno Gymnasium con renderizado en consola:

```bash
python train_agent.py test
```

### 2. Entrenar el Agente DRL
Para entrenar un agente con el algoritmo **PPO** (configurado por defecto con datos reales del GP de España 2024):

```bash
python train_agent.py
```
*Este proceso entrena al agente durante 2.000.000 de pasos de tiempo, registra las métricas en TensorBoard, guarda el modelo en `race_agent_ppo.zip`, genera la gráfica `training_progress.png` y ejecuta un estudio Monte Carlo al finalizar.*

### 3. Evaluar el Agente y comparar estrategias

El script `evaluate_agent.py` permite evaluar el rendimiento del agente entrenado y compararlo contra pilotos reales y estrategias heurísticas:

#### A) Estudio Monte Carlo (100 simulaciones estocásticas)
Ejecuta múltiples simulaciones independientes para evaluar la distribución de posiciones y consistencia del agente frente a la aleatoriedad de carrera:

```bash
python evaluate_agent.py --model race_agent_ppo --gp data/spanish_gp_2024.json --episodes 100
```

#### B) Suite de comparación de estrategias (`--compare`)
Compara en paralelo tres enfoques estratégicos bajo las mismas condiciones de carrera:
1. **Agente DRL**: Decisiones autónomas en tiempo real según el estado de la pista y desgaste.
2. **Piloto Real (FastF1)**: Reproducción de la estrategia histórica del piloto en el GP real.
3. **Estrategia Heurística Base**: Estrategia estática convencional de 2 paradas.

```bash
python evaluate_agent.py --compare --gp data/spanish_gp_2024.json
```
*Genera las comparativas detalladas en el directorio `plots/` (`average_positions.png`, `cumulative_rewards.png`, `evaluation_comparison.png` y `tire_degradation.png`).*

#### Opciones principales de `evaluate_agent.py`:
| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `--model` | `str` | `race_agent_ppo` | Ruta del archivo de modelo guardado (con o sin `.zip`) |
| `--gp` | `str` | `data/spanish_gp_2024.json` | Archivo JSON con la configuración del Gran Premio |
| `--episodes` | `int` | `100` | Número de episodios para el estudio Monte Carlo |
| `--compare` | `flag` | `False` | Ejecuta la suite de comparación de estrategias |
| `--stochastic`| `flag` | `False` | Utiliza política estocástica en lugar de determinista |
| `--render` | `flag` | `False` | Muestra el log paso a paso de cada vuelta en consola |
| `--algo` | `choice`| `ppo` | Algoritmo RL del modelo (`ppo`, `a2c`, `dqn`) |

### 4. Extraer datos de nuevos Grandes Premios
Puedes descargar y modelar datos reales de cualquier circuito utilizando `fetch_race_data.py`. FastF1 requiere conexión a Internet la primera vez que se descarga un evento:

```bash
# Ejemplo: Extraer datos del GP de España de 2024
python fetch_race_data.py --year 2024 --gp "Spanish" --output data/spanish_gp_2024.json

# Ejemplo: Extraer datos del GP de Bahréin de 2024
python fetch_race_data.py --year 2024 --gp "Bahrain" --output data/bahrain_2024.json
```

---

## ⚙️ Características de la simulación

*   **Modelo cuadrático de degradación de neumáticos (`QuadraticTireModel`)**:
    La pérdida de rendimiento por vuelta $n$ con un juego de neumáticos se modela como:
    $$\Delta t(n) = a \cdot n^2 + b \cdot n + c$$
    Los coeficientes $(a, b, c)$ de cada compuesto (`SOFT`, `MEDIUM`, `HARD`) se obtienen mediante ajuste polinomial por mínimos cuadrados sobre tiempos reales de carrera extraídos vía FastF1.
*   **Gestión de carga de combustible (`FuelModel`)**:
    Simula la masa inicial requerida para el Gran Premio y el consumo de combustible vuelta a vuelta (aprox. 1.6 kg/vuelta), aportando una ganancia progresiva de ritmo conforme los monoplazas se vuelven más ligeros.
*   **Coste físico de parada en boxes**:
    Modelado realista del tiempo perdido en el paso por el pit lane (`pit_stop_loss_s`) y cambio de neumáticos.
*   **Adelantamientos probabilísticos y tráfico**:
    Los adelantamientos se calculan dinámicamente considerando la brecha entre coches, el diferencial de ritmo y vida de neumáticos, y la probabilidad empírica de adelantamiento del circuito.
*   **Estrategias reales**:
    Soporte para cargar y reproducir las secuencias de parada reales registradas por los pilotos oficiales durante el Gran Premio.
*   **Espacio de observaciones y acciones DRL**:
    *   **Acciones**: `0` = Mantenerse en pista, `1` = Box montando SOFT, `2` = Box montando MEDIUM, `3` = Box montando HARD.
    *   **Observaciones**: Posición actual, vuelta de carrera, vueltas con el neumático actual, tipo de compuesto, nivel de combustible y gaps relativos frente a los competidores más cercanos.

---

## 🧪 Pruebas unitarias e integración

Para verificar que todos los componentes, la física del modelo y el entorno Gymnasium funcionan correctamente:

```bash
# Ejecutar pruebas con pytest
pytest

# Alternativamente, con unittest
python -m unittest discover -s tests
```

---

## 📈 Visualización

*   **TensorBoard**: Supervisa la evolución de la función de recompensa, longitud de episodios, pérdida de valor y métricas de política en tiempo real:
    ```bash
    tensorboard --logdir=race_tensorboard
    ```
*   **Gráfica de entrenamiento (`training_progress.png`)**:
    Muestra la curva de recompensa media acumulada y la evolución de la posición promedio en carrera a lo largo de los episodios de entrenamiento.
*   **Gráficas de evaluación (`plots/`)**:
    *   `evaluation_comparison.png`: Histograma de frecuencias de posiciones finales en Monte Carlo comparado con la posición real del piloto y su puesto en parrilla.
    *   `average_positions.png`: Evolución de la posición media vuelta a vuelta del agente DRL frente a la estrategia real y la heurística.
    *   `tire_degradation.png`: Curvas de vida del neumático y ventanas de parada de cada compuesto en las diferentes estrategias evaluadas.
    *   `cumulative_rewards.png`: Acumulación de recompensa por vuelta a lo largo del Gran Premio.

