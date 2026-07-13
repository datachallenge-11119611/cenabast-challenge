# Challenge Analista de Datos - CENABAST

## 1. Enfoque general

El objetivo del challenge es operacionalizar un modelo predictivo que permita estimar el consumo de medicamentos a partir de datos históricos de movimientos y stock.

La solución implementada considera que el consumo relevante corresponde a los movimientos de salida (`S`) registrados en `movimientos.csv`. A partir de esos movimientos se construyen variables temporales para capturar patrones de tendencia, estacionalidad y comportamiento por producto.

La solución se estructura en tres componentes principales:

- `challenge/model.py`: implementación del modelo predictivo.
- `challenge/api.py`: API REST desarrollada con FastAPI.
- `challenge/train.py`: script para entrenar y guardar el modelo localmente.

Además, se mantiene la documentación técnica y logística en este archivo.

---

## 2. Datos utilizados

El repositorio contiene tres archivos principales en la carpeta `dataset/`:

### `productos.csv`

Contiene el maestro de productos. Sus principales campos son:

- `gtin`: identificador del producto.
- `material`: nombre del medicamento.
- `uso_principal`: indicación terapéutica principal.
- `linea_terapeutica`: línea terapéutica asociada.
- `canasta_vigente`: estado del producto en la canasta.

### `movimientos.csv`

Contiene los movimientos históricos de inventario. Sus principales campos son:

- `gtin`: identificador del producto.
- `fecha`: fecha del movimiento.
- `cantidad`: cantidad de unidades movidas.
- `tipo_movimiento`: tipo de movimiento.

Para el modelo predictivo se consideran principalmente los movimientos con:

```text
tipo_movimiento = "S"
```

Estos movimientos se interpretan como salidas o consumo.

### `stock.csv`

Contiene el registro diario de stock por producto:

- `gtin`
- `fecha`
- `stock`

Este archivo es especialmente relevante para el análisis logístico de reabastecimiento, ya que permite comparar el consumo proyectado con las existencias disponibles.

---

## 3. Preparación de datos

La preparación de datos se realiza en el método `preprocess()` de la clase `ReplenishmentModel`.

Las principales transformaciones realizadas son:

1. Conversión de `gtin` a texto.
2. Conversión de `fecha` a formato fecha.
3. Filtrado de movimientos de salida (`S`) cuando existe la columna `tipo_movimiento`.
4. Creación de variables temporales:
   - año;
   - mes;
   - día;
   - día de semana;
   - semana del año;
   - fecha ordinal;
   - días desde el inicio de la serie.

Estas variables permiten capturar patrones temporales de consumo, como tendencia y estacionalidad.

Cuando el método recibe `target_column="cantidad"`, retorna:

```text
features, target
```

Cuando no recibe `target_column`, retorna sólo:

```text
features
```

Esto permite usar el mismo método tanto para entrenamiento como para predicción desde la API.

---

## 4. Modelo predictivo

El modelo implementado utiliza `RandomForestRegressor` de `scikit-learn`.

Se escogió este modelo porque permite capturar relaciones no lineales entre las variables temporales y el consumo observado, sin requerir una parametrización compleja. Además, es una alternativa robusta para un primer modelo funcional y explicable en un contexto de challenge técnico.

El modelo no corresponde a tres modelos separados para tendencia, estacionalidad y demanda inelástica. Se implementa un único modelo de regresión que utiliza variables temporales y de producto para capturar esos patrones en conjunto. La tendencia se representa mediante variables de avance temporal; la estacionalidad mediante variables de calendario; y la demanda inelástica se aproxima mediante el comportamiento histórico persistente de cada `gtin`, dado que el dataset no contiene variables de precio, sustitución o elasticidad explícita.

La solución usa un `Pipeline` de `scikit-learn` compuesto por:

1. Preprocesamiento de variables categóricas:
   - `gtin`
   - `tipo_movimiento`

2. Preprocesamiento de variables numéricas:
   - fecha ordinal;
   - días desde inicio;
   - año;
   - mes;
   - día;
   - día de semana;
   - semana del año.

3. Modelo de regresión:
   - `RandomForestRegressor`.

El resultado final de la predicción se redondea y se restringe a valores no negativos, ya que el consumo de unidades no puede ser menor que cero.

---

## 5. API

La API fue implementada con FastAPI en el archivo:

```text
challenge/api.py
```

La API expone dos endpoints principales:

### `GET /health`

Permite verificar que la API esté activa.

Respuesta esperada:

```json
{
  "status": "OK"
}
```

### `POST /predict`

Permite solicitar predicciones de consumo para uno o más productos.

Ejemplo de entrada:

```json
{
  "products": [
    {
      "gtin": "7804587213648",
      "fecha": "2026-07-10"
    }
  ]
}
```

Ejemplo de salida:

```json
{
  "predict": [
    {
      "fecha": "2026-07-10",
      "cantidad": 12
    }
  ]
}
```

La API valida dos condiciones relevantes:

1. Que el `gtin` exista en `productos.csv`.
2. Que la fecha entregada sea válida.

Si el producto no existe, retorna error `400`.

Si la fecha no es válida, retorna error `400`.

---

## 6. Entrenamiento del modelo

El archivo `challenge/train.py` permite entrenar el modelo y guardarlo localmente.

Comando de ejecución:

```bash
python -m challenge.train
```

Este script realiza los siguientes pasos:

1. Carga `dataset/movimientos.csv`.
2. Preprocesa los datos.
3. Entrena el modelo.
4. Guarda el modelo entrenado como `model.pkl`.

El archivo `model.pkl` se considera un artefacto local generado y no se sube al repositorio.

---

## 7. Tests ejecutados

Se ejecutaron los tests locales de modelo y API.

### Tests de modelo

Comando:

```bash
python -m pytest tests/model -q
```

Resultado local:

```text
7 passed
```

### Tests de API

Comando:

```bash
python -m pytest tests/api -q
```

Resultado local:

```text
4 passed
```

### Tests conjuntos

Comando:

```bash
python -m pytest tests/model tests/api -q
```

Resultado local:

```text
11 passed
```

También se verificó manualmente la API local mediante Swagger UI en:

```text
http://127.0.0.1:8000/docs
```

Se probaron los siguientes casos:

- `/health` con respuesta `200 OK`.
- `/predict` con producto válido y fecha válida, con respuesta `200 OK`.
- `/predict` con producto desconocido, con respuesta `400 Bad Request`.
- `/predict` con fecha inválida, con respuesta `400 Bad Request`.

---

## 8. Análisis logístico para reabastecimiento

El modelo predictivo entrega una estimación de consumo esperado por producto y fecha. Esta predicción puede utilizarse como insumo para definir decisiones de reabastecimiento.

Una lógica operacional de reabastecimiento debería considerar al menos los siguientes elementos:

### 8.1 Consumo esperado

Para cada producto `gtin`, se estima el consumo futuro esperado usando el modelo predictivo.

Para un horizonte de planificación de `n` días:

```text
consumo_esperado_horizonte = suma de predicciones diarias en los próximos n días
```

### 8.2 Stock disponible

El stock disponible se obtiene desde el último registro disponible en `stock.csv`.

```text
stock_actual = último stock observado para el producto
```

### 8.3 Lead time

El lead time corresponde al tiempo esperado entre la emisión de una orden de reabastecimiento y la disponibilidad efectiva del producto en bodega.

Si el lead time es de `L` días, el consumo esperado durante ese período puede estimarse como:

```text
consumo_durante_lead_time = suma de consumo proyectado durante L días
```

### 8.4 Stock de seguridad

El stock de seguridad permite cubrir variabilidad de la demanda, retrasos de reposición o eventos no anticipados.

Una aproximación simple puede ser:

```text
stock_seguridad = consumo_promedio_diario * días_de_cobertura_adicional
```

Una aproximación más robusta puede incorporar la variabilidad histórica:

```text
stock_seguridad = z * desviación_estándar_consumo_diario * raíz(lead_time)
```

Donde `z` representa un nivel de servicio objetivo.

### 8.5 Punto de pedido

El punto de pedido puede estimarse como:

```text
punto_pedido = consumo_durante_lead_time + stock_seguridad
```

Si el stock actual es menor o igual al punto de pedido, se recomienda emitir una orden de reposición.

```text
si stock_actual <= punto_pedido:
    recomendar_reabastecimiento
```

### 8.6 Cantidad sugerida de pedido

La cantidad sugerida de pedido puede estimarse como:

```text
cantidad_pedido = consumo_esperado_horizonte + stock_seguridad - stock_actual
```

Restringiendo el resultado a valores no negativos:

```text
cantidad_pedido = max(0, cantidad_pedido)
```

### 8.7 Priorización logística

No todos los medicamentos deberían tratarse igual. Para priorizar decisiones de abastecimiento, se recomienda incorporar criterios adicionales:

- línea terapéutica;
- criticidad clínica;
- canasta vigente;
- variabilidad del consumo;
- riesgo de quiebre;
- historial de entradas y salidas;
- tiempo de reposición;
- existencia de productos sustitutos.

En particular, productos de alta criticidad o con baja posibilidad de sustitución deberían tener mayores niveles de stock de seguridad.

---

## 9. Limitaciones de la solución

La solución implementada prioriza funcionalidad, trazabilidad y cumplimiento de la interfaz solicitada por el challenge.

Entre sus principales limitaciones se encuentran:

1. El modelo usa principalmente variables temporales y de producto.
2. No se incorporan explícitamente variables externas como estacionalidad epidemiológica, campañas sanitarias, cambios de proveedor o restricciones presupuestarias.
3. El stock diario se analiza conceptualmente para reabastecimiento, pero no se integra directamente como variable predictiva del modelo actual.
4. No se modelan explícitamente quiebres de stock, lead time real ni órdenes de compra.
5. El modelo puede mejorarse con validación temporal, comparación de algoritmos y optimización de hiperparámetros.

---

## 10. Mejoras futuras

Como mejoras futuras se proponen:

1. Integrar `stock.csv` directamente como variable explicativa.
2. Construir variables de rezago de consumo por producto.
3. Incorporar promedios móviles de consumo.
4. Evaluar modelos adicionales, como Gradient Boosting o XGBoost.
5. Implementar validación temporal en vez de división aleatoria.
6. Estimar riesgo de quiebre de stock por producto.
7. Calcular automáticamente punto de pedido y cantidad sugerida de reposición.
8. Incorporar niveles de servicio diferenciados según criticidad terapéutica.
9. Monitorear drift de demanda y degradación del modelo.
10. Incorporar logging y métricas operacionales de la API.

---

## 11. Resumen

La solución implementa una API funcional para predecir consumo de medicamentos a partir de datos históricos. El modelo se entrena con movimientos de salida, genera variables temporales y entrega predicciones de cantidad consumida.

La API permite consultar predicciones por producto y fecha, e incorpora validaciones para productos inexistentes y fechas inválidas.

Desde el punto de vista logístico, la predicción puede ser usada como insumo para definir puntos de pedido, stock de seguridad y cantidades sugeridas de reabastecimiento, reduciendo el riesgo de quiebre y mejorando la planificación de inventario.
## Despliegue en Cloud Run

La API fue desplegada en Google Cloud Run utilizando una imagen Docker construida localmente y subida a Artifact Registry.

URL de la API desplegada:

```text
https://cenabast-challenge-823361295040.southamerica-west1.run.app
```

Validaciones realizadas:

- `GET /health`: respuesta `200 OK`.
- `POST /predict`: respuesta `200 OK` con predicción para GTIN y fecha válidos.
- Validaciones de entrada implementadas para productos desconocidos y fechas inválidas.

## Stress test

Se ejecutó un stress test contra la API desplegada en Cloud Run usando Locust.

Configuración utilizada:

- Usuarios: 100
- Tiempo de ejecución: 60 segundos
- Endpoint: `/predict`
- URL base: `https://cenabast-challenge-823361295040.southamerica-west1.run.app`

Resultado resumido:

- Requests totales: 1647
- Fallas: 0
- Tasa de fallas: 0.00%
- Tiempo promedio de respuesta: 1090 ms
- Tiempo mínimo: 45 ms
- Tiempo máximo: 2999 ms
- Mediana: 1000 ms

El reporte HTML fue generado localmente en:

```text
reports/stress-test.html
```