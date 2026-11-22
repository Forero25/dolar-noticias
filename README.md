# ¿Las noticias ayudan a pronosticar el dólar en Colombia?

Este repositorio pronostica la TRM (la tasa de cambio oficial del peso colombiano frente al dólar) a 1, 5 y 20 días hábiles, y responde una pregunta concreta: **¿las noticias le agregan algo a un pronóstico hecho con variables de mercado?**

El punto de partida es incómodo y conocido: desde Meese y Rogoff (1983), casi ningún modelo le gana de forma consistente al paseo aleatorio, es decir, a pronosticar que mañana la tasa será igual a la de hoy. Todo aquí se mide contra ese rival.

## Datos

| Fuente | Qué | Desde |
|---|---|---|
| [datos.gov.co](https://www.datos.gov.co/Econom-a-y-Finanzas/Tasa-de-Cambio-Representativa-del-Mercado-TRM/32sa-8pi3) (Superintendencia Financiera) | TRM diaria | 1991 |
| [FRED](https://fred.stlouisfed.org/) | Brent, dólar amplio, Tesoro a 2 y 10 años, VIX, tasa de la Fed, inflación esperada, peso mexicano, real brasileño | 2016 |
| Yahoo Finance | Peso chileno, sol peruano, bonos emergentes (EMB) | 2016 |
| [GDELT DOC 2.0](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) | Tono y volumen diario de noticias en 12 temas, en 65 idiomas traducidos al inglés, prensa colombiana incluida | 2017 |

La TRM que rige un día se calcula con las operaciones del día hábil anterior; aquí cada valor se fecha el día de las operaciones, para que el modelo nunca use información que no se conocía.

Temas de noticias: peso colombiano, Banco de la República, déficit y reforma tributaria, calificación crediticia, petróleo y Ecopetrol, gobierno, orden público, prensa colombiana sobre economía y sobre el dólar, la Reserva Federal, precio del petróleo y mercados emergentes. Para cada tema: tono de 7 días, su desviación frente a los 90 días previos y la participación del tema en el total de noticias.

## Método

- **Origen móvil:** desde 2020, el modelo se reentrena cada 20 días hábiles solo con resultados ya conocidos en la fecha del pronóstico.
- **Modelos:** paseo aleatorio, media histórica, regresión ridge y gradient boosting; cada uno con tres conjuntos de variables: solo la TRM, más mercado y más noticias.
- **Pruebas:** Diebold-Mariano con corrección de Harvey, Leybourne y Newbold frente al paseo aleatorio y de "con noticias" frente a "sin noticias"; Pesaran-Timmermann para la dirección; intervalos conformales del 90 % y su cobertura real; errores por año.
- **Choques de noticias:** proyecciones locales (Jordà, 2005) del cambio de la TRM en los 20 días siguientes a un tono una desviación estándar más negativo que lo normal, con errores de Newey-West.
- **Validación del código:** pruebas con datos sintéticos donde la respuesta se conoce. Con una señal de noticias plantada, el modelo la encuentra; con ruido puro, ningún modelo le gana al paseo aleatorio; los intervalos del 90 % cubren cerca del 90 %.

GitHub Actions baja los datos y corre todo cada día hábil (`.github/workflows/datos.yml`). Los resultados quedan en `resultados/evaluacion.json`.

## Lo que este proyecto no hace

- **No es una recomendación de inversión ni de cobertura cambiaria.** Mide qué tan bien se habría pronosticado en el pasado.
- **No promete ganarle al paseo aleatorio.** Si las noticias no ayudan, el resultado lo dice.
- **No lee los artículos completos en la historia.** Antes de los últimos tres meses usa el tono y el volumen que calcula GDELT, no texto propio.
- **No publica precios de mercado.** Solo la TRM (dato abierto oficial) y las series de noticias de GDELT quedan en el repositorio; el resto se descarga en cada corrida.
- **No explica causas.** Un tono negativo que antecede a una devaluación puede reflejar lo que ya estaba pasando.

## Reproducir

```
pip install -r requirements.txt
pytest
python ingesta/trm.py && python ingesta/mercados.py && python ingesta/noticias.py
python analisis/correr.py
```

## Licencias

TRM: datos abiertos del Gobierno de Colombia. Noticias: The GDELT Project, uso libre con atribución. FRED: Federal Reserve Bank of St. Louis. Código bajo licencia MIT.
