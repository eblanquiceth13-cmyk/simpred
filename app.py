"""
SIMPRED — Tablero de mantenimiento predictivo con IA explicable
Entrega 2 · Implementación — Parte 2: tablero web

Ejecutar:  streamlit run app.py

Las partes marcadas con  # 🔎 TRANSPARENCIA  son los puntos donde se aplica el
principio ético priorizado (transparencia y explicación).
"""
import json
from datetime import datetime
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

CARPETA = Path(__file__).parent
ARCHIVO_MODELO = CARPETA / "simpred_modelo.joblib"
ARCHIVO_FICHA = CARPETA / "ficha_modelo.json"
ARCHIVO_EJEMPLO = CARPETA / "lecturas_ejemplo.csv"
ARCHIVO_REGISTRO = CARPETA / "registro_decisiones.csv"

COLOR_AUMENTA, COLOR_REDUCE = "#eb6834", "#2a78d6"
COLUMNAS_LECTURA = ["equipo", "tipo", "temp_aire", "temp_proceso", "velocidad", "torque", "desgaste"]

# Rangos (mín–máx) observados en el dataset AI4I 2020. Fuera de ellos el modelo no tiene experiencia.
RANGOS = {"temp_aire": (295.3, 304.5), "temp_proceso": (305.7, 313.8), "velocidad": (1168, 2886),
          "torque": (3.8, 76.6), "desgaste": (0, 253)}

# 🔎 TRANSPARENCIA: aviso de limitaciones visible en cada resultado
AVISO = ("Este resultado es un apoyo, no una orden. El modelo se entrenó con datos simulados (AI4I 2020) "
         "y puede equivocarse. La decisión de intervenir la toma el técnico.")

NIVELES = {
    "ALTO": ("🔴", "Intervenir: revisar el equipo cuanto antes."),
    "MEDIO": ("🟠", "Vigilar: revisarlo en la próxima ronda de mantenimiento."),
    "BAJO": ("🟢", "Operación normal: seguir monitoreando."),
}

st.set_page_config(page_title="SIMPRED", page_icon="🛠️", layout="wide")


# ───────────────────────────── Carga del modelo ─────────────────────────────
ARCHIVO_XGB = CARPETA / "simpred_modelo_xgb.json"
ARCHIVO_CONFIG = CARPETA / "simpred_config.json"


@st.cache_resource
def cargar_modelo():
    ficha = json.loads(ARCHIVO_FICHA.read_text(encoding="utf-8"))
    if ARCHIVO_XGB.exists():
        # Formato nativo de XGBoost: funciona igual en Windows, Linux y Mac
        modelo = XGBClassifier()
        modelo.load_model(ARCHIVO_XGB)
        paquete = {"pipeline": Pipeline([("modelo", modelo)]),
                   **json.loads(ARCHIVO_CONFIG.read_text(encoding="utf-8"))}
    else:  # Random Forest o regresión logística guardados con joblib
        paquete = joblib.load(ARCHIVO_MODELO)
    explicador = shap.TreeExplainer(paquete["pipeline"].named_steps["modelo"])
    return paquete, ficha, explicador


try:
    PAQUETE, FICHA, EXPLICADOR = cargar_modelo()
except FileNotFoundError:
    st.error("Faltan archivos del modelo (`simpred_modelo_xgb.json`, `simpred_config.json`, `ficha_modelo.json`). "
             "Cópialos junto a `app.py` (se generan con el notebook de la Parte 1).")
    st.stop()
except Exception as e:  # versiones de librerías distintas, archivo dañado, etc.
    st.error(f"No se pudo cargar el modelo: {e}. Revisa que las versiones de `requirements.txt` estén instaladas.")
    st.stop()

PIPE = PAQUETE["pipeline"]
VARIABLES = PAQUETE["variables"]
ETIQUETAS = PAQUETE["etiquetas"]
UMBRAL, UMBRAL_MEDIO = PAQUETE["umbral"], PAQUETE["umbral_medio"]


# ───────────────────────────── Lógica del modelo ─────────────────────────────
def preparar(lecturas):
    """Igual que en el notebook: crea las variables que usa el modelo."""
    d = lecturas.copy()
    d["tipo"] = d["tipo"].map({"L": 0, "M": 1, "H": 2})
    d["dif_temp"] = d["temp_proceso"] - d["temp_aire"]
    d["potencia"] = d["torque"] * d["velocidad"] * 2 * np.pi / 60
    d["esfuerzo"] = d["desgaste"] * d["torque"]
    return d[VARIABLES].astype(float)


def transformar(X):
    """Pasos previos al modelo (escalado si lo hay). SMOTE solo actúa al entrenar."""
    for _, paso in PIPE.steps[:-1]:
        if hasattr(paso, "transform"):
            X = paso.transform(X)
    return X


def predecir(lecturas):
    """Devuelve variables, probabilidad de falla y valores SHAP de cada lectura."""
    X = preparar(lecturas)
    prob = PIPE.predict_proba(X)[:, 1]
    valores = EXPLICADOR(transformar(X)).values
    if valores.ndim == 3:  # Random Forest devuelve una columna por clase
        valores = valores[:, :, 1]
    return X, prob, valores


def nivel_riesgo(p):
    return "ALTO" if p >= UMBRAL else "MEDIO" if p >= UMBRAL_MEDIO else "BAJO"


def formato(var, valor):
    if var == "tipo":
        return {0: "L", 1: "M", 2: "H"}[int(valor)]
    decimales = 1 if var in ("temp_aire", "temp_proceso", "torque", "dif_temp") else 0
    return f"{valor:,.{decimales}f} {ETIQUETAS[var][1]}".strip()


def factores(x_fila, shap_fila, n=None):
    """🔎 TRANSPARENCIA: traduce los valores SHAP a frases que entiende un técnico."""
    orden = np.argsort(-np.abs(shap_fila))[:n]
    return [{"var": VARIABLES[j], "Variable": ETIQUETAS[VARIABLES[j]][0], "Valor": formato(VARIABLES[j], x_fila[j]),
             "Contribución": float(shap_fila[j]),
             "Efecto": "Aumenta el riesgo" if shap_fila[j] > 0 else "Reduce el riesgo"} for j in orden]


def revisar_lectura(lectura):
    """Manejo de errores: avisa cuando la lectura está fuera de lo que el modelo conoce."""
    avisos = []
    for var, (lo, hi) in RANGOS.items():
        if not lo <= lectura[var] <= hi:
            avisos.append(f"{ETIQUETAS[var][0]} = {formato(var, lectura[var])} está fuera del rango de entrenamiento "
                          f"({lo:g}–{hi:g} {ETIQUETAS[var][1]}). La predicción es menos confiable.")
    if lectura["temp_proceso"] <= lectura["temp_aire"]:
        avisos.append("La temperatura del proceso es menor o igual a la del aire: revisa la lectura del sensor.")
    return avisos


# ───────────────────────────── Componentes visuales ─────────────────────────────
def grafico_factores(lista):
    datos = pd.DataFrame(lista)
    datos["Etiqueta"] = datos["Variable"] + " = " + datos["Valor"]
    return (
        alt.Chart(datos)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("Contribución:Q", title="Influencia en el riesgo (SHAP)"),
            y=alt.Y("Etiqueta:N", sort=None, title=None, scale=alt.Scale(paddingInner=0.35), axis=alt.Axis(labelLimit=320, labelOverlap=False, labelFontSize=12)),
            color=alt.Color("Efecto:N", scale=alt.Scale(domain=["Aumenta el riesgo", "Reduce el riesgo"],
                                                        range=[COLOR_AUMENTA, COLOR_REDUCE]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["Variable", "Valor", "Efecto", alt.Tooltip("Contribución:Q", format="+.2f")],
        )
        .properties(height=alt.Step(32))
    )


def mostrar_resultado(equipo, p, lista, avisos):
    nivel = nivel_riesgo(p)
    icono, accion = NIVELES[nivel]
    c1, c2 = st.columns([1, 2])
    c1.metric("Riesgo de falla", f"{p:.1%}")
    caja = {"ALTO": c2.error, "MEDIO": c2.warning, "BAJO": c2.success}[nivel]
    caja(f"{icono} **Nivel {nivel}** — {accion}")

    for a in avisos:
        st.warning(f"⚠️ {a}")

    # 🔎 TRANSPARENCIA: explicación de la predicción en lenguaje sencillo
    st.markdown(f"**¿Por qué {equipo} tiene este riesgo?** Los factores que más influyeron:")
    for f in lista[:3]:
        flecha = "⬆️" if f["Contribución"] > 0 else "⬇️"
        st.markdown(f"- {flecha} **{f['Variable']}** = {f['Valor']} → {f['Efecto'].lower()}")
    with st.expander("Ver la influencia de todas las variables"):
        st.altair_chart(grafico_factores(lista), width="stretch")
        st.caption("El largo de cada barra indica cuánto influyó la variable en esta predicción. "
                   "SHAP muestra en qué se basó el modelo; no prueba la causa física de la falla.")
    st.info(f"ℹ️ {AVISO}")


def guardar_decision(fila):
    """🔎 TRANSPARENCIA: registro auditable de cada predicción y de la decisión humana."""
    nueva = pd.DataFrame([fila])
    if ARCHIVO_REGISTRO.exists():
        nueva = pd.concat([pd.read_csv(ARCHIVO_REGISTRO), nueva], ignore_index=True)
    nueva.to_csv(ARCHIVO_REGISTRO, index=False)


# ───────────────────────────── Interfaz ─────────────────────────────
st.title("🛠️ SIMPRED — Mantenimiento predictivo con IA explicable")
st.caption("Estima el riesgo de falla de una máquina a partir de sus variables operativas y explica por qué. "
           "Apoya al técnico; no lo reemplaza.")

met = FICHA["metricas_prueba"]
err = FICHA["errores_prueba"]
total_fallas = err["fallas_detectadas"] + err["fallas_no_detectadas"]

with st.sidebar:  # 🔎 TRANSPARENCIA: qué modelo es, qué tan bien funciona y sus límites, siempre visibles
    st.header("Sobre el sistema")
    st.markdown(f"**Modelo:** {FICHA['modelo']} · v{FICHA['version']} ({FICHA['fecha']})  \n"
                f"**Datos:** AI4I 2020, simulados ({FICHA['datos']['registros']:,} registros)")
    st.markdown(f"**En pruebas:**  \n"
                f"- Detectó **{err['fallas_detectadas']} de {total_fallas}** fallas (recall {met['Recall']:.0%}).  \n"
                f"- De cada 100 alertas, unas **{met['Precisión'] * 100:.0f}** fueron fallas reales.  \n"
                f"- Dejó pasar **{err['fallas_no_detectadas']}** fallas.")
    st.markdown(f"**Niveles de riesgo:**  \n🔴 Alto ≥ {UMBRAL:.0%} · 🟠 Medio {UMBRAL_MEDIO:.0%}–{UMBRAL:.0%} · "
                f"🟢 Bajo < {UMBRAL_MEDIO:.0%}")
    st.warning(AVISO)
    st.caption("Detalles completos en la pestaña **Ficha del modelo**.")

tab_eval, tab_panel, tab_registro, tab_ficha = st.tabs(
    ["🔍 Evaluar máquina", "🏭 Panel de equipos", "📋 Registro de decisiones", "📄 Ficha del modelo"])

# ── Pestaña 1: evaluar una máquina ──
with tab_eval:
    col_form, col_res = st.columns([1, 1.5], gap="large")
    with col_form:
        with st.form("lectura"):
            st.subheader("Lectura de la máquina")
            equipo = st.text_input("Equipo", "Fresadora-01")
            tipo = st.selectbox("Calidad del producto", ["L", "M", "H"],
                                format_func=lambda t: {"L": "L — baja", "M": "M — media", "H": "H — alta"}[t])
            ayuda = lambda v: f"Rango de entrenamiento: {RANGOS[v][0]:g}–{RANGOS[v][1]:g} {ETIQUETAS[v][1]}"
            temp_aire = st.number_input("Temperatura del aire [K]", 280.0, 330.0, 300.0, 0.1, help=ayuda("temp_aire"))
            temp_proceso = st.number_input("Temperatura del proceso [K]", 290.0, 340.0, 310.0, 0.1,
                                           help=ayuda("temp_proceso"))
            velocidad = st.number_input("Velocidad [rpm]", 500, 4000, 1500, 10, help=ayuda("velocidad"))
            torque = st.number_input("Torque [Nm]", 0.0, 120.0, 40.0, 0.5, help=ayuda("torque"))
            desgaste = st.number_input("Desgaste de la herramienta [min]", 0, 400, 100, 1, help=ayuda("desgaste"))
            enviado = st.form_submit_button("Evaluar riesgo", type="primary", width="stretch")

    if enviado:
        lectura = {"equipo": equipo.strip() or "Equipo sin nombre", "tipo": tipo, "temp_aire": temp_aire,
                   "temp_proceso": temp_proceso, "velocidad": velocidad, "torque": torque, "desgaste": desgaste}
        X, prob, valores = predecir(pd.DataFrame([lectura]))
        st.session_state["resultado"] = {"lectura": lectura, "p": float(prob[0]),
                                         "factores": factores(X.iloc[0].values, valores[0]),
                                         "avisos": revisar_lectura(lectura)}

    with col_res:
        r = st.session_state.get("resultado")
        if r is None:
            st.info("Ingresa la lectura de una máquina y presiona **Evaluar riesgo**.")
        else:
            st.subheader(f"Resultado — {r['lectura']['equipo']}")
            mostrar_resultado(r["lectura"]["equipo"], r["p"], r["factores"], r["avisos"])

            # 🔎 TRANSPARENCIA: validación humana antes de cualquier intervención
            st.subheader("✅ Decisión del técnico")
            with st.form("decision", clear_on_submit=True):
                tecnico = st.text_input("Nombre del técnico *")
                decision = st.radio("¿Qué decides?", ["Intervenir el equipo", "Programar revisión",
                                                      "Descartar la alerta (no hay problema)"])
                util = st.radio("¿La explicación te ayudó a entender el riesgo?", ["Sí", "Parcialmente", "No"],
                                horizontal=True)
                comentario = st.text_area("Comentario (opcional)", placeholder="Qué revisaste y qué encontraste")
                guardar = st.form_submit_button("Guardar decisión")
            if guardar:
                if not tecnico.strip():
                    st.error("Escribe el nombre del técnico para guardar la decisión.")
                else:
                    l = r["lectura"]
                    guardar_decision({
                        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), **l,
                        "riesgo_%": round(r["p"] * 100, 1), "nivel": nivel_riesgo(r["p"]),
                        "factores_mostrados": "; ".join(f"{f['Variable']} {f['Efecto'].lower()}"
                                                        for f in r["factores"][:3]),
                        "fuera_de_rango": "sí" if r["avisos"] else "no",
                        "decision": decision, "explicacion_util": util, "tecnico": tecnico.strip(),
                        "comentario": comentario.strip(), "version_modelo": FICHA["version"]})
                    st.success("Decisión guardada en el registro.")

# ── Pestaña 2: panel con varios equipos ──
with tab_panel:
    st.subheader("Riesgo de cada equipo")
    st.caption("Sube un CSV con las columnas `" + "`, `".join(COLUMNAS_LECTURA) + "` o usa las lecturas de ejemplo.")
    archivo = st.file_uploader("Lecturas de los equipos (CSV)", type="csv")
    try:
        lecturas = pd.read_csv(archivo) if archivo else pd.read_csv(ARCHIVO_EJEMPLO)
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        lecturas = None

    if lecturas is not None:
        faltan = [c for c in COLUMNAS_LECTURA if c not in lecturas.columns]
        if faltan:
            st.error(f"Al archivo le faltan estas columnas: {', '.join(faltan)}")
        else:
            lecturas = lecturas[COLUMNAS_LECTURA].copy()
            lecturas["tipo"] = lecturas["tipo"].astype(str).str.strip().str.upper()
            for c in COLUMNAS_LECTURA[2:]:
                lecturas[c] = pd.to_numeric(lecturas[c], errors="coerce")
            invalidas = lecturas.isna().any(axis=1) | ~lecturas["tipo"].isin(["L", "M", "H"])
            if invalidas.any():
                st.warning(f"Se omitieron {int(invalidas.sum())} filas con datos vacíos o inválidos.")
            lecturas = lecturas[~invalidas].reset_index(drop=True)

            if lecturas.empty:
                st.info("No hay lecturas válidas para evaluar.")
            else:
                X, prob, valores = predecir(lecturas)
                niveles = [nivel_riesgo(p) for p in prob]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Equipos evaluados", len(lecturas))
                m2.metric("🔴 Riesgo alto", niveles.count("ALTO"))
                m3.metric("🟠 Riesgo medio", niveles.count("MEDIO"))
                m4.metric("🟢 Riesgo bajo", niveles.count("BAJO"))

                tabla = pd.DataFrame({
                    "Equipo": lecturas["equipo"], "Calidad": lecturas["tipo"], "Riesgo": prob * 100,
                    "Nivel": [f"{NIVELES[n][0]} {n}" for n in niveles],
                    "Factor principal": [("⬆️ " if f["Contribución"] > 0 else "⬇️ ") + f["Variable"]
                                         for f in (factores(X.iloc[i].values, valores[i], 1)[0] for i in range(len(X)))],
                    "Fuera de rango": ["⚠️ sí" if revisar_lectura(lecturas.iloc[i]) else "no"
                                       for i in range(len(X))],
                }).sort_values("Riesgo", ascending=False)
                st.dataframe(tabla, hide_index=True, width="stretch", column_config={
                    "Riesgo": st.column_config.ProgressColumn("Riesgo de falla", format="%.1f %%",
                                                              min_value=0, max_value=100)})

                st.caption("Factor principal: la variable que más influyó (⬆️ sube el riesgo, ⬇️ lo baja).")
                st.markdown("#### Explicación por equipo")
                orden = tabla.index.tolist()
                i = st.selectbox("Equipo", orden, format_func=lambda k: lecturas.loc[k, "equipo"])
                mostrar_resultado(lecturas.loc[i, "equipo"], float(prob[i]),
                                  factores(X.loc[i].values, valores[i]), revisar_lectura(lecturas.loc[i]))
                st.caption("Para registrar una decisión sobre un equipo, evalúalo en la pestaña **Evaluar máquina**.")

# ── Pestaña 3: registro de decisiones ──
with tab_registro:
    st.subheader("Registro de decisiones")
    st.caption("Cada decisión queda guardada con la predicción, la explicación mostrada y quién decidió. "
               "Así se puede auditar el sistema y detectar sus errores.")
    if ARCHIVO_REGISTRO.exists():
        registro = pd.read_csv(ARCHIVO_REGISTRO)
        alertas = registro[registro["nivel"] == "ALTO"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Decisiones registradas", len(registro))
        c2.metric("Alertas altas descartadas por el técnico",
                  int(alertas["decision"].str.startswith("Descartar").sum()) if len(alertas) else 0)
        c3.metric("Explicaciones útiles", f"{(registro['explicacion_util'] == 'Sí').mean():.0%}")
        st.dataframe(registro.iloc[::-1], hide_index=True, width="stretch")
        st.download_button("Descargar registro (CSV)", registro.to_csv(index=False).encode("utf-8"),
                           "registro_decisiones.csv", "text/csv")
    else:
        st.info("Aún no hay decisiones registradas. Evalúa una máquina y guarda la decisión del técnico.")

# ── Pestaña 4: ficha del modelo ──
with tab_ficha:  # 🔎 TRANSPARENCIA: información adaptada a cada audiencia
    st.subheader(FICHA["nombre"])
    st.markdown(f"**Uso previsto:** {FICHA['uso_previsto']}")

    a_tec, a_jefe, a_dev = st.tabs(["👷 Para técnicos", "📊 Para jefes de mantenimiento", "💻 Para desarrolladores"])
    with a_tec:
        st.markdown(f"""
- El sistema calcula el **riesgo de falla** (0–100 %) con la temperatura, la velocidad, el torque y el desgaste.
- 🔴 **Alto** (≥ {UMBRAL:.0%}): revisar el equipo cuanto antes. 🟠 **Medio**: vigilar. 🟢 **Bajo**: operación normal.
- Cada resultado muestra **qué variables subieron o bajaron el riesgo**. Compruébalas en la máquina antes de actuar.
- **Tú decides.** Si revisas y no hay problema, descarta la alerta y escribe por qué: eso ayuda a mejorar el sistema.
- Variables derivadas: *diferencia de temperatura* (proceso − aire), *potencia* (torque × velocidad) y
  *esfuerzo* (desgaste × torque).
""")
    with a_jefe:
        st.markdown(f"En el conjunto de prueba ({total_fallas} fallas reales), el modelo **detectó "
                    f"{err['fallas_detectadas']}**, **dejó pasar {err['fallas_no_detectadas']}** y dio "
                    f"**{err['falsas_alarmas']} falsas alarmas**.")
        st.markdown("**Comparación de modelos (datos de prueba)**")
        st.dataframe(pd.DataFrame(FICHA["comparacion_modelos_prueba"]).set_index("Modelo"),
                     width="stretch")
        st.markdown("**Desempeño por calidad del producto**")
        st.dataframe(pd.DataFrame(FICHA["metricas_por_calidad"]).set_index("Calidad"), width="stretch")
        st.caption("Con tan pocas fallas en algunos grupos (sobre todo H), sus métricas son poco confiables.")
    with a_dev:
        d = FICHA["datos"]
        st.markdown(f"""
- **Datos:** {d['fuente']} · DOI [{d['doi']}](https://doi.org/{d['doi']}) · licencia {d['licencia']} · {d['tipo']}
  · {d['registros']:,} registros · {d['porcentaje_fallas']} % de fallas.
- **Modelo:** {FICHA['modelo']} · balanceo: {FICHA['balanceo']} · umbrales: alto {UMBRAL}, medio {UMBRAL_MEDIO}.
- **Métricas de validación cruzada:** {', '.join(f'{k} {v}' for k, v in FICHA['metricas_validacion_cruzada'].items())}.
- **Métricas de prueba:** {', '.join(f'{k} {v}' for k, v in met.items())}.
- **Versiones:** {', '.join(f'{k} {v}' for k, v in FICHA['versiones'].items())}.
""")
        imp = pd.DataFrame({"Variable": list(FICHA["importancia_global_shap"]),
                            "Importancia": list(FICHA["importancia_global_shap"].values())})
        st.markdown("**Importancia global de las variables (promedio |SHAP|)**")
        st.altair_chart(alt.Chart(imp).mark_bar(cornerRadiusEnd=4, height=16, color=COLOR_REDUCE).encode(
            x=alt.X("Importancia:Q", title="Promedio |SHAP|"),
            y=alt.Y("Variable:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            tooltip=["Variable", alt.Tooltip("Importancia:Q", format=".3f")]).properties(height=300),
            width="stretch")
        with st.expander("Ficha completa (JSON)"):
            st.json(FICHA)

    st.markdown("#### Limitaciones")
    for lim in FICHA["limitaciones"]:
        st.markdown(f"- {lim}")

st.divider()
st.caption("SIMPRED v1.0 · Proyecto académico — Universidad Simón Bolívar · Inteligencia Artificial")
