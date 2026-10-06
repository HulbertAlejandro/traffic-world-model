"""Genera docs/RESULTADOS_PARA_ARTICULO.md solo a partir de los archivos de resultados versionados.

Solo lectura: no simula, no entrena y no modifica ningún resultado. Depende solo de json, csv, glob y
numpy (sin pandas ni SUMO). Las pruebas estadísticas que calcula son las del proyecto, que son numpy puro:
`paired_t` de `scripts/evaluate_multiseed_statistical.py` y el Wilcoxon exacto de
`docs/results/ppo_10_seeds/analyze.py`.

Uso: python scripts/v2/make_article_report.py
"""
import datetime
import glob
import importlib.util
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_multiseed_statistical import paired_t  # noqa: E402

_spec = importlib.util.spec_from_file_location("ppo10_analyze", ROOT / "docs" / "results" / "ppo_10_seeds" / "analyze.py")
_analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_analyze)
wilcoxon = _analyze.wilcoxon

R = ROOT / "docs" / "results" / "v2"
S = ("A0", "B0", "C0", "D0")
out = []
w = out.append


def J(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def f(x, d=1):
    if x is None:
        return "—"
    s = f"{x:,.{d}f}"
    return s.replace("-", "−")


def pv(p):
    if p is None:
        return "—"
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}".replace("e-0", "e-").replace("-", "−")


def sg(s):
    """Signo menos tipográfico en una cifra ya formateada."""
    return s.replace("-", "−")


def src(*paths):
    return "Fuente: " + ", ".join(f"`{p}`" for p in paths) + "."


w("# Resultados para el artículo (v2, corredor de 4 intersecciones)")
w("")
w(f"Informe de **solo lectura**, generado el {datetime.date.today().isoformat()} con `scripts/v2/make_article_report.py` a partir de los archivos de resultados "
  "versionados de la rama `v2/four-intersections`. No se simuló ni se entrenó nada para producirlo, y no "
  "modifica ningún resultado. Cada tabla indica el archivo de donde sale. Los valores calculados aquí (por "
  "ejemplo, las pruebas por intersección de la sección 1) usan las funciones estadísticas del proyecto "
  "(`paired_t` de `scripts/evaluate_multiseed_statistical.py` y el Wilcoxon de "
  "`docs/results/ppo_10_seeds/analyze.py`). Convención: retorno = suma de la recompensa por paso; más cerca "
  "de 0 es mejor. El signo de cada diferencia se indica en su tabla.")
w("")

# ------------------------------------------------------------------ 1. Escenario
cal_p = "docs/results/v2/demand_calibration/it5_offset_300s.json"
fail_p = "docs/results/v2/demand_calibration/analysis_reference_failures_it5_it6_it7.json"
cal = J(cal_p)
spec = cal["spec"]
w("## 1. Escenario")
w("")
w("**Red.** Corredor de 4 intersecciones en línea (A0, B0, C0, D0, de oeste a este) sobre una arterial "
  "Este–Oeste, cada una con su calle transversal Norte–Sur; un carril por sentido, 13.89 m/s; parámetros de "
  "SUMO y sumo-rl iguales a la v1 (`delta_time` 5 s, amarillo 2 s, verde mínimo 5 s, episodios de 300 s = 60 "
  "pasos de control). Fuente: `docs/v2/DISENO_RED_4_INTERSECCIONES.md`, secciones 1 y 2.")
w("")
w(f"**Demanda final (candidata `{cal['candidate']}`), en veh/h.** Llegadas `{spec['arrivals']}`. Patrón "
  "pulsado de dos mitades de 150 s que se repite cada 300 s. " + src(cal_p, "docs/v2/DISENO_RED_4_INTERSECCIONES.md (3.4)"))
w("")
w("| Segmento | Arterial desde el oeste (→E) | Arterial desde el este (→O) | A0 N/S | B0 N/S | C0 N/S | D0 N/S |")
w("|---|---|---|---|---|---|---|")
t0 = 0
for seg in spec["segments"]:
    c = seg["cross"]
    w(f"| {t0}–{t0 + seg['length']} s | {seg['arterial_west']} | {seg['arterial_east']} | "
      + " | ".join(f"{c[s][0]}/{c[s][1]}" for s in S) + " |")
    t0 += seg["length"]
c = spec["cross"]
w(f"| **Media** | {spec['arterial_west']} | {spec['arterial_east']} | " + " | ".join(f"{c[s][0]}/{c[s][1]}" for s in S) + " |")
w("")
w(f"- Demanda total media: {f(cal['total_demand_veh_h'], 0)} veh/h. Giros desde la arterial: "
  f"{spec['arterial_turn_off'] * 100:.2f}% hacia el Norte y otro tanto hacia el Sur en cada intersección; en las "
  f"transversales, {spec['cross_straight'] * 100:.0f}% sigue recto y el resto gira hacia la arterial, mitad a cada lado.")
w("- **Desfase del pulso:** cada episodio arranca el patrón en `offset = default_rng(semilla).integers(0, 300)`; "
  "la demanda en el instante t es la del patrón en (t + offset) mod 300 (`scripts/v2/corridor_demand.py`, "
  "`pulse_offset`; `DISENO_RED_4_INTERSECCIONES.md`, 7.1). Solo C0 y la arterial varían entre mitades; A0, B0 y "
  "D0 tienen transversal constante.")
seg_c0 = [seg["cross"]["C0"] for seg in spec["segments"]]
w(f"- **Transversal de C0 (pulsada, N/S):** dos niveles de {spec['segments'][0]['length']} s cada uno, "
  + " y ".join(f"{a}/{b}" for a, b in seg_c0)
  + f" veh/h; su media es {c['C0'][0]}/{c['C0'][1]} veh/h, que es la cifra de la fila «Media» y **no** un flujo "
  "que ocurra en ningún momento del episodio. En A0 ("
  + f"{c['A0'][0]}/{c['A0'][1]}), B0 ({c['B0'][0]}/{c['B0'][1]}) y D0 ({c['D0'][0]}/{c['D0'][1]}) el flujo es constante. "
  + src(cal_p) + " (`spec.segments[].cross.C0` y `spec.cross.C0`)")
w("")
w("**Criterios de calibración** (fijados antes de la primera candidata; 20 semillas de calibración 11000–11019; "
  f"`DISENO_RED_4_INTERSECCIONES.md`, 3.1). Valores en `{cal_p}` → `criteria`:")
w("")
cr = cal["criteria"]
w(f"1. Sin cola invisible: con la mejor referencia, como máximo {cr['max_pending_best']} vehículos pendientes de "
  "inserción después de los primeros 30 s, en todos los episodios.")
w(f"2. Demora no trivial: la mejor referencia le gana a **cada** política trivial con reducción media pareada "
  f"≥ {cr['min_reduction'] * 100:.0f}% y p < {cr['max_p_value']} (t pareada y Wilcoxon).")
w(f"3. Recompensa estilo v1 no trivial: mejora frente a cada trivial con p < {cr['max_p_value']} (t pareada).")
w("")
an = cal["analysis"]
bvt = an["best_reference_vs_best_trivial"]
w(f"**Resultado agregado con desfase aleatorio** (mejor referencia `{an['best_reference']}` frente a la mejor "
  f"trivial `{an['best_trivial_by_reward_v1_style']}`, 20 semillas). " + src(cal_p))
w("")
w("| Métrica | Reducción media pareada | t pareada p | Wilcoxon p | Semillas ganadas | ¿Pasa? |")
w("|---|---|---|---|---|---|")
for m, label, ok in (("delay", "Demora", an["checks"]["beats_every_trivial_delay"]),
                     ("reward_v1_style", "Recompensa estilo v1", an["checks"]["beats_every_trivial_reward_v1_style"])):
    x = bvt[m]
    w(f"| {label} | {x['reduction'] * 100:+.1f}% | {pv(x['t_p'])} | {pv(x['wilcoxon_p'])} | {x['ref_wins']}/20 | {'sí' if ok else 'no'} |")
w("")
mp = an["summary"][an["best_reference"]].get("max_pending_after_warmup_max")
w(f"- Cola invisible: máximo de pendientes después de 30 s con `{an['best_reference']}` = {mp} "
  f"(`summary.{an['best_reference']}.max_pending_after_warmup_max`); criterio 1 cumplido: "
  f"{'sí' if an['checks']['no_invisible_queue'] else 'no'}. `accepted` = {an['accepted']}.")
w("- La demanda se adoptó **con el criterio 3 evaluado por intersección**, por decisión del autor tomada "
  "**después de ver los datos** (`DISENO_RED_4_INTERSECCIONES.md`, 7.11).")
w("")
fail = J(fail_p)["it5"]
w("**Por intersección** (criterio 3 desagregado): recompensa estilo v1 de `cola_mas_larga` menos "
  "`min_verde_y_cambiar` (> 0: la referencia es mejor), 20 semillas, desfase aleatorio. Calculado aquí a partir "
  f"de `per_seed[].gap_per_signal` de `{fail_p}` (clave `it5`).")
w("")
w("| Intersección | Ventaja media | Mediana | Semillas ganadas | t pareada p | Wilcoxon p | Peor semilla |")
w("|---|---|---|---|---|---|---|")
for s in S:
    g = np.array([p["gap_per_signal"][s] for p in fail["per_seed"]], dtype=float)
    t = paired_t(g, np.zeros_like(g))
    wl = wilcoxon(g, np.zeros_like(g))
    w(f"| {s} | {f(g.mean())} | {f(np.median(g))} | {int((g > 0).sum())}/{len(g)} | {pv(t['p'])} | {pv(wl['p'])} | {f(g.min(), 0)} |")
w("")

# ------------------------------------------------------------------ 2. Dataset
man_p = "docs/results/v2/dataset/manifest.json"
chk_p = "docs/results/v2/dataset/checks.json"
man, chk = J(man_p), J(chk_p)
w("## 2. Dataset")
w("")
w("**Splits y semillas** (cada semilla fija la de SUMO, el desfase del pulso y el generador de la política "
  "aleatoria; una recolección nueva da el mismo dataset). " + src(man_p, "docs/v2/ADDENDUM_DATASET.md"))
w("")
w("| Split | Semillas | Episodios | `aleatoria` | `fijo_2_3` | `cola_mas_larga` | Transiciones |")
w("|---|---|---|---|---|---|---|")
for split, (a, b) in man["splits"].items():
    eps = [e for e in man["episodes"] if e["split"] == split]
    cnt = {p: sum(e["policy"] == p for e in eps) for p in ("aleatoria", "fijo_2_3", "cola_mas_larga")}
    w(f"| `{split}` | {a}–{b} | {len(eps)} | {cnt['aleatoria']} | {cnt['fijo_2_3']} | {cnt['cola_mas_larga']} | {sum(e['transitions'] for e in eps):,} |")
w("")
w("- Políticas de recolección: `aleatoria` (cada bit mantener/cambiar con probabilidad 1/2, generador sembrado "
  "con la semilla del episodio), `fijo_2_3` (2 pasos en la fase 0, transversal Norte/Sur, y 3 en la fase 1, arterial; el mismo programa en los 4 semáforos) y `cola_mas_larga` (verde "
  "a la fase con más vehículos detenidos en sus carriles de entrada), asignadas en turno rotativo dentro de cada "
  "split. `ood` se reservó desde el principio; `train`, `validation` y `test` se usaron para el modelo del mundo.")
w(f"- Por política (los 190 episodios; `{chk_p}` → `per_policy`):")
w("")
w("| Política | Episodios | Retorno medio | Mínimo | Máximo | Cambios por semáforo (A0/B0/C0/D0) |")
w("|---|---|---|---|---|---|")
for p, v in chk["per_policy"].items():
    w(f"| `{p}` | {v['episodes']} | {f(v['return_mean'])} | {f(v['return_min'], 0)} | {f(v['return_max'], 0)} | "
      + " / ".join(f"{x:.1f}" for x in v["switches_per_signal_mean"]) + " |")
w("")
w("**Estado: 104 dimensiones = 4 semáforos × 26** (bloque del semáforo k en las columnas 26k…26k+25, orden "
  "A0, B0, C0, D0). Cada bloque es el `TrafficState` de la v1 (`environments/traffic_state.py`, `to_vector`):")
w("")
w("| Columnas del bloque | Variable | Qué es (por carril de entrada, 4 carriles) |")
w("|---|---|---|")
for i, (name, desc) in enumerate((
        ("vehicle_counts", "vehículos en el carril"),
        ("queue_lengths", "vehículos detenidos (cola)"),
        ("waiting_times", "tiempo de espera acumulado de los vehículos del carril (s)"),
        ("mean_speeds", "velocidad media en el carril"),
        ("occupancies", "ocupación del carril"))):
    w(f"| {4 * i}–{4 * i + 3} | `{name}` | {desc} |")
w("| 20–23 | `phase_one_hot` | fase verde actual en one-hot de 4 posiciones; solo se activan 2 (la lógica de sumo-rl tiene 2 fases verdes) |")
w("| 24 | `elapsed_phase_time` | segundos desde el último cambio de fase |")
w("| 25 | `remaining_phase_time` | `min_green − elapsed`, acotado en 0 |")
w("")
w(f"**Las 8 columnas constantes** (`{chk_p}` → `constant_columns`): " + ", ".join(f"`{c}`" for c in chk["constant_columns"])
  + ". Son las 2 posiciones del one-hot de fase que nunca se activan, en cada uno de los 4 semáforos. El escalador "
  "les da desviación 1 (`scripts/normalize_dataset.py`).")
w("")

# ------------------------------------------------------------------ 3. Autoencoder
w("## 3. Autoencoder (Fase 2: ¿comprimir el estado ayuda al modelo temporal?)")
w("")
sel_p = "docs/results/v2/autoencoder/selection.json"
w("**Arquitectura y protocolo** (`models/representation/`; `training/v2_compression_experiment.py`, "
  "`AutoencoderProtocol`; los valores salen de `autoencoders[].protocol` en " + f"`{sel_p}`):")
w("")
proto = J(sel_p)["autoencoders"][0]["protocol"]
w(f"- Encoder: Linear(104 → {proto['hidden_dim']}) + {proto['activation'].upper()} + Linear({proto['hidden_dim']} → k); "
  f"decoder simétrico. La capa latente k es el único cuello de botella.")
w(f"- Entrenamiento: MSE de reconstrucción sobre el estado normalizado, Adam, tasa {proto['learning_rate']}, lote "
  f"{proto['batch_size']}, {proto['epochs']} épocas fijas, se conserva la mejor época de validación; semillas de "
  "Autoencoder 0, 1 y 2.")
w("- Modelo temporal sobre z o sobre el estado crudo con el mismo protocolo (Adam 1e-3, weight decay 1e-4, lote "
  "32, tope 300 épocas, early stopping con paciencia 15, ventanas de 16 pasos); 3 semillas por Autoencoder y 3 "
  "en la rama cruda.")
w("- **Métrica:** log de la media geométrica del `reward_mse` de rollouts autoregresivos en h = 1…10 sobre el "
  "test del dataset (22000–22023). **Δ = exp(media(log GM z) − media(log GM crudo)) − 1** (diferencia relativa; "
  "Δ > 0: comprimir **empeora**). IC al 95% por bootstrap por conglomerados de dos niveles (Autoencoders y, dentro "
  "de cada uno, semillas del modelo temporal; 10,000 remuestreos, semilla 0). Definición en "
  "`scripts/v2/analyze_compression_experiment.py`; `delta` y `ci95` de los JSON ya son relativos.")
w("")
recon = {}
for fn in ("selection.json", "selection_k96.json", "selection_transformer.json", "selection_tsmixer.json"):
    p = R / "autoencoder" / fn
    if p.exists():
        for ae in json.loads(p.read_text(encoding="utf-8")).get("autoencoders", []):
            recon.setdefault(ae["latent_dim"], {})[ae["seed"]] = ae["reconstruction"]["mse"]
w("**Error de reconstrucción del Autoencoder** (MSE de validación sobre el estado normalizado, media de las 3 "
  "semillas; `autoencoders[].reconstruction.mse` de `docs/results/v2/autoencoder/selection*.json`):")
w("")
w("| k | MSE (semillas 0, 1, 2) | Media |")
w("|---|---|---|")
for k in sorted(recon):
    v = recon[k]
    w(f"| {k} | " + ", ".join(f"{v[s]:.4f}" for s in sorted(v)) + f" | {st.mean(v.values()):.4f} |")
w("")
arch_files = {"LSTM": ["selection_analysis.json", "selection_k96_analysis.json"],
              "Transformer": ["selection_transformer_analysis.json"],
              "TSMixer": ["selection_tsmixer_analysis.json"]}
w("**Δ por tamaño y arquitectura** (por fila: 3 Autoencoders × 3 modelos temporales sobre z, frente a 3 modelos "
  "sobre el estado crudo). Varianzas de log GM: *entre* Autoencoders, *dentro* de un Autoencoder (semillas del "
  "modelo temporal) y entre las semillas crudas.")
w("")
w("| Arquitectura | k | log GM z | log GM crudo | Diferencia de log GM | Δ relativo | IC 95% de Δ | ¿Excluye 0? | Var. entre AE | Var. dentro AE | Var. crudo | Fuente |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for arch, files in arch_files.items():
    for fn in files:
        d = J(f"docs/results/v2/autoencoder/{fn}")
        for k, v in d["sizes"].items():
            dlog = v["mean_log_gm_z"] - v["mean_log_gm_raw"]
            assert abs(math.exp(dlog) - 1 - v["delta"]) < 1e-9, (fn, k)
            rel, lo, hi = (sg(f"{x * 100:+.1f}%") for x in (v["delta"], *v["ci95"]))
            w(f"| {arch} | {k} | {v['mean_log_gm_z']:.4f} | {v['mean_log_gm_raw']:.4f} | {sg(f'{dlog:+.4f}')} | "
              f"{rel} | [{lo}, {hi}] | "
              f"{'sí' if v['ci_excludes_zero'] else 'no'} | {v['variance_between_ae']:.5f} | {v['variance_within_ae']:.5f} | "
              f"{v['variance_raw_seeds']:.5f} | `{fn}` |")
w("")
w("- Lectura (`docs/v2/ADDENDUM_AUTOENCODER.md`, secciones 8.1, 9.1 y 10.1): con mucha compresión (k ≤ 48) "
  "comprimir empeora a las tres arquitecturas. Con poca compresión (k = 80–96) es neutro con la LSTM (IC que "
  "incluyen 0), empeora al Transformer en k = 96 (en k = 80 el IC apenas incluye 0) y **ayuda a TSMixer en "
  "k = 96** (IC entero por debajo de 0). Aun así, TSMixer comprimido en k = 96 (7.6843) queda por encima de la "
  "LSTM sin comprimir (7.4247): su referencia sin comprimir es débil (sección 10.1, contexto fuera del "
  "pre-registro). La v2 sigue **sin Autoencoder** (`docs/v2/DISENO_CONTROL.md`, 2.1).")
w("")

# ------------------------------------------------------------------ 4. Arquitecturas igualadas
ap = "docs/results/v2/arch_comparison/analysis_three_architectures.json"
A = J(ap)
w("## 4. Arquitecturas con presupuesto de parámetros igualado (estado crudo de 104, 10 semillas cada una)")
w("")
w("Métrica: log GM del `reward_mse` (h = 1…10) en el test del dataset. " + src(ap))
w("")
w("| Arquitectura | Parámetros | Hiperparámetros | Media log GM | DE log GM | GM media del reward_mse | Mejores épocas (media; rango) | Épocas de parada (media) | Tocó el tope |")
w("|---|---|---|---|---|---|---|---|---|")
for arch, v in A["per_arch"].items():
    be, se_ = v["best_epochs"], v["stopped_epochs"]
    w(f"| {arch} | {v['n_params']:,} | {json.dumps(v['hparams'])} | {v['mean_log_gm']:.4f} | {v['sd_log_gm']:.4f} | "
      f"{f(math.exp(v['mean_log_gm']))} | {st.mean(be):.1f}; {min(be)}–{max(be)} | {st.mean(se_):.1f} | {v['hit_max_epochs']} |")
w("")
w(f"**Comparaciones por pares** (Welch sobre log GM, IC al {A['level'] * 100:.2f}%, Bonferroni sobre 3 pares; "
  "diferencia = primera − segunda; negativo = la primera predice mejor):")
w("")
w("| Par | Δ log GM | Relativa | p | IC de Δ | IC relativo | Decisión |")
w("|---|---|---|---|---|---|---|")
for pname, v in A["pairs"].items():
    w(sg(f"| {pname} | {v['d']:+.4f} | {v['relative'] * 100:+.1f}% | ") + pv(v['p'])
      + sg(f" | [{v['welch_ci'][0]:+.4f}, {v['welch_ci'][1]:+.4f}] | "
           f"[{v['relative_welch_ci'][0] * 100:+.1f}%, {v['relative_welch_ci'][1] * 100:+.1f}%] | ") + f"{v['decision']} |")
w("")
w(f"Decisión global: **{A['overall_decision']}**; pasan a control: {', '.join(A['advance_to_control'])}. "
  f"Razones de varianza: {json.dumps(A['variance_ratios'])}.")
w("")
w("**`reward_mse` por horizonte, 10 semillas por arquitectura.** Media geométrica entre semillas = "
  "exp(`per_arch.<arq>.mean_log_reward_mse_per_horizon`), la escala de la métrica; entre paréntesis, la media "
  f"aritmética de `runs.<arq>_s<i>.reward_mse` (calculada aquí). Fuente: `{ap}`.")
w("")
w("| h | " + " | ".join(f"{a}: geométrica (aritmética)" for a in A["per_arch"]) + " |")
w("|---|" + "---|" * len(A["per_arch"]))
per_h, per_h_gm = {}, {}
for arch in A["per_arch"]:
    runs = [v for k, v in A["runs"].items() if k.startswith(arch + "_s")]
    assert len(runs) == 10, arch
    per_h[arch] = {h: st.mean(r["reward_mse"][str(h)] for r in runs) for h in range(1, 11)}
    per_h_gm[arch] = {h: math.exp(A["per_arch"][arch]["mean_log_reward_mse_per_horizon"][str(h)]) for h in range(1, 11)}
    for h in range(1, 11):  # the stored mean log must be the mean of the 10 runs' logs
        assert abs(math.exp(st.mean(math.log(r["reward_mse"][str(h)]) for r in runs)) - per_h_gm[arch][h]) < 1e-6
for h in range(1, 11):
    w(f"| {h} | " + " | ".join(f"{f(per_h_gm[a][h])} ({f(per_h[a][h])})" for a in A["per_arch"]) + " |")
w("")

# ------------------------------------------------------------------ 5. Fase 3
s3p = "docs/results/v2/control/test_stage3_analysis.json"
s2p = "docs/results/v2/control/test_stage2_analysis.json"
S3 = J(s3p)
w("## 5. Controlador PPO en imaginación, sin planificar (Fase 3; prueba 22000–22023, 24 escenarios)")
w("")
w("10 semillas por brazo aprendido; catastróficos con < −3,600 (umbral de la Fase 3). " + src(s3p))
w("")
w("| Política | Media | Mediana | A0 | B0 | C0 | D0 | Catastróficos |")
w("|---|---|---|---|---|---|---|---|")
for n, v in S3["summary"].items():
    m = v["mean"]
    w(f"| `{n}` | {f(m['total'])} | {f(v['median_total'])} | " + " | ".join(f(m[s]) for s in S) + f" | {v['catastrophic']}/{v['n_episodes']} |")
w("")
w(f"**Las 8 comparaciones** (α' = 0.05/8 = {S3['alpha_bonferroni']:.5f}, IC al {S3['level'] * 100:.3f}%; test principal Welch sobre medias por semilla, o t de una muestra contra una regla; t pareada y Wilcoxon por escenario son secundarios y pseudorreplicación). " + src(s3p))
w("")
w("| # | Comparación | Diferencia | p principal | IC | ¿Sig.? | t pareada p | Wilcoxon p | Gana |")
w("|---|---|---|---|---|---|---|---|---|")
for lab, c in S3["planned"].items():
    sl = c["seed_level"]
    w(f"| {lab} | `{c['a']}` − `{c['b']}` ({c['metric']}) | {f(c['mean_diff'])} | {pv(sl['p'])} | [{f(sl['ci'][0])}, {f(sl['ci'][1])}] | "
      f"{'sí' if c['significant'] else 'no'} | {pv(c['paired_t']['p'])} | {pv(c['wilcoxon']['p'])} | {c['a_wins_scenarios']}/{c['n_scenarios']} |")
w("")
eps = J("docs/results/v2/control/test_stage2.json")["episodes"] + J("docs/results/v2/control/test_stage3.json")["episodes"]
S2 = J(s2p)
w("**Bloqueos** (controlador bloqueado: media < 3 cambios por episodio en algún semáforo; episodio bloqueado: < 3 "
  f"cambios en algún semáforo). Controladores de `{s2p}` → `blocked_count` y `{s3p}` → `blocked_count`; episodios "
  "calculados aquí de `test_stage2.json` y `test_stage3.json`.")
w("")
w("| Brazo | Controladores bloqueados | Episodios bloqueados | Por semáforo (A0/B0/C0/D0) |")
w("|---|---|---|---|")
bc = {**S2["blocked_count"], **S3["blocked_count"]}
for n in ("sueno_lstm", "sueno_transformer", "directo_10k", "directo_30k"):
    E = [e for e in eps if e["policy"] == n]
    be = sum(min(e[f"switches_{s}"] for s in S) < 3 for e in E)
    per = " / ".join(str(sum(e[f"switches_{s}"] < 3 for e in E)) for s in S)
    w(f"| `{n}` | {bc[n]}/10 | {be}/{len(E)} | {per} |")
w("")
fv_p, fo_p = "docs/results/v2/control/fidelity_validation.json", "docs/results/v2/control/fidelity_onpolicy_validation.json"
FV, FO = J(fv_p), J(fo_p)
w("**Fidelidad del retorno imaginado a 7 pasos** (validación; media de los 10 modelos; IC al 95% por bootstrap; "
  "sueño con la alineación `legacy`, la de todos los resultados de la Fase 3):")
w("")
w("| Arquitectura | Acciones del dataset: Pearson [IC] | Sesgo imaginado − real [IC] | Acciones del controlador (on-policy): Pearson [IC] | Sesgo [IC] |")
w("|---|---|---|---|---|")
for arch in ("lstm", "transformer"):
    a, o = FV["architectures"][arch]["clipped"], FO["summary"][arch]["clipped"]
    w(f"| {arch} | {a['pearson_mean']:.3f} [{a['pearson_mean_ci95'][0]:.3f}, {a['pearson_mean_ci95'][1]:.3f}] | "
      f"{f(a['bias_mean'])} [{f(a['bias_mean_ci95'][0])}, {f(a['bias_mean_ci95'][1])}] | "
      f"{o['pearson_mean_over_controllers']:.3f} [{o['pearson_mean_ci95'][0]:.3f}, {o['pearson_mean_ci95'][1]:.3f}] | "
      f"{f(o['bias_mean_over_controllers'])} [{f(o['bias_mean_ci95'][0])}, {f(o['bias_mean_ci95'][1])}] |")
w("")
w(f"Fuentes: `{fv_p}` → `architectures.<arq>.clipped` (validación 21000–21023, 24 episodios × 39 ventanas por modelo); "
  f"`{fo_p}` → `summary.<arq>.clipped` (20 controladores en 21000–21005, 6 escenarios: IC orientativos).")
w("")
w("**Sesgo on-policy según la fase mantenida más larga entre los 4 semáforos** (pasos desde el último cambio "
  f"real al empezar a imaginar; recompensa recortada). `{fo_p}` → `summary.<arq>.bias_by_longest_hold`:")
w("")
w("| Retención | LSTM: ventanas | LSTM: retorno real medio | LSTM: sesgo | LSTM: imaginado > real | Transformer: ventanas | Transformer: retorno real | Transformer: sesgo | Transformer: imaginado > real |")
w("|---|---|---|---|---|---|---|---|---|")
for b in ("0-5", "6-10", "11-19", ">=20"):
    L, T = FO["summary"]["lstm"]["bias_by_longest_hold"][b], FO["summary"]["transformer"]["bias_by_longest_hold"][b]
    w(f"| {b.replace('>=', '≥ ')} | {L['n_windows']:,} | {f(L['real_mean'])} | {f(L['bias_clipped'])} | {L['share_positive_bias_raw'] * 100:.0f}% | "
      f"{T['n_windows']:,} | {f(T['real_mean'])} | {f(T['bias_clipped'])} | {T['share_positive_bias_raw'] * 100:.0f}% |")
w("")

# ------------------------------------------------------------------ 6. Desalineación
fva_p = "docs/results/v2/dream_alignment/fidelity_validation_aligned.json"
foa_p = "docs/results/v2/dream_alignment/fidelity_onpolicy_validation_aligned.json"
va_p = "docs/results/v2/dream_alignment/validation_v21_analysis.json"
FVA, FOA, VA = J(fva_p), J(foa_p), J(va_p)
w("## 6. Desalineación de la ventana de acciones del Dream Environment")
w("")
w("`CorridorDreamEnvironment.step` (y el `DreamEnvironment` de la v1) avanzaba la ventana de acciones con la "
  "ventana anterior a insertar la acción elegida (`legacy`); `aligned` empareja cada estado con su acción, como "
  "`rollout_episode`. Ver `docs/v2/ADDENDUM_PLANIFICACION.md`, sección 11.")
w("")
w("**Efecto en la fidelidad** (recortado; media de los modelos o controladores):")
w("")
w("| | Pearson legacy | Pearson aligned | Sesgo legacy | Sesgo aligned |")
w("|---|---|---|---|---|")
for arch in ("lstm", "transformer"):
    a, b = FV["architectures"][arch]["clipped"], FVA["architectures"][arch]["clipped"]
    w(f"| Acciones del dataset, {arch} | {a['pearson_mean']:.3f} | {b['pearson_mean']:.3f} | {f(a['bias_mean'])} | {f(b['bias_mean'])} |")
for arch in ("lstm", "transformer"):
    a, b = FO["summary"][arch]["clipped"], FOA["summary"][arch]["clipped"]
    w(f"| On-policy, {arch} | {a['pearson_mean_over_controllers']:.3f} | {b['pearson_mean_over_controllers']:.3f} | "
      f"{f(a['bias_mean_over_controllers'])} | {f(b['bias_mean_over_controllers'])} |")
w("")
w(f"Fuentes: `{fv_p}`, `{fva_p}`, `{fo_p}`, `{foa_p}`. Sesgo por retención con `aligned` (LSTM / Transformer): "
  + "; ".join(f"{b.replace('>=', '≥ ')}: {f(FOA['summary']['lstm']['bias_by_longest_hold'][b]['bias_clipped'])} / "
              f"{f(FOA['summary']['transformer']['bias_by_longest_hold'][b]['bias_clipped'])}" for b in ("0-5", "6-10", "11-19", ">=20")) + ".")
w("")
w("**Efecto en el reentrenamiento de los 20 PPO del sueño** (misma configuración, `aligned`; evaluados con los de la "
  f"Fase 3 en 24000–24023; catastróficos con < −3,600). " + src(va_p))
w("")
w("| Brazo | Media | Mediana | A0 | B0 | C0 | D0 | Catastróficos | Episodios bloqueados | Controladores bloqueados |")
w("|---|---|---|---|---|---|---|---|---|---|")
for k, v in VA["arms"].items():
    w(f"| {k} | {f(v['mean'])} | {f(v['median'])} | " + " | ".join(f(v['per_signal'][s]) for s in S)
      + f" | {v['catastrophic']}/{v['n_episodes']} | {v['blocked_episodes']}/{v['n_episodes']} | {v['blocked_controllers']}/10 |")
w("")
w("| Arquitectura | Corregido − Fase 3 | Welch p | IC 95% | ¿Cambio importante (criterio pre-registrado)? |")
w("|---|---|---|---|---|")
for arch, c in VA["comparisons"].items():
    x = c["aligned_minus_phase3"]
    w(f"| {arch} | {f(x['d'])} | {pv(x['p'])} | [{f(x['ci95'][0])}, {f(x['ci95'][1])}] | {'sí' if c['important'] else 'no'} ({json.dumps(c['criteria'])}) |")
w("")

# ------------------------------------------------------------------ 7. Validación del planificador
vp = "docs/results/v2/planning/validation_analysis.json"
V = J(vp)
w("## 7. Validación del planificador (24000–24023, réplicas 0–2, 24 escenarios)")
w("")
w("Catastróficos con < " + f(V["selected"]["threshold"], 0) + ". * = H elegido. " + src(vp))
w("")
w("| Brazo | H | Media | Mediana | A0 | B0 | C0 | D0 | Catastróficos | Episodios bloqueados | Controladores bloqueados | ms por decisión |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for arm, byh in V["planners"].items():
    for h, v in byh.items():
        star = " *" if V["selected"]["chosen_H"][arm] == int(h) else ""
        w(f"| `{arm}` | {h}{star} | {f(v['mean'])} | {f(v['median'])} | " + " | ".join(f(v['per_signal'][s]) for s in S)
          + f" | {v['catastrophic']}/{v['n_episodes']} | {v['blocked_episodes']}/{v['n_episodes']} | {len(v['blocked_controllers'])}/3 | {v['decision_seconds'] * 1000:.0f} |")
for name, v in V["phase3_dream"].items():
    w(f"| `{name}` | — | {f(v['mean'])} | {f(v['median'])} | " + " | ".join(f(v['per_signal'][s]) for s in S)
      + f" | {v['catastrophic']}/{v['n_episodes']} | {v['blocked_episodes']}/{v['n_episodes']} | {len(v['blocked_controllers'])}/{len(v['replica_means'])} | — |")
for name, v in V["rules"].items():
    w(f"| `{name}` | — | {f(v['mean'])} | {f(v['median'])} | " + " | ".join(f(v['per_signal'][s]) for s in S)
      + f" | {v['catastrophic']}/{v['n_episodes']} | {v['blocked_episodes']}/{v['n_episodes']} | — | — |")
w("")
sel = V["selected"]
w("**Valores fijados por las reglas pre-registradas** (`selected`):")
w("")
w(f"- H por brazo: " + ", ".join(f"`{a}` = {h}" for a, h in sel["chosen_H"].items()) + " (regla: menor costo; empate → H más chico).")
w(f"- Mejor brazo por arquitectura: " + ", ".join(f"{a}: `{b}` ({f(sel['best_arm_values'][a])})" for a, b in sel["best_arm"].items()) + ".")
w(f"- Umbral de catastróficos: {f(sel['threshold'], 0)} (1.5 × peor `fijo_2_3` = 1.5 × {f(sel['worst_fijo_2_3'])}, redondeado hacia abajo a la centena).")
w(f"- Mejores reglas: " + ", ".join(f"{m}: `{r}` ({f(sel['best_rule_values'][m])})" for m, r in sel["best_rule"].items()) + ".")
w("")


# ------------------------------------------------------------------ 8 / 9. Test y OOD
def final_section(title, path, labels_prefix, n_sc, extra=None):
    D = J(path)
    w(title)
    w("")
    w(f"10 réplicas o semillas por método aprendido; {n_sc} escenarios; catastróficos con < {f(D['threshold'], 0)}. " + src(path))
    w("")
    w("| Política | Media | Mediana | A0 | B0 | C0 | D0 | Catastróficos | Episodios bloqueados | Controladores bloqueados | ms por decisión |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for n, v in D["arms"].items():
        ms = f"{v['ms_per_decision']:.1f}" if "ms_per_decision" in v else "—"
        nb = f"{len(v['blocked_controllers'])}/{len(v['seed_means'])}" if len(v["seed_means"]) > 1 else "—"
        w(f"| `{n}` | {f(v['mean'])} | {f(v['median'])} | " + " | ".join(f(v['per_signal'][s]) for s in S)
          + f" | {v['catastrophic']}/{v['n_episodes']} | {v['blocked_episodes']}/{v['n_episodes']} | {nb} | {ms} |")
    w("")
    w(f"**Las 12 comparaciones planificadas** (α' = 0.05/12 = {D['alpha_bonferroni']:.6f}, IC al {D['level'] * 100:.3f}%; principal: Welch "
      "sobre las 10 medias, o t de una muestra contra una regla; t pareada y Wilcoxon por escenario, secundarios y pseudorreplicación):")
    w("")
    w("| # | Comparación | Diferencia | p principal | IC | ¿Sig.? | t pareada p | Wilcoxon p | Gana |")
    w("|---|---|---|---|---|---|---|---|---|")
    for lab, c in D["planned"].items():
        sl = c["seed_level"]
        w(f"| {lab} | `{c['a']}` − `{c['b']}` ({c['metric']}) | {f(c['mean_diff'])} | {pv(sl['p'])} | [{f(sl['ci'][0])}, {f(sl['ci'][1])}] | "
          f"{'sí' if c['significant'] else 'no'} | {pv(c['paired_t']['p'])} | {pv(c['wilcoxon']['p'])} | {c['a_wins_scenarios']}/{c['n_scenarios']} |")
    w("")
    w("**Medias por réplica o semilla, y la que más aporta a la varianza** (fracción de la suma de desvíos al cuadrado):")
    w("")
    w("| Método | Medias (s0…s9) | Semilla dominante | Fracción |")
    w("|---|---|---|---|")
    for n, v in D["arms"].items():
        if len(v["seed_means"]) < 2:
            continue
        sh = v["variance_share_by_seed"]
        top = max(sh, key=sh.get)
        name = top.split("/")[-2] if "/" in top else top
        w(f"| `{n}` | " + ", ".join(f(x, 0) for x in v["seed_means"].values()) + f" | {name} | {sh[top] * 100:.0f}% |")
    w("")
    if extra:
        extra(D)
    return D


T = final_section("## 8. Prueba final con el planificador (25000–25047, 48 escenarios)", "docs/results/v2/planning/test_analysis.json", "Q", 48)
w("Planificador con H = 3 en los 4 brazos; `plan_ppo` continúa con los PPO del sueño de la Fase 3 "
  "(`ADDENDUM_PLANIFICACION.md`, secciones 13.1 y 15).")
w("")


def ood_extra(D):
    r = D["reproducibility"]
    w(f"**Control de reproducibilidad** (`reproducibility`): en los {r['checked']} escenarios OOD donde el dataset usó "
      f"`fijo_2_3` o `cola_mas_larga`, el retorno evaluado frente al del manifiesto (`{man_p}`): diferencia máxima "
      f"{r['max_abs_diff']}, discrepancias {len(r['mismatches'])}.")
    w("")
    w(f"**Lectura pre-registrada** (`ADDENDUM_OOD.md`, sección 3) aplicada a O2_transformer: **{D['pre_registered_reading_O2_transformer']}**.")
    w("")


O = final_section("## 9. OOD (23000–23029, 30 escenarios)", "docs/results/v2/ood_analysis.json", "O", 30, ood_extra)
w("OOD = semillas reservadas con la misma red, demanda y generador de escenarios (no una demanda distinta); "
  "sus 30 escenarios solo se habían simulado al recolectar el dataset (`ADDENDUM_OOD.md`, sección 4).")
w("")

# ------------------------------------------------------------------ 10. Interacciones y tiempos
w("## 10. Interacciones reales y tiempos medidos")
w("")
w("**Interacciones reales por réplica o semilla** (`interactions` de `docs/results/v2/planning/test_analysis.json`; "
  "pasos reales del RL directo y de la selección contados en `models/checkpoints/v2/control/*/run_info.json` → `real_steps_total`):")
w("")
w("| Método | Sin compartir el dataset | Compartiéndolo entre 10 semillas | Composición |")
w("|---|---|---|---|")
I = T["interactions"]
comp = {"planner (section 13, H and arm validated with replicas 0-2)": "dataset 9,600 + selección del PPO 3,000 + validación de H y brazo 8,640 (sin compartir) o 2,592 (compartido)",
        "sueno (Phase 3)": "dataset 9,600 (o 960) + selección en SUMO real 3,000",
        "directo_10k": "10,240 de entrenamiento + 3,000 de evaluación periódica",
        "directo_30k": "30,208 de entrenamiento + 9,000 de evaluación periódica"}
label = {"planner (section 13, H and arm validated with replicas 0-2)": "Planificador (H y brazo validados con las réplicas 0–2)",
         "sueno (Phase 3)": "PPO del sueño (Fase 3)", "directo_10k": "RL directo 10k", "directo_30k": "RL directo 30k"}
for k, v in I.items():
    w(f"| {label[k]} | {v['unshared']:,} | {v['shared']:,} | {comp.get(k, '')} |")
w("")
w("Razones frente al planificador: directo 30k / planificador = "
  f"{I['directo_30k']['unshared'] / 21240:.2f}x (sin compartir) y {I['directo_30k']['shared'] / 6552:.2f}x (compartido); "
  f"directo 10k / planificador = {13240 / 21240:.2f}x y {13240 / 6552:.2f}x.")
w("")
w("**Tiempos de entrenamiento por corrida** (`run_info.json` → `wall_seconds`, con 4 procesos en paralelo, un hilo cada uno; "
  "`seconds_per_training_step` incluye las actualizaciones de PPO):")
w("")
w("| Corridas | n | Minutos por corrida: media (mín–máx) | ms por paso de entrenamiento | Pasos reales |")
w("|---|---|---|---|---|")
for label, pat in (("PPO del sueño LSTM (Fase 3)", "models/checkpoints/v2/control/dream_lstm_s*"),
                   ("PPO del sueño Transformer (Fase 3)", "models/checkpoints/v2/control/dream_transformer_s*"),
                   ("RL directo 10k", "models/checkpoints/v2/control/direct_10k_s*"),
                   ("RL directo 30k", "models/checkpoints/v2/control/direct_30k_s*"),
                   ("PPO del sueño corregido LSTM", "models/checkpoints/v2/control_aligned/dream_lstm_s*"),
                   ("PPO del sueño corregido Transformer", "models/checkpoints/v2/control_aligned/dream_transformer_s*")):
    ws = [json.loads(Path(d, "run_info.json").read_text(encoding="utf-8")) for d in sorted(glob.glob(str(ROOT / pat)))]
    mins = [x["wall_seconds"] / 60 for x in ws]
    w(f"| {label} | {len(ws)} | {st.mean(mins):.1f} ({min(mins):.1f}–{max(mins):.1f}) | "
      f"{st.mean(x['seconds_per_training_step'] for x in ws) * 1000:.1f} | {ws[0]['real_steps_total']:,} |")
w("")
w("**Planificador, ms por decisión (H = 3)** en la prueba final: "
  + ", ".join(f"`{n}` {T['arms'][n]['ms_per_decision']:.1f}" for n in T["arms"] if "ms_per_decision" in T["arms"][n])
  + "; en OOD: " + ", ".join(f"`{n}` {O['arms'][n]['ms_per_decision']:.1f}" for n in O["arms"] if "ms_per_decision" in O["arms"][n])
  + " (`arms.<método>.ms_per_decision`).")
w("")
w("**Duraciones totales de evaluación** (registradas en los addendums, no en un JSON): validación del planificador "
  "29 min (`ADDENDUM_PLANIFICACION.md` 16); prueba final 1 h 44 min para 4,080 episodios (18); OOD 50 min 40 s para "
  "1,950 episodios (`ADDENDUM_OOD.md` 7); 20 PPO del sueño 40 min (`ADDENDUM_CONTROL.md` 11.3); 20 PPO corregidos "
  "39 min 52 s (`ADDENDUM_SUENO_CORREGIDO.md` 7). Todo con 4 procesos.")
w("")

(ROOT / "docs" / "RESULTADOS_PARA_ARTICULO.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("ok", len(out), "lines")
# values used by Part 2
print("per_h arit", {a: (round(per_h[a][1], 1), round(per_h[a][10], 1)) for a in per_h})
print("per_h geo", {a: (round(per_h_gm[a][1], 1), round(per_h_gm[a][10], 1)) for a in per_h_gm})
