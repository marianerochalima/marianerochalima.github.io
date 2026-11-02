# -*- coding: utf-8 -*-
"""
CRAS — Curva de Retenção de Água no Solo (versão desktop)

Ajuste dos modelos de Gardner (1958), van Genuchten (1980) e Fredlund e Xing (1994)
a pares de sucção matricial (kPa) e teor de umidade volumétrico (cm³/cm³),
avaliação estatística dos ajustes e estimativa da sucção matricial.

Bibliotecas: NumPy, SciPy (curve_fit e brentq), pandas, Matplotlib, openpyxl e Tkinter.
Autoria: Mariane Rocha Lima.
"""

import math
import os
import re
import sys
import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont

import numpy as np
from scipy.optimize import curve_fit, brentq
from scipy import stats

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# =============================================================================
#  MODELOS MATEMÁTICOS
# =============================================================================

PSI_R = 1e6          # sucção de referência do fator corretivo C(ψ) (kPa)
PSI_SECO = 1e6       # sucção correspondente ao solo seco (kPa)


def fator_corretivo(psi):
    psi = np.asarray(psi, dtype=float)
    return 1.0 - np.log1p(psi / PSI_R) / np.log1p(PSI_SECO / PSI_R)


def f_gardner(psi, tr, ts, a):
    psi = np.asarray(psi, dtype=float)
    return tr + (ts - tr) * np.exp(np.clip(-a * psi, -700, 700))


def f_van_genuchten(psi, tr, ts, al, n, m):
    psi = np.asarray(psi, dtype=float)
    return tr + (ts - tr) / np.power(1.0 + np.power(np.maximum(al * psi, 1e-12), n), m)


def f_fredlund_xing(psi, ts, a, n, m):
    psi = np.asarray(psi, dtype=float)
    r = np.power(np.maximum(psi / a, 1e-12), n)
    return ts / np.power(np.log(np.e + r), m) * fator_corretivo(psi)


def inv_gardner(th, p):
    tr, ts, a = p
    if not (tr < th <= ts):
        return float("nan")
    s = (th - tr) / (ts - tr)
    if s < 1e-6:
        return float("nan")
    return -math.log(s) / a


def inv_van_genuchten(th, p):
    tr, ts, al, n, m = p
    if not (tr < th <= ts):
        return float("nan")
    se = (th - tr) / (ts - tr)
    t = se ** (-1.0 / m) - 1.0
    if t < 0:
        return float("nan")
    return t ** (1.0 / n) / al


def inv_fredlund_xing(th, p):
    """Sem inversa fechada: método de Brent (scipy.optimize.brentq) em escala logarítmica."""
    g = lambda lx: float(f_fredlund_xing(10.0 ** lx, *p)) - th
    lo, hi = -4.0, 6.0
    try:
        if g(lo) * g(hi) > 0:
            return float("nan")
        return 10.0 ** brentq(g, lo, hi, xtol=1e-12, maxiter=500)
    except (ValueError, RuntimeError):
        return float("nan")


MODELOS = {
    "gardner": dict(
        nome="Gardner (1958)", curto="Gardner", cor="#C1442A", estilo=(0, (6, 3)),
        eq="θ(ψ) = θr + (θs − θr)·exp(−a·ψ)",
        pars=["θr", "θs", "a"], unid=["cm³/cm³", "cm³/cm³", "1/kPa"],
        signif=["teor de umidade residual", "teor de umidade na saturação",
                "parâmetro de ajuste relacionado à sucção"],
        f=f_gardner, inversa=inv_gardner, entrada_ar=lambda p: 1.0 / p[2]),
    "van_genuchten": dict(
        nome="van Genuchten (1980)", curto="van Genuchten", cor="#1F5FA8", estilo="-",
        eq="θ(ψ) = θr + (θs − θr) / [1 + (α·ψ)ⁿ]ᵐ",
        pars=["θr", "θs", "α", "n", "m"], unid=["cm³/cm³", "cm³/cm³", "1/kPa", "—", "—"],
        signif=["teor de umidade residual", "teor de umidade na saturação",
                "parâmetro relacionado à pressão de entrada de ar",
                "parâmetro relacionado à distribuição de poros", "parâmetro de ajuste da curva"],
        f=f_van_genuchten, inversa=inv_van_genuchten, entrada_ar=lambda p: 1.0 / p[2]),
    "fredlund_xing": dict(
        nome="Fredlund e Xing (1994)", curto="Fredlund e Xing", cor="#1B7A3D", estilo=(0, (1, 1.6)),
        eq="θ(ψ) = C(ψ)·θs / {ln[e + (ψ/a)ⁿ]}ᵐ",
        pars=["θs", "a", "n", "m"], unid=["cm³/cm³", "kPa", "—", "—"],
        signif=["teor de umidade na saturação", "parâmetro relacionado à sucção de entrada de ar",
                "parâmetro relacionado à distribuição dos poros", "parâmetro de ajuste da curva"],
        f=f_fredlund_xing, inversa=inv_fredlund_xing, entrada_ar=lambda p: p[1]),
}
ORDEM = ["gardner", "van_genuchten", "fredlund_xing"]


# =============================================================================
#  AJUSTE — mínimos quadrados não lineares (scipy.optimize.curve_fit, algoritmo TRF)
# =============================================================================

def limites_e_chutes(chave, psi, theta):
    th_max, th_min = float(theta.max()), float(theta.min())
    amp = max(th_max - th_min, 1e-4)
    alvo = th_min + amp / 2
    idx = int(np.argmin(np.abs(theta - alvo)))
    meio = max(psi[idx] if psi[idx] > 0 else psi[len(psi) // 2], 1e-3)

    if chave == "gardner":
        lo = [0.0, th_max * 0.5, 1e-8]
        hi = [th_max + amp, th_max * 1.5, 1e4]
        chutes = [[th_min * 0.9, th_max, 1 / meio],
                  [1e-6, th_max, 0.693 / meio],
                  [th_min * 0.5, th_max * 1.02, 0.1 / meio]]
    elif chave == "van_genuchten":
        lo = [0.0, th_max * 0.5, 1e-6, 1.01, 0.01]
        hi = [th_max, th_max * 1.5, 1e3, 30.0, 10.0]
        chutes = [[max(th_min * 0.8, 1e-6), th_max, 1 / meio, n0, 1 - 1 / n0]
                  for n0 in (1.5, 2.5, 1.2, 5.0)]
    else:
        lo = [th_max * 0.5, 1e-4, 0.1, 0.01]
        hi = [th_max * 1.5, 1e6, 50.0, 20.0]
        chutes = [[th_max, meio, n0, m0] for n0, m0 in ((2, 1), (1, 0.5), (4, 0.8), (1.5, 2))]
    return np.array(lo), np.array(hi), chutes


def estatisticas(obs, calc, n_par):
    n = len(obs)
    res = obs - calc
    sqr = float(np.sum(res ** 2))
    sqt = float(np.sum((obs - obs.mean()) ** 2))
    r2 = float("nan") if sqt == 0 else 1 - sqr / sqt
    den = n - n_par - 1
    aic = n * math.log(max(sqr, 1e-300) / n) + 2 * n_par
    if den > 0:
        aic += (2 * n_par * (n_par + 1)) / den
    return dict(
        r2=r2,
        r2aj=1 - (1 - r2) * (n - 1) / den if den > 0 else float("nan"),
        rmse=math.sqrt(sqr / n),
        mae=float(np.mean(np.abs(res))),
        aic=aic,
        sqr=sqr, sqt=sqt, media=float(obs.mean()), soma_abs=float(np.sum(np.abs(res))))


def jacobiano(chave, psi, p):
    f = MODELOS[chave]["f"]
    J = np.zeros((len(psi), len(p)))
    for j in range(len(p)):
        h = max(abs(p[j]) * 1e-6, 1e-10)
        pm, pp = np.array(p, float), np.array(p, float)
        pm[j] -= h
        pp[j] += h
        J[:, j] = (f(psi, *pp) - f(psi, *pm)) / (2 * h)
    return J


def ajustar(chave, psi, theta):
    M = MODELOS[chave]
    n_par, n = len(M["pars"]), len(psi)
    r = dict(chave=chave, nome=M["nome"], n_par=n_par, convergiu=False, mensagem="")
    if n < 5:
        r["mensagem"] = f"são necessários pelo menos 5 pontos (foram informados {n})"
        return r
    if n_par >= n:
        r["mensagem"] = f"o modelo tem {n_par} parâmetros para apenas {n} pontos"
        return r

    lo, hi, chutes = limites_e_chutes(chave, psi, theta)
    folga = (hi - lo) * 1e-9
    partidas, melhor = [], None
    for chute in chutes:
        p0 = np.clip(np.array(chute, float), lo + folga, hi - folga)
        try:
            popt, _ = curve_fit(M["f"], psi, theta, p0=p0, bounds=(lo, hi), method="trf",
                                x_scale="jac", max_nfev=20000)
            sqr = float(np.sum((theta - M["f"](psi, *popt)) ** 2))
        except Exception:
            partidas.append(dict(inicio=list(p0), sqr=float("nan")))
            continue
        partidas.append(dict(inicio=list(p0), sqr=sqr))
        if melhor is None or sqr < melhor[1]:
            melhor = (popt, sqr, len(partidas) - 1)

    if melhor is None or not np.isfinite(melhor[1]):
        r["mensagem"] = "a otimização não convergiu"
        return r

    p = np.array(melhor[0], float)
    calc = M["f"](psi, *p)
    r.update(estatisticas(theta, calc, n_par))
    gl = n - n_par

    # erro padrão: Cov = s²·(JᵀJ)⁻¹, com s² = SQR/(N − p)
    J = jacobiano(chave, psi, p)
    JtJ = J.T @ J
    try:
        inv = np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        inv = np.full_like(JtJ, np.nan)
    s2 = r["sqr"] / gl
    diag = np.diag(inv) * s2
    ep = np.where(diag > 0, np.sqrt(np.abs(diag)), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(ep > 0, p / ep, np.nan)
    p_par = np.array([2 * stats.t.sf(abs(t), gl) if np.isfinite(t) else np.nan for t in tstat])

    r.update(convergiu=True, p=p, calc=calc, partidas=partidas, idx_melhor=melhor[2],
             lo=lo, hi=hi, gl=gl, s2=s2, JtJ=JtJ, inv=inv, ep=ep, tstat=tstat, p_par=p_par)
    return r


def qualificacao(res):
    """Posição de cada modelo (1º = melhor) em R² ajustado, RMSE, MAE e AICc e média das posições."""
    conv = [k for k in ORDEM if k in res and res[k]["convergiu"]]
    if not conv:
        return []
    criterios = [("r2aj", True), ("rmse", False), ("mae", False), ("aic", False)]
    pos = {k: [] for k in conv}
    for chave_crit, maior in criterios:
        def valor(k):
            v = res[k][chave_crit]
            return v if np.isfinite(v) else res[k]["r2"]
        ordem = sorted(conv, key=valor, reverse=maior)
        for i, k in enumerate(ordem):
            pos[k].append(i + 1)
    media = {k: sum(pos[k]) / len(pos[k]) for k in conv}
    ordem = sorted(conv, key=lambda k: (media[k], res[k]["aic"]))

    def nota(m):
        return "Muito bom" if m <= 1.5 else "Bom" if m <= 2.25 else "Regular"
    return [dict(chave=k, posicoes=pos[k], media=media[k], nota=nota(media[k])) for k in ordem]


# =============================================================================
#  SOLOS DE REFERÊNCIA — Carsel e Parrish (1988)
# =============================================================================

CM_KPA = 0.0980665
CARSEL = {
    "Areia": (0.045, 0.43, 0.145, 2.68), "Areia franca": (0.057, 0.41, 0.124, 2.28),
    "Franco arenoso": (0.065, 0.41, 0.075, 1.89), "Franco": (0.078, 0.43, 0.036, 1.56),
    "Silte": (0.034, 0.46, 0.016, 1.37), "Franco siltoso": (0.067, 0.45, 0.020, 1.41),
    "Franco argiloarenoso": (0.100, 0.39, 0.059, 1.48), "Franco argiloso": (0.095, 0.41, 0.019, 1.31),
    "Franco argilossiltoso": (0.089, 0.43, 0.010, 1.23), "Argila arenosa": (0.100, 0.38, 0.027, 1.23),
    "Argila siltosa": (0.070, 0.36, 0.005, 1.09), "Argila": (0.068, 0.38, 0.008, 1.09),
}


def gerar_conjunto(classe):
    tr, ts, al_cm, n = CARSEL[classe]
    p = (tr, ts, al_cm / CM_KPA, n, 1 - 1 / n)
    psi_b = 1 / p[2]
    ini, fim = max(psi_b / 50, 0.01), min(max(psi_b * 3000, 1500), 1e5)
    psi = np.logspace(math.log10(ini), math.log10(fim), 16)
    theta = np.clip(f_van_genuchten(psi, *p), 0, 1)
    return dict(psi=psi, theta=theta, nome=f"{classe} (Carsel e Parrish, 1988)", avisos=[], classe=classe)


# =============================================================================
#  LEITURA E TRATAMENTO DOS DADOS
# =============================================================================

def ler_texto(texto, nome="Dados informados"):
    avisos, brutos, pares = [], 0, []
    for bruta in texto.strip().splitlines():
        linha = bruta.strip()
        if not linha or linha.startswith("#"):
            continue
        if re.search(r"[;\t]", linha):
            partes = re.split(r"[;\t]", linha)
        elif len(linha.split(",")) == 2 and not re.search(r"\d,\d", linha):
            partes = linha.split(",")
        else:
            partes = linha.split()
        if len(partes) < 2:
            continue
        brutos += 1
        try:
            a = float(partes[0].strip().replace(",", "."))
            b = float(partes[1].strip().replace(",", "."))
        except ValueError:
            continue                      # provavelmente é o cabeçalho
        if math.isfinite(a) and math.isfinite(b):
            pares.append((a, b))
    if not pares:
        raise ValueError("nenhum par numérico foi reconhecido. Informe duas colunas: "
                         "sucção em kPa e teor de umidade em cm³/cm³.")

    incompletos = brutos - len(pares)
    if incompletos > 0:
        avisos.append(f"{incompletos} linha(s) sem par numérico válido foram ignoradas.")

    psi = np.array([p[0] for p in pares], float)
    theta = np.array([p[1] for p in pares], float)

    manter = psi > 0
    if (~manter).sum():
        avisos.append(f"{(~manter).sum()} ponto(s) com sucção menor ou igual a zero foram excluídos.")
    psi, theta = psi[manter], theta[manter]

    if len(theta):
        th_max = theta.max()
        if 1.5 < th_max <= 100:
            theta = theta / 100
            avisos.append("os valores de umidade estavam em porcentagem e foram divididos por 100 "
                          "para virar cm³/cm³.")
        elif th_max > 100:
            avisos.append("há valores de umidade acima de 100 — verifique a unidade da coluna.")
    manter = (theta >= 0) & (theta <= 1)
    if (~manter).sum():
        avisos.append(f"{(~manter).sum()} ponto(s) com umidade fora da faixa de 0 a 1 cm³/cm³ "
                      "foram excluídos.")
    psi, theta = psi[manter], theta[manter]

    ordem = np.argsort(psi, kind="stable")
    psi, theta = psi[ordem], theta[ordem]
    if len(theta) > 1 and np.any(np.diff(theta) > 1e-6):
        avisos.append("a umidade não decresce de forma monotônica com a sucção — confira se os "
                      "dados são de um único ramo (secagem ou umedecimento).")
    if len(psi) < 5:
        avisos.append(f"o conjunto tem apenas {len(psi)} ponto(s). A metodologia exige no mínimo 5 "
                      "para o ajuste dos modelos de três ou mais parâmetros.")
    return dict(psi=psi, theta=theta, nome=nome, avisos=avisos, classe=None)


def ler_arquivo(caminho):
    nome = os.path.basename(caminho)
    if caminho.lower().endswith((".xlsx", ".xlsm", ".xls")):
        import pandas as pd
        df = pd.read_excel(caminho, header=None)
        if df.shape[1] < 2:
            raise ValueError("a planilha precisa ter ao menos duas colunas (sucção e umidade).")
        linhas = ["\t".join("" if (isinstance(v, float) and math.isnan(v)) else str(v) for v in row)
                  for row in df.iloc[:, :2].itertuples(index=False)]
        texto = "\n".join(linhas)
    else:
        for cod in ("utf-8-sig", "latin-1"):
            try:
                with open(caminho, encoding=cod) as fh:
                    texto = fh.read()
                break
            except UnicodeDecodeError:
                continue
    return ler_texto(texto, nome), texto


# =============================================================================
#  FORMATAÇÃO
# =============================================================================

def g5(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    if v != 0 and (abs(v) < 1e-4 or abs(v) >= 1e6):
        return f"{v:.4e}"
    return f"{v:.6g}"


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "—"
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def fmt(v, d=4):
    if v is None or not np.isfinite(v):
        return "—"
    if abs(v) >= 1e5 or (abs(v) < 1e-3 and v != 0):
        return f"{v:.2e}"
    return f"{v:.{d}f}"


# =============================================================================
#  GRÁFICO
# =============================================================================

def desenhar_figura(fig, conj, res, titulo=False):
    fig.clear()
    fig.set_facecolor("white")
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)

    psi, th = conj["psi"], conj["theta"]
    conv = [k for k in ORDEM if k in res and res[k]["convergiu"]]
    psi_min = max(psi.min() * 0.5, 1e-3)
    psi_max = max(psi.max() * 5, 1500)
    xs = np.logspace(math.log10(psi_min), math.log10(psi_max), 500)

    ax.set_xscale("log")
    for k in conv:
        M, r = MODELOS[k], res[k]
        solida = k == "van_genuchten"
        ax.plot(xs, M["f"](xs, *r["p"]), color=M["cor"], linestyle=M["estilo"],
                linewidth=3.6 if solida else 2.1, zorder=2 if solida else (3 if k == "gardner" else 4),
                dash_capstyle="round", solid_capstyle="round",
                label=f"{M['nome']} — R² = {r['r2']:.4f}")
    ax.scatter(psi, th, s=44, facecolor="white", edgecolor="#15202b", linewidths=1.5, zorder=6,
               label="Dados experimentais")
    ax.set_ylabel("Teor de umidade volumétrico, θ (cm³/cm³)")
    ax.grid(True, which="major", color="#e1e6eb", linewidth=0.9)
    ax.grid(True, which="minor", axis="x", color="#f0f3f5", linewidth=0.6)
    ax.tick_params(labelbottom=False)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ordem = [labels.index("Dados experimentais")] + [i for i, l in enumerate(labels)
                                                         if l != "Dados experimentais"]
        ax.legend([handles[i] for i in ordem], [labels[i] for i in ordem], loc="upper right",
                  fontsize=9, frameon=True, framealpha=0.95, edgecolor="#d5dbe1", handlelength=3)

    ax2.axhline(0, color="#9aa6b2", linewidth=1)
    for k in conv:
        ax2.scatter(psi, th - res[k]["calc"], s=22, color=MODELOS[k]["cor"], zorder=3, alpha=0.9)
    ax2.set_xscale("log")
    ax2.set_xlabel("Sucção matricial, ψ (kPa) — escala logarítmica")
    ax2.set_ylabel("Resíduo\n(cm³/cm³)")
    ax2.grid(True, which="major", color="#e1e6eb", linewidth=0.9)
    for lado in ("top", "right"):
        ax2.spines[lado].set_visible(False)
    ax.set_xlim(psi_min, psi_max)
    if titulo:
        fig.suptitle(f"Curva de retenção de água no solo — {conj['nome']}", fontsize=12,
                     fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.93 if titulo else 0.97, bottom=0.1)


# =============================================================================
#  INTERFACE
# =============================================================================

# ícone do programa (PNG 64 e 32 px, em base64, para não depender de arquivos externos)
ICONE_64 = "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"
ICONE_32 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAIT0lEQVR4Xn1XXYxdVRX+9s8599y5P/PPdIa2oCIDlBaipYkpRsQY+wAmEArxCYwP9UF9KsILiVEfSMEX0xg0JkLUmBQbLYlaHmwIQaUFBfszlLbT3+n833tn7sy99/ztvc3a+5zOtM7Mvtm5956z9/rWWt9aa6/N4AYzxoAxZn5y9NSWXZ/bvLfki4eYUX1aG56t+b/BsidmvQUAOGdaQ9TbifrgxPjEmy/t2X7NGMMJC4BhMIZ+McaY/suZmR9uHao+P9AdDEiC3UgyAJW9F7kmG2iaamB+MZy/MtM88Ni2oVesEoCRhwDSRh09O/f6ztGBZ9MIiDqpikiYMeuIZki1Rn/gA9CohSkk30hjZhgDBsvBwHB/cODoJ7PbGGPPHTLG6X7k5PUXHt4+8nJ7WcVxqjzOSbm1B2MOvK8g0UiX7aJeWUYtSuFxDqJyvaG1Nr4nkmJJ+P86Of3C4w8MH2CvHj2/5ZH7h8+M9BdL7TBlG4Jnbi8KjnJB4aWT/4E2Bj/d8QW0YomO0paOjZgjJboCaSZr7dY/zk/dx0c3V54eGixVwkgbxjgjA9abOpM8UGR47cJZ1KIItU6Ig2fH7DMatGYjGYRBWEMD5cqd/eVneMmXuziDIUvW3LgqFmNtsKUkcWTyCv5dq6HIOYpC4MTcHN68fAFbyxJxFpn5vrVkEhaxVQn8XZxx3ksZqG02rPGxEoBIaWwqehhbquOtiWuoSA+J0nZWpYdD4xfxcX0WwyXPrqU9tHetj8WirOeslxsYtqHbKOKNQZcUMDzGr8fPIVWauITWxn0bDY9xHDx9GokO7VraQ3s3opPS3xaZ3F3aJtXKtM+NgdIGwyWB31w6Z4vP7sFBazm9I3cqpSEZw3Srg9fGTtu1qSZKSYmbZZL81bRy56p1JoBIG4yUfByvT+OjWh1PbbkTVekj0coBkBeMQaI1Kp7EsWvX8c71q9hc9hEpp8B68uklX239jcDJtCTLKeUUi/G7yxexZ3gzlpMEb1+fgACz7x0NpIi21ARc4FenzyBSRAW3a26Vu9oLjoI8YFZ5gkaiDW4vSRyeuISq8HBXpWLBlaYgy/l3HrBUaAPJGaaW2/jtJ2MYoazIcvdmL7iMs2fFzdrlMevAewoeJqMlvDszg6/ctgkn63VMttuWb2tZpoSlIftPXqh4Ht4av4SLi3X0By5bHM6K/BzXeuDWLKClBDAQcByZuIIez0eP7+O/9RokVUObAVkWWCXy6Z6ReWGS4vdjY+gPeHZo3Vxn8qLmgjDjPOeGrK/6EjPxsi0y91S70YgizIchOFjGec49CdaA1mBaAVpBpwoVKfDulQlcXqhZTyZZQN6UBRSE9sHqLADs0dlXEHh/bhqdNEWJM8y2lpGmiQWigHOgDjA1QJt5aPIATVZAk/tY5gGmlMDhyxPoK3Jr1K0ZQdjkUZerriGxyhDHKdf4YHERgVfEsXrTltjU64JSigiCAkPMJVIA5WgJd7RmMdKeQ0+8BF8lUIxhxvjofDqPZNu98ISwBQu2ODllaMgbWmWakGWFQgGsdglfP3YQRhSw6JcwX+xFrdiHmt+NFvetF3o7DYw2LuLexcvoj5uQhtRyBtH5T7+T2gnI2qOQ1VHEYeiMzL2eeyAPQjCXWsJj4LWrGJ76FLJYAdMp9WwwjCMWHhIu7f9ARZBGI+YeOlzAMOtQN0gBxqGjJvTCNGTfKEJtwIULxjwIJRFxozSuekGKdJgHSdYy6ayi6m4MhErsf3pPHSO3Esm9mWlWAdLA2IDUFCs5zauCnrBXKHAIrgClAO/qoV7NBRpxlxUPgnHiaHkm2FrL7HpmW7NMFGMQ0rOySGYebzeC3lGQF4mcPI5OrMD6tgJ+GWHYgRCCulsLsFazRkEVxynaYYJOlCBOyGoKaoOenm4ri2TS2Zed7lkQakh7+lnX0+HpLImjCMmmAdQLt2Ps+N9Q6u61BaPgC/ieRMETEIJbkDhJ0eokaIexA85qLBcC7eYCtj/yAOLuAcTTbfiSMiGroIRJHmCOwZU+jooDgNm2wbZvfgen3vkrwiiBVgpLrZX0WT2IKesd8pKLPnDOERQ8fHHvPsx2jJW52nqXhtxwrfWCw7d3E9cuCY76QgfB/buw+5nvotWYQyEowJMCnuRuipVJLTmBW0U4g+/5WKrN4kt798G/7yE0Gh0rM2/7bHTaVkM3eJQm7yttg3ulJBtACIbzUx3s+PaLePAbezE9cc1WPDBhQ4yoyicZTUmQaoblUOHChXHc8+iTdi/JIFk21fNmxNB5AhZFyXE5X1s+VO5a/FF3tVSKwthQW+4SwnaqODOX4OHnfw5V7MHbb/zCFqCuchm+71sFaFBHFMUxWstL4Fxgz3M/wFe//2O7lxKIcacAraa2vBD4rNFoNucbc4eshMMfT+6/6+7hV5YaYayU9hgR6u5rth5QCzs6FKDx0Xt47w+/xLkP/4lmo4Y0SawCnueh2tePu3fuxpe/tQ/dD+7GuZnQGkCUWG7pvqCNEYInld7Av3hhav8TO0Z+xuh69DRj6k+nZl//zGcHn201U6SpzRnaaG2k0EtSjf7eCoa6ADU3icWr59Guz1oFuvpuQ/fWz0MMjmCmBdQWlmycWC+6zLIESOmLUlXi0vjMG0/s2OSuZgTi1jD955MzL1a7K/uLpWK/vTnmJ0ZWWFJl7NkeBAzFACh4LheiBOiEQBgaezOSwgHnw9YzA3RandrC4uKrT+4Yfpkup+52nA1rLWPm4N/P3LFlZNNTUvCdRus+rfWN63lehLTS9n5ILrbKcQbJBXh2Tc7rvE00zjXjvJ4q/eG1yek/fu9r267kWPT+f9zdpIlh4R+lAAAAAElFTkSuQmCC"

AZUL = "#1F4E79"
FUNDO = "#f4f6f8"
TINTA = "#15202b"
TINTA2 = "#44515e"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CRAS — Curva de Retenção de Água no Solo")
        self.geometry("1380x880")
        self.minsize(1000, 640)
        self.configure(bg=FUNDO)
        try:
            self._ico64 = tk.PhotoImage(data=ICONE_64)
            self._ico32 = tk.PhotoImage(data=ICONE_32)
            self.iconphoto(True, self._ico64, self._ico32)
        except tk.TclError:
            self._ico32 = None
        if sys.platform.startswith("win"):
            try:
                self.state("zoomed")
            except tk.TclError:
                pass

        self.conjunto = None
        self.resultados = {}
        self.sel = {k: tk.BooleanVar(value=True) for k in ORDEM}

        self._estilos()
        self._montar()
        self.carregar_solo()
        self.executar()

    # ---------------------------------------------------------------- estilos
    def _estilos(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        base = tkfont.nametofont("TkDefaultFont")
        familia = base.actual("family")
        self.f_txt = (familia, 10)
        self.f_neg = (familia, 10, "bold")
        self.f_tit = (familia, 13, "bold")
        self.f_kpi = (familia, 15, "bold")
        mono = "Consolas" if sys.platform.startswith("win") else "DejaVu Sans Mono"
        self.f_mono = (mono, 10)
        serif = "Cambria" if sys.platform.startswith("win") else "DejaVu Serif"
        self.f_eq = (serif, 11)

        st.configure(".", background=FUNDO, foreground=TINTA, font=self.f_txt)
        st.configure("Painel.TFrame", background="white")
        st.configure("Painel.TLabelframe", background="white", bordercolor="#dfe4e9", relief="solid")
        st.configure("Painel.TLabelframe.Label", background="white", foreground=TINTA, font=self.f_neg)
        st.configure("Branco.TLabel", background="white")
        st.configure("Eq.TLabel", background="white", foreground=TINTA2, font=self.f_mono)
        st.configure("Nota.TLabel", background="white", foreground="#6d7a87")
        st.configure("Branco.TCheckbutton", background="white")
        st.configure("TButton", padding=(10, 5))
        st.configure("Primario.TButton", background=AZUL, foreground="white", font=self.f_neg,
                     padding=(10, 8), bordercolor=AZUL)
        st.map("Primario.TButton", background=[("active", "#173d60")])
        st.configure("Treeview", rowheight=24, background="white", fieldbackground="white")
        st.configure("Treeview.Heading", font=self.f_neg, background="#e9edf1")
        st.configure("TNotebook", background=FUNDO)
        st.configure("TNotebook.Tab", padding=(14, 6), font=self.f_txt)
        st.map("TNotebook.Tab", background=[("selected", "white")], foreground=[("selected", AZUL)])
        st.configure("Kpi.TFrame", background=FUNDO, relief="solid", borderwidth=1)
        st.configure("KpiR.TLabel", background=FUNDO, foreground="#6d7a87", font=(familia, 9))
        st.configure("KpiV.TLabel", background=FUNDO, foreground=TINTA, font=self.f_kpi)

    # --------------------------------------------------------------- montagem
    def _montar(self):
        topo = tk.Frame(self, bg="#10283f", height=52)
        topo.pack(fill="x")
        if self._ico32 is not None:
            tk.Label(topo, image=self._ico32, bg="#10283f").pack(side="left", padx=(14, 0), pady=8)
        tk.Label(topo, text="CRAS", bg="#10283f", fg="white", font=(self.f_tit[0], 15, "bold")
                 ).pack(side="left", padx=(8, 8), pady=10)
        tk.Label(topo, text="Curva de Retenção de Água no Solo — ajuste de modelos e estimativa "
                 "da sucção matricial", bg="#10283f", fg="#9fb3c8", font=self.f_txt).pack(side="left")

        corpo = ttk.Frame(self, padding=12)
        corpo.pack(fill="both", expand=True)
        corpo.columnconfigure(1, weight=1)
        corpo.rowconfigure(0, weight=1)

        # painel lateral com barra de rolagem: em telas pequenas ou com zoom do Windows
        # (125%, 150%) o conteúdo não cabe na altura e os botões de baixo sumiam
        caixa = ttk.Frame(corpo)
        caixa.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        cv = tk.Canvas(caixa, bg=FUNDO, highlightthickness=0, bd=0)
        sb = ttk.Scrollbar(caixa, orient="vertical", command=cv.yview)
        lateral = ttk.Frame(cv)
        cv.create_window((0, 0), window=lateral, anchor="nw")
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side="left", fill="y")
        sb.pack(side="left", fill="y")

        def ajustar_rolagem(_=None):
            cv.configure(scrollregion=cv.bbox("all"), width=lateral.winfo_reqwidth())
            precisa = lateral.winfo_reqheight() > cv.winfo_height() > 1
            if precisa and not sb.winfo_ismapped():
                sb.pack(side="left", fill="y")
            elif not precisa and sb.winfo_ismapped():
                sb.pack_forget()
        lateral.bind("<Configure>", ajustar_rolagem)
        cv.bind("<Configure>", ajustar_rolagem)

        def rolar(ev):
            if lateral.winfo_reqheight() > cv.winfo_height():
                cv.yview_scroll(int(-ev.delta / 120) if ev.delta else (1 if ev.num == 5 else -1), "units")
        for w in (cv, lateral):
            w.bind("<Enter>", lambda e: (cv.bind_all("<MouseWheel>", rolar), cv.bind_all("<Button-4>", rolar),
                                         cv.bind_all("<Button-5>", rolar)))
            w.bind("<Leave>", lambda e: (cv.unbind_all("<MouseWheel>"), cv.unbind_all("<Button-4>"),
                                         cv.unbind_all("<Button-5>")))
        self._painel_lateral(lateral)

        self.nb = ttk.Notebook(corpo)
        self.nb.grid(row=0, column=1, sticky="nsew")
        self._aba_grafico()
        self._aba_parametros()
        self._aba_estatisticas()
        self._aba_succao()
        self._aba_dados()
        self._aba_solos()
        self._aba_processo()

        self.status = tk.StringVar()
        tk.Label(self, textvariable=self.status, anchor="w", bg="#e9edf1", fg=TINTA2,
                 font=self.f_txt, padx=12, pady=4).pack(fill="x", side="bottom")

    def _painel_lateral(self, pai):
        # 1 — dados
        lf = ttk.LabelFrame(pai, text=" 1  Dados experimentais ", style="Painel.TLabelframe", padding=10)
        lf.pack(fill="x")
        ttk.Label(lf, text="Solo de referência (Carsel e Parrish, 1988)", style="Branco.TLabel").pack(anchor="w")
        linha = ttk.Frame(lf, style="Painel.TFrame")
        linha.pack(fill="x", pady=(2, 8))
        self.cb_solo = ttk.Combobox(linha, values=list(CARSEL), state="readonly", width=24)
        self.cb_solo.set("Franco arenoso")
        self.cb_solo.pack(side="left", fill="x", expand=True)
        ttk.Button(linha, text="Carregar", command=self.carregar_solo).pack(side="left", padx=(6, 0))

        ttk.Label(lf, text="Ou informe os seus dados — ψ (kPa) e θ (cm³/cm³)",
                  style="Branco.TLabel").pack(anchor="w")
        self.txt_dados = tk.Text(lf, height=8, width=34, font=self.f_mono, relief="solid", bd=1,
                                 highlightthickness=0, wrap="none")
        self.txt_dados.pack(fill="x", pady=(2, 6))
        self.txt_dados.insert("1.0", "# sucção (kPa)\tumidade (cm³/cm³)\n1\t0.452\n6\t0.425\n10\t0.402\n"
                                     "33\t0.338\n100\t0.281\n300\t0.238\n1000\t0.205\n1500\t0.196")
        linha = ttk.Frame(lf, style="Painel.TFrame")
        linha.pack(fill="x")
        ttk.Button(linha, text="Usar estes dados", command=self.usar_texto).pack(side="left", fill="x", expand=True)
        ttk.Button(linha, text="Abrir arquivo…", command=self.abrir_arquivo).pack(side="left", fill="x",
                                                                                 expand=True, padx=(6, 0))
        self.txt_avisos = tk.Text(lf, height=4, width=34, font=(self.f_txt[0], 9), relief="flat",
                                  bg="#eef7f0", wrap="word", highlightthickness=0, padx=6, pady=4)
        self.txt_avisos.pack(fill="x", pady=(8, 0))
        self.txt_avisos.configure(state="disabled")

        # 2 — modelos
        lf = ttk.LabelFrame(pai, text=" 2  Modelos a ajustar ", style="Painel.TLabelframe", padding=10)
        lf.pack(fill="x", pady=12)
        for k in ORDEM:
            M = MODELOS[k]
            fr = ttk.Frame(lf, style="Painel.TFrame")
            fr.pack(fill="x", pady=2)
            tk.Label(fr, text="■", fg=M["cor"], bg="white", font=self.f_neg).pack(side="left")
            ttk.Checkbutton(fr, text=M["nome"], variable=self.sel[k], style="Branco.TCheckbutton"
                            ).pack(side="left")
            ttk.Label(lf, text="   " + M["eq"], style="Eq.TLabel").pack(anchor="w", pady=(0, 4))

        # 3 — análise
        lf = ttk.LabelFrame(pai, text=" 3  Análise ", style="Painel.TLabelframe", padding=10)
        lf.pack(fill="x")
        ttk.Button(lf, text="Ajustar modelos", style="Primario.TButton", command=self.executar).pack(fill="x")
        ttk.Button(lf, text="Exportar resultados (CSV)", command=self.exportar_csv).pack(fill="x", pady=(8, 0))
        linha = ttk.Frame(lf, style="Painel.TFrame")
        linha.pack(fill="x", pady=(6, 0))
        linha.columnconfigure((0, 1), weight=1, uniform="exp")
        ttk.Button(linha, text="Gráfico PNG", command=lambda: self.exportar_grafico("png")
                   ).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(linha, text="Gráfico SVG", command=lambda: self.exportar_grafico("svg")
                   ).grid(row=0, column=1, sticky="ew", padx=(3, 0))

    # ------------------------------------------------------------------ abas
    def _aba(self, titulo):
        fr = ttk.Frame(self.nb, style="Painel.TFrame", padding=14)
        self.nb.add(fr, text=titulo)
        return fr

    def _tabela(self, pai, altura=8):
        fr = ttk.Frame(pai, style="Painel.TFrame")
        tv = ttk.Treeview(fr, show="headings", height=altura)
        sy = ttk.Scrollbar(fr, orient="vertical", command=tv.yview)
        sx = ttk.Scrollbar(fr, orient="horizontal", command=tv.xview)
        tv.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        tv.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        fr.columnconfigure(0, weight=1)
        fr.rowconfigure(0, weight=1)
        tv.tag_configure("destaque", foreground=AZUL, font=self.f_neg)
        tv.tag_configure("titulo", font=self.f_neg)
        return fr, tv

    def _preencher(self, tv, cab, linhas, larguras=None, tags=None):
        tv.delete(*tv.get_children())
        tv["columns"] = [f"c{i}" for i in range(len(cab))]
        for i, h in enumerate(cab):
            w = (larguras[i] if larguras else 120)
            tv.heading(f"c{i}", text=h, anchor="w" if i == 0 else "e")
            tv.column(f"c{i}", width=w, minwidth=60, anchor="w" if i == 0 else "e", stretch=True)
        for j, l in enumerate(linhas):
            tv.insert("", "end", values=[str(c) for c in l], tags=(tags[j],) if tags and tags[j] else ())

    def _nota(self, pai, texto):
        lb = ttk.Label(pai, text=texto, style="Nota.TLabel", wraplength=900, justify="left")
        lb.pack(anchor="w", pady=(6, 4), fill="x")
        return lb

    def _titulo(self, pai, texto):
        ttk.Label(pai, text=texto, style="Branco.TLabel", font=self.f_tit).pack(anchor="w", pady=(10, 4))

    def _aba_grafico(self):
        fr = self._aba("Gráfico")
        self.fig = Figure(figsize=(9, 6.4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=fr)
        barra = NavigationToolbar2Tk(self.canvas, fr, pack_toolbar=False)
        barra.update()
        barra.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _aba_parametros(self):
        fr = self._aba("Parâmetros")
        t, self.tv_par = self._tabela(fr, 14)
        t.pack(fill="both", expand=True)
        self._nota(fr, "O erro padrão vem da matriz de covariância do ajuste, Cov = s²·(JᵀJ)⁻¹. "
                   "O p-valor é do teste t bicaudal (H₀: o parâmetro é igual a zero; t = valor ÷ erro "
                   "padrão; N − p graus de liberdade). p < 0,05 (em destaque) indica parâmetro "
                   "estatisticamente diferente de zero; p alto indica parâmetro mal determinado pelos dados.")

    def _aba_estatisticas(self):
        fr = self._aba("Estatísticas")
        self.fr_kpis = ttk.Frame(fr, style="Painel.TFrame")
        self.fr_kpis.pack(fill="x", pady=(0, 8))
        self.lb_alerta = ttk.Label(fr, text="", style="Branco.TLabel", foreground="#6b5310",
                                   wraplength=900, justify="left")
        self.lb_alerta.pack(anchor="w")
        t, self.tv_est = self._tabela(fr, 4)
        t.pack(fill="x")
        self._titulo(fr, "Qualificação dos modelos")
        t, self.tv_rank = self._tabela(fr, 4)
        t.pack(fill="x")
        self._nota(fr, "Cada modelo recebe uma posição (1º = melhor) em R² ajustado, RMSE, MAE e AICc. "
                   "A média das posições define a colocação: até 1,50 = muito bom; até 2,25 = bom; "
                   "acima disso = regular.")
        self._titulo(fr, "Índices derivados da curva ajustada")
        t, self.tv_ind = self._tabela(fr, 4)
        t.pack(fill="x")
        self._nota(fr, "O R² sempre melhora quando se acrescentam parâmetros; por isso o AICc aparece ao "
                   "lado — ele penaliza a complexidade e, entre ajustes de qualidade parecida, aponta o "
                   "modelo mais parcimonioso. Menor AICc é melhor. Água disponível: diferença de θ entre "
                   "10 kPa (capacidade de campo, CC) e 1500 kPa (ponto de murcha permanente, PMP).")

    def _aba_succao(self):
        fr = self._aba("Sucção estimada")
        ttk.Label(fr, text="Teor de umidade θ (cm³/cm³) para estimar a sucção", style="Branco.TLabel"
                  ).pack(anchor="w")
        linha = ttk.Frame(fr, style="Painel.TFrame")
        linha.pack(anchor="w", pady=(2, 8))
        self.var_alvo = tk.StringVar(value="0.25")
        en = ttk.Entry(linha, textvariable=self.var_alvo, width=12)
        en.pack(side="left")
        en.bind("<Return>", lambda e: self.estimar_pontual())
        ttk.Button(linha, text="Calcular ψ", command=self.estimar_pontual).pack(side="left", padx=6)
        self.fr_suc = ttk.Frame(fr, style="Painel.TFrame")
        self.fr_suc.pack(fill="x", pady=(0, 8))
        self._titulo(fr, "Tabela completa")
        t, self.tv_suc = self._tabela(fr, 11)
        t.pack(fill="both", expand=True)
        self._nota(fr, "Gardner e van Genuchten possuem inversa analítica. Fredlund e Xing é invertido "
                   "numericamente pelo método de Brent (scipy.optimize.brentq), porque o fator corretivo "
                   "C(ψ) impede a solução em forma fechada. Um traço indica teor de umidade fora da faixa "
                   "que o modelo consegue representar.")

    def _aba_dados(self):
        fr = self._aba("Tabela de dados")
        t, self.tv_dados = self._tabela(fr, 20)
        t.pack(fill="both", expand=True)

    def _aba_solos(self):
        fr = self._aba("Solos de referência")
        t, self.tv_solos = self._tabela(fr, 13)
        t.pack(fill="x")
        self._nota(fr, "Os 12 solos de referência seguem a classificação textural do USDA, com os "
                   "parâmetros de van Genuchten médios de Carsel e Parrish (1988), os mesmos usados no "
                   "HYDRUS-1D. O solo carregado aparece em destaque.")

    def _aba_processo(self):
        fr = self._aba("Processo de cálculo")
        self.txt_pc = tk.Text(fr, wrap="word", relief="flat", bg="white", padx=18, pady=10,
                              font=self.f_txt, highlightthickness=0, spacing1=2, spacing3=2)
        sb = ttk.Scrollbar(fr, orient="vertical", command=self.txt_pc.yview)
        self.txt_pc.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt_pc.pack(fill="both", expand=True)
        t = self.txt_pc
        t.tag_configure("h", font=(self.f_tit[0], 13, "bold"), foreground=TINTA, spacing1=14, spacing3=6)
        t.tag_configure("sep", font=(self.f_txt[0], 4))
        t.tag_configure("num", font=(self.f_tit[0], 13, "bold"), foreground="white", background=AZUL)
        t.tag_configure("p", foreground=TINTA2, lmargin1=34, lmargin2=34, spacing3=4)
        t.tag_configure("eq", font=self.f_eq, background="#f1f4f7", lmargin1=46, lmargin2=46,
                        rmargin=20, spacing1=3, spacing3=3)
        t.tag_configure("res", font=(self.f_eq[0], self.f_eq[1], "bold"), foreground=AZUL,
                        background="#f1f4f7")
        t.tag_configure("tab", font=self.f_mono, lmargin1=46, lmargin2=46, foreground=TINTA)
        t.tag_configure("tabcab", font=(self.f_mono[0], self.f_mono[1], "bold"), lmargin1=46,
                        lmargin2=46, foreground=TINTA2)
        for k in ORDEM:
            t.tag_configure("m_" + k, font=self.f_neg, foreground=MODELOS[k]["cor"],
                            lmargin1=34, spacing1=8)

    # --------------------------------------------------------------- dados
    def _mostrar_avisos(self):
        c = self.conjunto
        t = self.txt_avisos
        t.configure(state="normal")
        t.delete("1.0", "end")
        if c["avisos"]:
            t.configure(bg="#fff8e8", fg="#6b5310")
            t.insert("end", f"{c['nome']} — {len(c['psi'])} ponto(s)\n")
            for a in c["avisos"]:
                t.insert("end", "• " + a + "\n")
        else:
            t.configure(bg="#eef7f0", fg="#1B7A3D")
            t.insert("end", f"{c['nome']} — {len(c['psi'])} ponto(s).\nOs dados passaram em todas as "
                            "verificações de consistência.")
        t.configure(state="disabled")
        self.status.set(f"Conjunto atual: {c['nome']}, {len(c['psi'])} pontos.")

    def _novo_conjunto(self, conj):
        self.conjunto = conj
        self.resultados = {}
        self._mostrar_avisos()
        self.atualizar_tudo()

    def carregar_solo(self):
        self._novo_conjunto(gerar_conjunto(self.cb_solo.get()))

    def usar_texto(self):
        try:
            self._novo_conjunto(ler_texto(self.txt_dados.get("1.0", "end")))
        except ValueError as e:
            messagebox.showerror("Erro nos dados", str(e))

    def abrir_arquivo(self):
        caminho = filedialog.askopenfilename(
            title="Abrir dados de retenção",
            filetypes=[("Dados", "*.csv *.txt *.tsv *.xlsx"), ("Planilha Excel", "*.xlsx"),
                       ("Texto", "*.csv *.txt *.tsv"), ("Todos", "*.*")])
        if not caminho:
            return
        try:
            conj, texto = ler_arquivo(caminho)
        except Exception as e:
            messagebox.showerror("Erro ao ler o arquivo", f"{os.path.basename(caminho)}: {e}")
            return
        self.txt_dados.delete("1.0", "end")
        self.txt_dados.insert("1.0", texto)
        self._novo_conjunto(conj)

    # ------------------------------------------------------------- execução
    def executar(self):
        if self.conjunto is None:
            messagebox.showwarning("CRAS", "Carregue um conjunto de dados.")
            return
        if len(self.conjunto["psi"]) < 5:
            messagebox.showwarning("Pontos insuficientes",
                                   f"O conjunto tem {len(self.conjunto['psi'])} ponto(s) e são "
                                   "necessários no mínimo 5.")
            return
        escolhidos = [k for k in ORDEM if self.sel[k].get()]
        if not escolhidos:
            messagebox.showwarning("CRAS", "Selecione ao menos um modelo.")
            return
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            self.resultados = {k: ajustar(k, self.conjunto["psi"], self.conjunto["theta"])
                               for k in escolhidos}
        finally:
            self.config(cursor="")
        self.atualizar_tudo()
        falhas = [r["nome"] + ": " + r["mensagem"] for r in self.resultados.values() if not r["convergiu"]]
        if falhas:
            messagebox.showwarning("Ajuste", "\n".join(falhas))

    def _conv(self):
        return [k for k in ORDEM if k in self.resultados and self.resultados[k]["convergiu"]]

    def atualizar_tudo(self):
        desenhar_figura(self.fig, self.conjunto, self.resultados)
        self.canvas.draw_idle()
        self.tabela_parametros()
        self.tabela_estatisticas()
        self.tabela_succao()
        self.estimar_pontual(silencioso=True)
        self.tabela_dados()
        self.tabela_solos()
        self.processo_calculo()

    # --------------------------------------------------------------- tabelas
    def tabela_parametros(self):
        linhas, tags = [], []
        for k in ORDEM:
            if k not in self.resultados:
                continue
            M, r = MODELOS[k], self.resultados[k]
            if not r["convergiu"]:
                linhas.append([M["nome"], "—", "não convergiu", "—", "—", "—", "—", r["mensagem"]])
                tags.append(None)
                continue
            for i, nome in enumerate(M["pars"]):
                linhas.append([M["nome"] if i == 0 else "", nome, g5(r["p"][i]), g5(r["ep"][i]),
                               g5(r["tstat"][i]), fmt_p(r["p_par"][i]), M["unid"][i], M["signif"][i]])
                tags.append("destaque" if np.isfinite(r["p_par"][i]) and r["p_par"][i] < 0.05 else None)
        self._preencher(self.tv_par, ["Modelo", "Parâmetro", "Valor", "Erro padrão", "t", "p-valor",
                                      "Unidade", "Significado"], linhas,
                        [170, 80, 110, 110, 90, 90, 80, 330], tags)
        self.tv_par.column("c7", anchor="w")
        self.tv_par.heading("c7", anchor="w")

    def tabela_estatisticas(self):
        for w in self.fr_kpis.winfo_children():
            w.destroy()
        conv = self._conv()
        res = self.resultados
        self.lb_alerta.configure(text="")
        if conv:
            m_r2 = max(conv, key=lambda k: res[k]["r2"])
            m_aic = min(conv, key=lambda k: res[k]["aic"])
            kpis = [("MELHOR POR R²", MODELOS[m_r2]["curto"], f"R² = {res[m_r2]['r2']:.5f}", MODELOS[m_r2]["cor"]),
                    ("MELHOR POR AICc", MODELOS[m_aic]["curto"], f"AICc = {res[m_aic]['aic']:.2f}", MODELOS[m_aic]["cor"]),
                    ("MENOR RMSE", f"{min(res[k]['rmse'] for k in conv):.5f}", "cm³/cm³", TINTA),
                    ("PONTOS USADOS", str(len(self.conjunto["psi"])), "mínimo exigido: 5", TINTA)]
            for i, (rot, val, sub, cor) in enumerate(kpis):
                f = ttk.Frame(self.fr_kpis, style="Kpi.TFrame", padding=(12, 8))
                f.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
                self.fr_kpis.columnconfigure(i, weight=1)
                ttk.Label(f, text=rot, style="KpiR.TLabel").pack(anchor="w")
                ttk.Label(f, text=val, style="KpiV.TLabel", foreground=cor).pack(anchor="w")
                ttk.Label(f, text=sub, style="KpiR.TLabel").pack(anchor="w")
            apertados = [k for k in conv if len(self.conjunto["psi"]) - res[k]["n_par"] < 2]
            if apertados:
                self.lb_alerta.configure(text=(
                    "⚠ Atenção ao interpretar o R²: com " + str(len(self.conjunto["psi"])) +
                    " pontos, " + " e ".join(f"{MODELOS[k]['curto']} ({res[k]['n_par']} parâmetros)" for k in apertados) +
                    " ficam com menos de 2 graus de liberdade. Nessa situação a curva praticamente "
                    "interpola os pontos; use o AICc para comparar."))

        linhas, tags = [], []
        m_r2 = max(conv, key=lambda k: res[k]["r2"]) if conv else None
        for k in ORDEM:
            if k not in res:
                continue
            r = res[k]
            if not r["convergiu"]:
                linhas.append([MODELOS[k]["nome"], "—", "não convergiu", "—", "—", "—", "—"])
                tags.append(None)
                continue
            linhas.append([MODELOS[k]["nome"], r["n_par"], f"{r['r2']:.5f}",
                           f"{r['r2aj']:.5f}" if np.isfinite(r["r2aj"]) else "—",
                           f"{r['rmse']:.5f}", f"{r['mae']:.5f}", f"{r['aic']:.2f}"])
            tags.append("destaque" if k == m_r2 else None)
        self._preencher(self.tv_est, ["Modelo", "N.º de parâmetros", "R²", "R² ajustado",
                                      "RMSE (cm³/cm³)", "MAE (cm³/cm³)", "AICc"], linhas,
                        [190, 120, 110, 110, 130, 130, 100], tags)

        q = qualificacao(res)
        self._preencher(self.tv_rank, ["Colocação", "Modelo", "R² ajust.", "RMSE", "MAE", "AICc",
                                       "Média das posições", "Qualificação"],
                        [[f"{i + 1}º", MODELOS[x["chave"]]["nome"], *[f"{p}º" for p in x["posicoes"]],
                          f"{x['media']:.2f}", x["nota"]] for i, x in enumerate(q)],
                        [90, 190, 90, 80, 80, 80, 150, 120], ["destaque"] + [None] * (len(q) - 1))
        self.tv_rank.column("c1", anchor="w")

        linhas = []
        for k in conv:
            M, r = MODELOS[k], res[k]
            t10, t1500 = float(M["f"](10, *r["p"])), float(M["f"](1500, *r["p"]))
            linhas.append([M["nome"], fmt(M["entrada_ar"](r["p"])), fmt(t10), fmt(t1500), fmt(t10 - t1500)])
        self._preencher(self.tv_ind, ["Modelo", "Entrada de ar (kPa)", "θ a 10 kPa (CC)", "θ a 1500 kPa (PMP)",
                                      "Água disponível (cm³/cm³)"], linhas, [190, 160, 150, 160, 190])

    def tabela_succao(self):
        conv = self._conv()
        if not conv:
            self._preencher(self.tv_suc, ["θ (cm³/cm³)"], [])
            return
        lo, hi = -math.inf, math.inf
        for k in conv:
            M, p = MODELOS[k], self.resultados[k]["p"]
            tr = p[0] if M["pars"][0] == "θr" else 0.0
            ts = p[1] if M["pars"][0] == "θr" else p[0]
            lo, hi = max(lo, tr + 1e-4), min(hi, ts - 1e-4)
        linhas = []
        for i in range(11):
            th = hi - (hi - lo) * i / 10
            linha = [f"{th:.4f}"]
            for k in conv:
                v = MODELOS[k]["inversa"](th, self.resultados[k]["p"])
                linha.append(fmt(v, 3 if v < 10 else 1) if np.isfinite(v) else "—")
            linhas.append(linha)
        self._preencher(self.tv_suc, ["θ (cm³/cm³)"] + [f"ψ {MODELOS[k]['curto']} (kPa)" for k in conv],
                        linhas, [140] + [200] * len(conv))

    def estimar_pontual(self, silencioso=False):
        for w in self.fr_suc.winfo_children():
            w.destroy()
        conv = self._conv()
        if not conv:
            return
        try:
            th = float(self.var_alvo.get().replace(",", "."))
        except ValueError:
            if not silencioso:
                messagebox.showwarning("Sucção estimada", "Informe um teor de umidade válido, por exemplo 0,25.")
            return
        for i, k in enumerate(conv):
            v = MODELOS[k]["inversa"](th, self.resultados[k]["p"])
            f = ttk.Frame(self.fr_suc, style="Kpi.TFrame", padding=(12, 8))
            f.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
            self.fr_suc.columnconfigure(i, weight=1)
            ttk.Label(f, text=MODELOS[k]["curto"].upper(), style="KpiR.TLabel").pack(anchor="w")
            ttk.Label(f, text=g5(v) if np.isfinite(v) else "—", style="KpiV.TLabel",
                      foreground=MODELOS[k]["cor"]).pack(anchor="w")
            ttk.Label(f, text="kPa" if np.isfinite(v) else "fora da faixa do modelo",
                      style="KpiR.TLabel").pack(anchor="w")
        if not silencioso:
            self.processo_calculo()

    def tabela_dados(self):
        c = self.conjunto
        conv = self._conv()
        cab = ["n", "Sucção ψ (kPa)", "θ observado (cm³/cm³)"]
        for k in conv:
            cab += [f"θ {MODELOS[k]['curto']}", f"Resíduo {MODELOS[k]['curto']}"]
        linhas = []
        for i, x in enumerate(c["psi"]):
            l = [i + 1, fmt(x, 3 if x < 10 else 1), f"{c['theta'][i]:.4f}"]
            for k in conv:
                calc = self.resultados[k]["calc"][i]
                l += [f"{calc:.4f}", f"{c['theta'][i] - calc:.5f}"]
            linhas.append(l)
        self._preencher(self.tv_dados, cab, linhas, [50, 130, 170] + [140] * (2 * len(conv)))

    def tabela_solos(self):
        atual = self.conjunto.get("classe") if self.conjunto else None
        linhas, tags = [], []
        for k, (tr, ts, al, n) in CARSEL.items():
            linhas.append([k + ("  ◄ em uso" if k == atual else ""), f"{tr:.3f}", f"{ts:.2f}",
                           f"{al:.3f}", f"{n:.2f}", f"{1 - 1 / n:.3f}"])
            tags.append("destaque" if k == atual else None)
        self._preencher(self.tv_solos, ["Classe textural (USDA)", "θr (cm³/cm³)", "θs (cm³/cm³)",
                                        "α (1/cm)", "n", "m = 1 − 1/n"], linhas,
                        [240, 120, 120, 120, 100, 120], tags)

    # ------------------------------------------------------- processo de cálculo
    def processo_calculo(self):
        t = self.txt_pc
        t.configure(state="normal")
        t.delete("1.0", "end")
        conv = self._conv()
        if not conv:
            t.insert("end", "Clique em “Ajustar modelos” para ver o processo de cálculo com os seus dados.", "p")
            t.configure(state="disabled")
            return

        c, res = self.conjunto, self.resultados
        n, psi, th = len(c["psi"]), c["psi"], c["theta"]
        r0 = res[conv[0]]

        def passo(i, titulo):
            t.insert("end", "\n" if i > 1 else "", "sep")
            t.insert("end", f"  {i}  ", ("h", "num"))
            t.insert("end", f"  {titulo}\n", "h")

        def par(texto):
            t.insert("end", texto + "\n", "p")

        def eq(*linhas):
            for l in linhas:
                if " = " in l and l.rstrip().endswith("§"):
                    l = l.rstrip()[:-1]
                    pos = l.rfind(" = ")
                    t.insert("end", "  " + l[:pos + 3], "eq")
                    t.insert("end", l[pos + 3:] + "  \n", ("eq", "res"))
                else:
                    t.insert("end", "  " + l + "  \n", "eq")
            t.insert("end", "\n", "sep")

        def modelo(k):
            t.insert("end", MODELOS[k]["nome"] + "\n", "m_" + k)

        def tabela(cab, linhas):
            larg = [max(len(str(x)) for x in [h] + [l[i] for l in linhas]) for i, h in enumerate(cab)]
            t.insert("end", "  ".join(str(h).rjust(larg[i]) for i, h in enumerate(cab)) + "\n", "tabcab")
            for l in linhas:
                t.insert("end", "  ".join(str(x).rjust(larg[i]) for i, x in enumerate(l)) + "\n", "tab")
            t.insert("end", "\n", "p")

        t.insert("end", "Contas reais feitas com os dados carregados agora, do passo 1 ao 10. Cada passo traz "
                        "a equação, a substituição com os seus números e o resultado (em azul).\n", "p")

        # 1
        passo(1, "Dados de entrada")
        par(f"Foram informados N = {n} pares (ψ, θ), com ψ entre {g5(psi.min())} e {g5(psi.max())} kPa "
            f"e θ entre {g5(th.min())} e {g5(th.max())} cm³/cm³.")
        eq(f"θm (média) = Σθ / N = {g5(th.sum())} / {n} = {g5(r0['media'])} cm³/cm³§",
           f"SQT = Σ(θᵢ − θm)² = {g5(r0['sqt'])}§")

        # 2
        passo(2, "Verificação e tratamento dos dados")
        if c["avisos"]:
            for a in c["avisos"]:
                par("• " + a)
        else:
            par("Nenhuma correção foi necessária: ψ > 0, 0 ≤ θ ≤ 1, pontos ordenados por sucção e N ≥ 5.")

        # 3
        passo(3, "Equações dos modelos")
        par("Cada modelo descreve θ em função de ψ. Os parâmetros são as incógnitas a determinar.")
        for k in conv:
            modelo(k)
            eq(MODELOS[k]["eq"], "parâmetros: " + ", ".join(MODELOS[k]["pars"]))
        if "fredlund_xing" in conv:
            eq("C(ψ) = 1 − ln(1 + ψ/ψr) / ln(1 + 10⁶/ψr),  com ψr = 10⁶ kPa")

        # 4
        passo(4, "Ajuste por mínimos quadrados não lineares")
        par("Os parâmetros β são os que minimizam a soma dos quadrados dos resíduos:")
        eq("S(β) = SQR = Σ [ θᵢ − f(ψᵢ; β) ]²")
        par("A minimização usa scipy.optimize.curve_fit com o algoritmo Trust Region Reflective (TRF), "
            "que respeita limites físicos para os parâmetros. O ajuste parte de 3 ou 4 estimativas "
            "iniciais obtidas dos próprios dados; adota-se a solução de menor SQR.")
        for k in conv:
            M, r = MODELOS[k], res[k]
            modelo(k)
            tabela(["Partida", "Valores iniciais (" + ", ".join(M["pars"]) + ")", "SQR final"],
                   [[f"{i + 1}{' ◄' if i == r['idx_melhor'] else ''}", " · ".join(g5(v) for v in q["inicio"]),
                     g5(q["sqr"])] for i, q in enumerate(r["partidas"])])
            eq(f"SQR mínima = {g5(r['sqr'])}§",
               *[f"{nm} = {g5(r['p'][i])} {'' if M['unid'][i] == '—' else M['unid'][i]}§"
                 for i, nm in enumerate(M["pars"])])

        # 5
        passo(5, "Curvas ajustadas e resíduos")
        par("Substituindo os parâmetros ótimos nas equações do passo 3:")
        for k in conv:
            p = res[k]["p"]
            modelo(k)
            if k == "gardner":
                eq(f"θ(ψ) = {g5(p[0])} + ({g5(p[1])} − {g5(p[0])})·exp(−{g5(p[2])}·ψ)")
            elif k == "van_genuchten":
                eq(f"θ(ψ) = {g5(p[0])} + ({g5(p[1])} − {g5(p[0])}) / [1 + ({g5(p[2])}·ψ)^{g5(p[3])}]^{g5(p[4])}")
            else:
                eq(f"θ(ψ) = C(ψ)·{g5(p[0])} / {{ln[e + (ψ/{g5(p[1])})^{g5(p[2])}]}}^{g5(p[3])}")
        par("Resíduo = θ observado − θ calculado, em cada ponto:")
        cab = ["n", "ψ (kPa)", "θ obs."]
        for k in conv:
            cab += [f"θ {MODELOS[k]['curto']}", "resíduo"]
        linhas = []
        for i in range(n):
            l = [i + 1, g5(psi[i]), f"{th[i]:.5f}"]
            for k in conv:
                l += [f"{res[k]['calc'][i]:.5f}", f"{th[i] - res[k]['calc'][i]:.5f}"]
            linhas.append(l)
        linhas.append(["", "", "SQR=Σres²"] + sum([["", g5(res[k]["sqr"])] for k in conv], []))
        tabela(cab, linhas)

        # 6
        passo(6, "Erro padrão dos parâmetros")
        par("O jacobiano J reúne as derivadas de f em relação a cada parâmetro, em cada ponto "
            "(diferenças centrais). A covariância das estimativas é:")
        eq("Cov(β) = s² · (JᵀJ)⁻¹ ,  com  s² = SQR / (N − p)", "EP(βⱼ) = √( s² · [(JᵀJ)⁻¹]ⱼⱼ )")
        for k in conv:
            M, r = MODELOS[k], res[k]
            modelo(k)
            par(f"J tem dimensão {n} × {r['n_par']}. Matriz JᵀJ:")
            tabela([""] + M["pars"], [[M["pars"][i]] + [g5(v) for v in row] for i, row in enumerate(r["JtJ"])])
            eq(f"s² = {g5(r['sqr'])} / ({n} − {r['n_par']}) = {g5(r['sqr'])} / {r['gl']} = {g5(r['s2'])}§",
               *[f"EP({nm}) = √({g5(r['s2'])} · {g5(r['inv'][i][i])}) = {g5(r['ep'][i])}§"
                 for i, nm in enumerate(M["pars"])])

        # 7
        passo(7, "Estatística t e p-valor de cada parâmetro")
        par("Testa-se a hipótese nula de que o parâmetro é igual a zero:")
        eq("t = β / EP(β)", "p = 2 · [ 1 − Fₜ(|t| ; N − p) ]")
        for k in conv:
            M, r = MODELOS[k], res[k]
            modelo(k)
            eq(*[f"{nm}:  t = {g5(r['p'][i])} / {g5(r['ep'][i])} = {g5(r['tstat'][i])}   →   "
                 f"p = {fmt_p(r['p_par'][i])}   ({r['gl']} graus de liberdade)" for i, nm in enumerate(M["pars"])])
        par("Com p < 0,05 o parâmetro é estatisticamente diferente de zero.")

        # 8
        passo(8, "Indicadores de desempenho")
        eq("R² = 1 − SQR / SQT", "R²aj = 1 − (1 − R²)·(N − 1)/(N − p − 1)", "RMSE = √(SQR / N)",
           "MAE = Σ|θᵢ − θcalc,ᵢ| / N", "AICc = N·ln(SQR/N) + 2p + 2p(p + 1)/(N − p − 1)")
        for k in conv:
            r = res[k]
            npar = r["n_par"]
            corr = 2 * npar * (npar + 1) / (n - npar - 1) if n - npar - 1 > 0 else float("nan")
            modelo(k)
            eq(f"R² = 1 − {g5(r['sqr'])} / {g5(r['sqt'])} = {g5(r['r2'])}§",
               f"R²aj = 1 − (1 − {g5(r['r2'])})·{n - 1}/{n - npar - 1} = {g5(r['r2aj'])}§",
               f"RMSE = √({g5(r['sqr'])} / {n}) = {g5(r['rmse'])}§",
               f"MAE = {g5(r['soma_abs'])} / {n} = {g5(r['mae'])}§",
               f"AICc = {n}·ln({g5(r['sqr'])}/{n}) + {2 * npar} + {g5(corr)} = {g5(r['aic'])}§")

        # 9
        passo(9, "Qualificação dos modelos")
        par("Cada modelo recebe uma posição (1º = melhor) em R²aj, RMSE, MAE e AICc. A menor média das "
            "posições define a colocação.")
        q = qualificacao(res)
        tabela(["Colocação", "Modelo", "R²aj", "RMSE", "MAE", "AICc", "Média", "Qualificação"],
               [[f"{i + 1}º", MODELOS[x["chave"]]["nome"], *[f"{p}º" for p in x["posicoes"]],
                 f"{x['media']:.2f}", x["nota"]] for i, x in enumerate(q)])

        # 10
        passo(10, "Sucção matricial estimada")
        try:
            alvo = float(self.var_alvo.get().replace(",", "."))
        except ValueError:
            alvo = float("nan")
        if not np.isfinite(alvo):
            par("Informe um teor de umidade válido na aba “Sucção estimada”.")
        else:
            par(f"Para θ = {g5(alvo)} cm³/cm³ (valor da aba “Sucção estimada”), inverte-se cada modelo "
                "para obter ψ.")
            for k in conv:
                M, p = MODELOS[k], res[k]["p"]
                v = M["inversa"](alvo, p)
                modelo(k)
                if k == "gardner":
                    tr, ts, a = p
                    if tr < alvo <= ts:
                        s = (alvo - tr) / (ts - tr)
                        eq(f"s = (θ − θr)/(θs − θr) = ({g5(alvo)} − {g5(tr)}) / ({g5(ts)} − {g5(tr)}) = {g5(s)}",
                           f"ψ = −ln(s)/a = −ln({g5(s)}) / {g5(a)} = {g5(v)} kPa§")
                    else:
                        eq("θ fora do intervalo (θr; θs]: sem solução")
                elif k == "van_genuchten":
                    tr, ts, al, nn, m = p
                    if tr < alvo <= ts:
                        se = (alvo - tr) / (ts - tr)
                        tt = se ** (-1 / m) - 1
                        eq(f"Se = (θ − θr)/(θs − θr) = {g5(se)}",
                           f"t = Se^(−1/m) − 1 = {g5(se)}^({g5(-1 / m)}) − 1 = {g5(tt)}",
                           f"ψ = t^(1/n) / α = {g5(tt)}^({g5(1 / nn)}) / {g5(al)} = {g5(v)} kPa§")
                    else:
                        eq("θ fora do intervalo (θr; θs]: sem solução")
                else:
                    if np.isfinite(v):
                        eq(f"Sem inversa fechada: resolve-se θ(ψ) − {g5(alvo)} = 0 pelo método de Brent "
                           "(brentq), entre 10⁻⁴ e 10⁶ kPa",
                           f"ψ = {g5(v)} kPa§",
                           f"conferência: θ({g5(v)}) = {g5(float(M['f'](v, *p)))}")
                    else:
                        eq("θ fora da faixa do modelo: sem solução")
        t.configure(state="disabled")

    # --------------------------------------------------------------- exportação
    def exportar_csv(self):
        conv = self._conv()
        if not conv:
            messagebox.showwarning("Exportar", "Ajuste os modelos antes de exportar.")
            return
        caminho = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="resultados_CRAS.csv",
                                               filetypes=[("CSV", "*.csv")])
        if not caminho:
            return
        c, res = self.conjunto, self.resultados
        num = lambda v: str(v).replace(".", ",")
        L = ["RELATORIO DE AJUSTE DA CURVA DE RETENCAO DE AGUA NO SOLO",
             f"Conjunto;{c['nome']}", f"Pontos;{len(c['psi'])}",
             f"Data;{datetime.datetime.now():%d/%m/%Y %H:%M}", "",
             "PARAMETROS AJUSTADOS", "Modelo;Parametro;Valor;Erro padrao;t;p-valor;Unidade"]
        for k in conv:
            M, r = MODELOS[k], res[k]
            for i, nm in enumerate(M["pars"]):
                L.append(";".join([M["nome"], nm, num(r["p"][i]), num(r["ep"][i]), num(r["tstat"][i]),
                                   num(r["p_par"][i]), M["unid"][i]]))
        L += ["", "INDICADORES ESTATISTICOS", "Modelo;N parametros;R2;R2 ajustado;RMSE;MAE;AICc"]
        for k in conv:
            r = res[k]
            L.append(";".join([MODELOS[k]["nome"], str(r["n_par"])] +
                              [num(r[x]) for x in ("r2", "r2aj", "rmse", "mae", "aic")]))
        L += ["", "QUALIFICACAO DOS MODELOS", "Colocacao;Modelo;Media das posicoes;Qualificacao"]
        for i, x in enumerate(qualificacao(res)):
            L.append(f"{i + 1};{MODELOS[x['chave']]['nome']};{num(round(x['media'], 2))};{x['nota']}")
        L += ["", "VALORES OBSERVADOS E CALCULADOS",
              ";".join(["Succao_kPa", "theta_observado"] +
                       sum([[f"theta_{MODELOS[k]['curto']}", f"residuo_{MODELOS[k]['curto']}"] for k in conv], []))]
        for i, x in enumerate(c["psi"]):
            L.append(";".join([num(x), num(c["theta"][i])] +
                              sum([[num(res[k]["calc"][i]), num(c["theta"][i] - res[k]["calc"][i])] for k in conv], [])))
        try:
            with open(caminho, "w", encoding="utf-8-sig", newline="") as fh:
                fh.write("\n".join(L))
        except OSError as e:
            messagebox.showerror("Exportar", str(e))
            return
        self.status.set(f"Resultados salvos em {caminho}")

    def exportar_grafico(self, formato):
        if not self._conv():
            messagebox.showwarning("Exportar", "Ajuste os modelos antes de exportar.")
            return
        caminho = filedialog.asksaveasfilename(defaultextension="." + formato,
                                               initialfile=f"curva_retencao.{formato}",
                                               filetypes=[(formato.upper(), "*." + formato)])
        if not caminho:
            return
        fig = Figure(figsize=(10, 7.4), dpi=100)
        desenhar_figura(fig, self.conjunto, self.resultados, titulo=True)
        try:
            fig.savefig(caminho, dpi=300 if formato == "png" else None, facecolor="white")
        except OSError as e:
            messagebox.showerror("Exportar", str(e))
            return
        self.status.set(f"Gráfico salvo em {caminho}")


def main():
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("CRAS.CurvaRetencao.1")
        except Exception:
            pass
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
