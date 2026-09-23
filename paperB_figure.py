# -*- coding: utf-8 -*-
"""Figure e tabella di verifica per il Paper B (Applied Mathematical Finance).

Produce contenuti NUOVI, non riciclati dal Paper A:
  - Figure B1: profilo del modulatore g''(a/tau)/(K tau^2) per tre lisciature, e variazione
    totale integrata in funzione di tau (pendenza -1 in scala logaritmica). Illustra il
    Teorema 3.3(iv): la sensibilita' puntuale al benchmark diverge come tau^-2, il supporto si
    contrae come tau, l'influenza integrata cresce come tau^-1.
  - Figure B2: frazione di segnale attivo che sopravvive al filtro softmax in funzione della
    dispersione trasversale del Jacobiano pesato. Illustra il Corollario 3.5: un Jacobiano pesato uniforme tra asset annulla
    il segnale ai logits; traiettorie identiche costituiscono un caso esatto.
  - Tabella 1: esito della verifica numerica dei teoremi (test, errore osservato).

Uscite: pubblicazione/figure_paperB/figureB1_modulator.{png,pdf}, figureB2_filter.{png,pdf},
        pubblicazione/paperB_tabella_verifica.md
"""

import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import verifica_teoremi_paperB_v2 as v   # noqa: E402

OUT = 'pubblicazione/figure_paperB'
TAB = 'pubblicazione/paperB_tabella_verifica.md'

plt.rcParams.update({
    'figure.dpi': 500, 'savefig.dpi': 500,
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
})
LARGHEZZA = 140 / 25.4      # 140 mm, larghezza di colonna tipica


def sigma(x):
    return 1.0 / (1.0 + np.exp(-x))


def salva(fig, nome):
    # EPS vettoriale oltre a PNG e PDF per riuso editoriale senza perdita di qualita'.
    os.makedirs(OUT, exist_ok=True)
    for est in ('png', 'pdf', 'eps'):
        fig.savefig(os.path.join(OUT, '%s.%s' % (nome, est)), bbox_inches='tight')
    plt.close(fig)
    print('  scritta %s/%s.{png,pdf,eps}' % (OUT, nome))


def figura_modulatore(K=4, J=1.0):
    """Teorema 3.3(iii)-(iv)."""
    fig, axes = plt.subplots(1, 2, figsize=(LARGHEZZA, 2.5))
    a = np.linspace(-0.25, 0.25, 4001)
    for tau, stile in ((0.005, '-'), (0.02, '--'), (0.08, ':')):
        x = -a / tau                          # argomento (b - u)/tau con a = u - b
        mod = sigma(x) * (1 - sigma(x)) * J / (K * tau ** 2)
        axes[0].plot(a, mod, stile, lw=1.2, label=r'$\tau = %g$' % tau)
    axes[0].set_xlabel(r'block excess $a_l = u_l - b_l$')
    axes[0].set_ylabel(r'$|\partial^2 L / \partial b_l \partial w_i|$')
    axes[0].set_title('Modulator: where the benchmark acts')
    axes[0].set_yscale('log')
    # taglio in basso: sotto questa soglia si vedrebbe solo l'underflow del logistico in float
    axes[0].set_ylim(1e-6, None)
    axes[0].legend(frameon=False)

    # pannello destro: le DUE leggi di scala vanno distinte (Teorema 3.3(iv) corretto).
    # Il sup sulla traiettoria del benchmark diverge come tau^-2; la variazione totale come
    # tau^-1; a benchmark FISSATO con eccesso non nullo la sensibilita' va invece a zero.
    taus = np.logspace(-3, -0.7, 40)
    tv, sup, punt = np.empty_like(taus), np.empty_like(taus), np.empty_like(taus)
    a_fisso = 0.05
    for n, tau in enumerate(taus):
        griglia = np.linspace(-60 * tau, 60 * tau, 20001)
        val = sigma(-griglia / tau) * (1 - sigma(-griglia / tau)) * J / (K * tau ** 2)
        tv[n] = np.trapezoid(val, griglia) if hasattr(np, 'trapezoid') else np.trapz(val, griglia)
        sup[n] = val.max()
        punt[n] = sigma(-a_fisso / tau) * (1 - sigma(-a_fisso / tau)) * J / (K * tau ** 2)
    axes[1].loglog(taus, sup, 'o', ms=2.5, label=r'sup, $\kappa = 1$')
    axes[1].loglog(taus, J / (4 * K * taus ** 2), '-', lw=0.9,
                   label=r'$|J| / (4K\tau^{2})$')
    axes[1].loglog(taus, tv, 's', ms=2.5, label=r'variation, $\kappa = 1$')
    axes[1].loglog(taus, J / (K * taus), '--', lw=0.9, label=r'$|J| / (K\tau)$')
    axes[1].loglog(taus, taus * sup, 'v', ms=2.5, label=r'sup, $\kappa = \tau$')
    axes[1].loglog(taus, taus * tv, 'd', ms=2.5, label=r'variation, $\kappa = \tau$')
    axes[1].loglog(taus, np.full_like(taus, J / K), ':', lw=0.9, label=r'$|J| / K$')
    axes[1].loglog(taus, punt, '^', ms=2.5, label=r'fixed $b$, $a_l = %.2f$' % a_fisso)
    axes[1].set_xlabel(r'smoothing width $\tau$')
    axes[1].set_ylabel('benchmark sensitivity')
    axes[1].set_title('Scalings depend on the normalisation')
    axes[1].legend(frameon=False, fontsize=5.6, ncol=2)
    fig.tight_layout()
    salva(fig, 'figureB1_modulator')


def figura_filtro():
    """Corollario 3.5 corretto: rho incorniciato fra sqrt(w_min) delta e sqrt(w_max) delta,
    dove delta e' la dispersione NORMALIZZATA (invariante di scala). La versione precedente
    usava una dispersione non normalizzata, con cui l'enunciato era mal posto."""
    A_pieno = v.rendimenti()
    A_id = v.rendimenti(identici=True)
    B = v.blocchi()
    w = v.softmax(np.linspace(-0.4, 0.6, v.N))
    S = np.diag(w) - np.outer(w, w)
    delta, rho = [], []
    for q in np.linspace(0.0, 1.0, 61):
        A = (1 - q) * A_id + q * A_pieno       # da asset identici a pienamente idiosincratici
        J = v.jacobiano(w, A, B)
        u = v.u_di_w(w, A, B)
        b = u - np.array([0.031, -0.017, 0.052, -0.008])
        Jp = (J * sigma((b - u) / v.TAU)[:, None]).sum(axis=0)
        n_j = np.linalg.norm(Jp)
        if n_j == 0:
            continue
        var_w = float(w @ (Jp - w @ Jp) ** 2)
        delta.append(np.sqrt(var_w) / n_j)
        rho.append(np.linalg.norm(S @ Jp) / n_j)
    delta, rho = np.array(delta), np.array(rho)
    ordine = np.argsort(delta)
    delta, rho = delta[ordine], rho[ordine]

    fig, ax = plt.subplots(figsize=(LARGHEZZA * 0.66, 2.4))
    ax.plot(delta, rho, 'o', ms=2.5, label=r'$\rho = \|S\widehat{J}\|/\|\widehat{J}\|$')
    ax.plot(delta, np.sqrt(w.max()) * delta, '-', lw=1.0,
            label=r'$\sqrt{w_{\max}}\,\delta$')
    ax.plot(delta, np.sqrt(w.min()) * delta, '--', lw=1.0,
            label=r'$\sqrt{w_{\min}}\,\delta$')
    ax.set_xlabel(r'normalised dispersion $\delta = \sqrt{\mathrm{Var}_w(\widehat{J})}/\|\widehat{J}\|$')
    ax.set_ylabel('surviving fraction')
    ax.set_title('The parametrization filters the active signal')
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    salva(fig, 'figureB2_filter')


# Etichette inglesi per la tabella. La suite v2 porta gia' enunciato, tolleranza e scarto nella
# tupla di esito, quindi qui serve solo la descrizione di CIO' CHE SI VERIFICA.
DESCRIZIONI = {
    'T0': ('Block-return Jacobian', 'closed form against central finite differences'),
    'T1a': ('Theorem 3.1(a)', 'cross-curvature vanishes exactly for a separable objective'),
    'T1b': ('Theorem 3.1(b)', 'ambient gradient moves with the benchmark, its tangential '
                              'projection does not'),
    'T1c': ('Theorem 3.1(b), converse', 'an active objective moves the tangential projection'),
    'T1d': ('Lemma 3.4', 'the kernel identity $S\\mathbf{1} = w(1-\\mathbf{1}^{\\top}w)$ and '
                         'the loss of the filtering property off the simplex'),
    'T12': ('Coordinate convention', 'derivative taken inside the simplex equals the tangential '
                                     'component of the ambient gradient'),
    'T2a': ('Theorem 3.2(2)', 'gradient equals the Jacobian row of the selected rank'),
    'T2b': ('Theorem 3.2(2)', 'gradient constant within a cell of the arrangement'),
    'T3': ('Theorem 3.2(3)', 'jump equals $(c_m - c_{m\'})(J_j - J_i)$, checked from both sides'),
    'T4': ('Theorem 3.2(3)', 'no jump when the exchanged ranks carry equal weight'),
    'T11': ('Theorem 3.2(4)', 'full row rank is sufficient for activation; a fixed null vector '
                              'gives degeneracy at one weight only; repeated sub-periods give '
                              'inertness throughout a cell'),
    'T5': ('Theorem 3.3(1)', 'smoothed gradient against finite differences'),
    'T6': ('Theorem 3.3(2)', 'modulator form, checked at all block-asset pairs'),
    'T7': ('Theorem 3.3(3)', r'total variation equals $\lvert J_{l,i}\rvert/(K\tau)$'),
    'T7b': ('Theorem 3.3(3) and (4)', 'log-log slopes at $\\kappa = 1$; decay at fixed benchmark'),
    'T7c': ('Theorem 3.3(4)', 'slopes under both normalisations, $\\tau$-independence of the '
                              'total variation at $\\kappa = \\tau$, invariance of the ratio'),
    'T8': ('Lemma 3.4', 'symmetry, positive semidefiniteness, kernel, rank $N-1$, variance form'),
    'T9': ('Corollary 3.5(1)', 'a uniform weighted Jacobian is annihilated'),
    'T9b': ('Corollary 3.5(2)', 'two-sided bound by the weighted cross-sectional variance'),
    'T9c': ('Corollary 3.5(3)', 'scale invariance of $\\rho$ over nine orders of magnitude'),
    'T10': ('Corollary 3.5(4)', 'identical asset paths silence the objective exactly'),
}


def tabella_verifica():
    v.main()
    righe = ['**Table 1.** Numerical verification of analytical results in Sections 3 '
             'and 4 using central finite differences, algebraic identities, rank and kernel '
             'diagnostics, and scaling checks as appropriate. Synthetic single-factor returns, '
             'base seed %d (additional fixed seeds for targeted checks), %d assets, %d days, '
             '%d blocks, smoothing width %.3f. Test criteria and tolerances are specified '
             'in the replication code.'
             % (v.SEME, v.N, v.T, v.K, v.TAU),
             '',
             '| Test | Statement | What is checked | Tolerance | Discrepancy | Result |',
             '|---|---|---|---|---|---|']
    for nome, ok, _enunciato_it, tolleranza, discrepanza, _dettaglio in v.esiti:
        enunciato, cosa = DESCRIZIONI.get(nome, ('-', '-'))
        # Presentation-only English labels; do not modify numerical test criteria.
        if nome == 'T7':
            tolleranza = '1e-3 relative error'
        elif nome in ('T7b', 'T7c'):
            tolleranza = '0.02 slope deviation; 1e-3 relative error'
        righe.append('| %s | %s | %s | %s | %s | %s |'
                     % (nome, enunciato, cosa, tolleranza, discrepanza,
                        'pass' if ok else 'FAIL'))
    n_ok = sum(1 for e in v.esiti if e[1])
    righe += ['', '*Note.* Four checks are retained as regression '
                  'checks because they discriminate between statements that a less careful '
                  'treatment would conflate: T1b realises an objective whose learning signal is '
                  'benchmark-free although its ambient gradient is not; T1d records that the '
                  'filtering property of the softmax Jacobian holds on the simplex and nowhere '
                  'else, which is why the hypotheses of Theorem 3.1(b) are stated there; T11 '
                  'separates the two ways the rank condition can fail, since an objective built '
                  'on a null vector of the Jacobian at one weight is degenerate there and active '
                  'at neighbouring weights of the same cell, whereas repeating a sub-period makes '
                  'a non-affine function of the order statistics inert throughout the cell; and '
                  'T9c records the scale invariance '
                  'that makes an unnormalised surviving-fraction ratio uninformative. The script '
                  'that produces this table is part of the replication material.']
    # la nota non ripete piu' il conteggio, che resta nel corpo del manoscritto; l'assert
    # continua a impedire che tabella e testo si scollino
    assert n_ok == 21, 'il manoscritto dichiara ventuno controlli, la suite ne ha superati %d' % n_ok
    open(TAB, 'w', encoding='utf-8').write('\n'.join(righe) + '\n')
    print('  scritta %s (%d test)' % (TAB, len(v.esiti)))


if __name__ == '__main__':
    print('verifica numerica:')
    tabella_verifica()
    print('figure:')
    figura_modulatore()
    figura_filtro()
