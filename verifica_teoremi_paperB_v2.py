# -*- coding: utf-8 -*-
"""Verifica numerica dei teoremi del Paper B, versione 2 (dopo l'audit matematico del 03/09).

Sostituisce verifica_teoremi_paperB.py, che resta sul disco come traccia di audit. Le differenze
sostanziali rispetto alla v1 riguardano gli enunciati corretti, non il codice:

  - convenzione di dominio: l'obiettivo e' definito su un aperto CONVESSO di R^N (formulazione
    ambientale) e il simplesso entra solo attraverso la parametrizzazione softmax. Le derivate
    rispetto a w sono derivate ambientali; cio' che il training vede e' la loro proiezione
    tangenziale S grad_w L. Nuovo test T12 sulla coerenza fra le due letture;
  - nuova caratterizzazione dell'inerzia EFFETTIVA (quella che conta per il training):
    S grad_w L e' invariante in b se e solo se L(w;b) = F(w) + G(1'w, b). Nuovi test T1b e T1c;
  - Teorema 3.2(4) richiede Lambda di classe C2 e la condizione di rango sullo Jacobiano dei
    blocchi. Nuovo test T11 con CONTROESEMPIO quando i blocchi sono piu' degli asset;
  - Corollary 3.5 riformulato come "se e solo se" piu' limitazione bilaterale con la varianza
    pesata, invece di un rapporto che poteva essere 0/0. Nuovi test T9b e T9c (caso limite in cui
    numeratore e denominatore vanno a zero insieme);
  - legge di scala in tau: si distingue il SUP sulla traiettoria del benchmark (che diverge come
    tau^-2) dal valore a b FISSATO con eccesso non nullo (che va a zero). Nuovo test T7b su due
    ordini di grandezza.

Tolleranze: dichiarate nella tabella finale, una per test. Semi: SEME (dati) e SEME_B (benchmark).
Nessun sistema allenabile, nessun parametro di produzione, nessun dato reale.
"""

import numpy as np

SEME = 20260902
N = 6            # asset
T = 84           # giorni
K = 4            # blocchi (K <= N: serve alla condizione di rango del Teorema 3.2)
TAU = 0.02
esiti = []       # (nome, ok, enunciato, tolleranza, discrepanza, dettaglio)


def registra(nome, ok, enunciato, tolleranza, discrepanza, dettaglio=''):
    esiti.append((nome, bool(ok), enunciato, tolleranza, discrepanza, dettaglio))
    print('%-6s %-8s %-34s tol %-9s scarto %-11s %s'
          % (nome, 'PASS' if ok else 'FALLITO', enunciato, tolleranza, discrepanza, dettaglio))


# ── strumenti di base ───────────────────────────────────────────────────────────────────────
def rendimenti(n_asset=N, identici=False, seme=SEME):
    rng = np.random.default_rng(seme)
    fattore = rng.normal(0.0004, 0.010, size=T)
    if identici:
        return np.repeat(fattore[:, None], n_asset, axis=1)
    beta = rng.uniform(0.6, 1.4, size=n_asset)
    idio = rng.normal(0.0, 0.006, size=(T, n_asset))
    return fattore[:, None] * beta[None, :] + idio


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def blocchi():
    m = T // K
    return [np.arange(k * m, (k + 1) * m) for k in range(K)]


def u_di_w(w, A, B):
    rp = A @ w
    return np.array([np.prod(1.0 + rp[b]) - 1.0 for b in B])


def jacobiano(w, A, B):
    rp = A @ w
    J = np.zeros((len(B), A.shape[1]))
    for k, b in enumerate(B):
        for t in b:
            J[k, :] += A[t, :] * np.prod(1.0 + rp[b[b != t]])
    return J


def grad_num(f, w, h=1e-7):
    """Gradiente AMBIENTALE: ogni componente varia da sola, senza vincolo di simplesso."""
    g = np.zeros(len(w))
    for i in range(len(w)):
        wp, wm = w.copy(), w.copy()
        wp[i] += h
        wm[i] -= h
        g[i] = (f(wp) - f(wm)) / (2 * h)
    return g


def curvatura_mista(L, w, b, l, i, h=1e-5):
    def dLdw(bb):
        wp, wm = w.copy(), w.copy()
        wp[i] += h
        wm[i] -= h
        return (L(wp, bb) - L(wm, bb)) / (2 * h)
    bp, bm = b.copy(), b.copy()
    bp[l] += h
    bm[l] -= h
    return (dLdw(bp) - dLdw(bm)) / (2 * h)


def sigma(x):
    return 1.0 / (1.0 + np.exp(-x))


def matrice_S(w):
    return np.diag(w) - np.outer(w, w)


def main():
    A = rendimenti()
    B = blocchi()
    z = np.linspace(-0.4, 0.6, N)
    w = softmax(z)
    S = matrice_S(w)
    J = jacobiano(w, A, B)
    u = u_di_w(w, A, B)
    b0 = u - np.array([0.031, -0.017, 0.052, -0.008])          # eccessi distinti, cella interna

    # ── T0 Jacobiano dei blocchi ────────────────────────────────────────────────────────────
    err = max(np.abs(grad_num(lambda x: u_di_w(x, A, B)[k], w) - J[k]).max() for k in range(K))
    registra('T0', err < 1e-6, 'Jacobiano J[k,i] dei blocchi', '1e-6', '%.2e' % err,
             'forma analitica contro differenze finite centrali')

    # ── T1a Teorema 3.1(a): ostruzione ambientale ───────────────────────────────────────────
    def L_separabile(x, b):
        return -u_di_w(x, A, B).mean() + b.mean()

    def L_attivo(x, b):
        return np.logaddexp(0.0, (b - u_di_w(x, A, B)) / TAU).mean()

    cm_sep = max(abs(curvatura_mista(L_separabile, w, b0, l, i))
                 for l in range(K) for i in range(N))
    cm_att = max(abs(curvatura_mista(L_attivo, w, b0, l, i))
                 for l in range(K) for i in range(N))
    registra('T1a', cm_sep < 1e-4 < cm_att, 'Teorema 3.1(a)', '1e-4', '%.2e' % cm_sep,
             'separabile: curvatura mista nulla; attivo: %.3f' % cm_att)

    # ── T1b Teorema 3.1(b): inerzia EFFETTIVA (nuovo) ───────────────────────────────────────
    # L = F(w) + G(1'w, b) e' ambientalmente ATTIVA ma effettivamente INERTE: il benchmark
    # entra solo lungo la direzione uniforme, che il Jacobiano del softmax annulla.
    def L_effettiva(x, b):
        s = x.sum()
        return -u_di_w(x, A, B).mean() + (0.7 + b.mean()) * s + 0.4 * b[1] * s ** 2

    b1 = b0 + np.array([0.004, -0.003, 0.006, -0.002])
    g0 = grad_num(lambda x: L_effettiva(x, b0), w)
    g1 = grad_num(lambda x: L_effettiva(x, b1), w)
    diff_ambientale = np.abs(g1 - g0).max()
    diff_tangenziale = np.abs(S @ g1 - S @ g0).max()
    registra('T1b', diff_ambientale > 1e-3 and diff_tangenziale < 1e-8,
             'Teorema 3.1(b)', '1e-8', '%.2e' % diff_tangenziale,
             'grad ambientale varia di %.4f, la sua proiezione S no' % diff_ambientale)

    # ── T1c Teorema 3.1(b), direzione inversa ───────────────────────────────────────────────
    # se le colonne della curvatura mista NON stanno in span{1}, la proiezione dipende da b
    gA0 = grad_num(lambda x: L_attivo(x, b0), w)
    gA1 = grad_num(lambda x: L_attivo(x, b1), w)
    var_tang = np.abs(S @ gA1 - S @ gA0).max()
    registra('T1c', var_tang > 1e-6, 'Teorema 3.1(b) inverso', '>1e-6', '%.2e' % var_tang,
             'obiettivo attivo: la proiezione tangenziale dipende da b')

    # ── T1d il nucleo di S e' la direzione uniforme SOLO sul simplesso (nuovo) ──────────────
    # Serve a giustificare perche' le ipotesi del Teorema 3.1(b) sono imposte sul simplesso:
    # S(w) 1 = w (1 - 1'w), quindi fuori dal simplesso il nucleo e' banale e per somma > 1 la
    # matrice ha perfino un autovalore negativo.
    esiti_1d = []
    for s_tot in (1.0, 0.8, 1.3):
        ws = w / w.sum() * s_tot
        Ss = matrice_S(ws)
        atteso = ws * (1.0 - ws.sum())
        esiti_1d.append((s_tot, np.abs(Ss @ np.ones(N) - atteso).max(),
                         np.linalg.matrix_rank(Ss, tol=1e-12),
                         np.linalg.eigvalsh(Ss).min()))
    err1d = max(e[1] for e in esiti_1d)
    ok1d = (err1d < 1e-14
            and esiti_1d[0][2] == N - 1 and esiti_1d[1][2] == N and esiti_1d[2][2] == N
            and esiti_1d[2][3] < -1e-6)
    registra('T1d', ok1d, 'Lemma 3.4, normalizzazione', '1e-14', '%.1e' % err1d,
             'S1 = w(1-1\'w) esatta; rango %d sul simplesso, %d fuori; autoval min %.3f a somma 1,3'
             % (esiti_1d[0][2], esiti_1d[1][2], esiti_1d[2][3]))

    # ── T12 coerenza fra derivata ambientale e vincolo di simplesso (nuovo) ─────────────────
    rng = np.random.default_rng(SEME + 1)
    d = rng.normal(size=N)
    d = d - d.mean()                       # direzione tangente al simplesso: 1'd = 0
    h = 1e-7
    dir_num = (L_attivo(w + h * d, b0) - L_attivo(w - h * d, b0)) / (2 * h)
    err12 = abs(dir_num - d @ gA0)
    P = np.eye(N) - np.ones((N, N)) / N
    err12b = abs(dir_num - d @ (P @ gA0))
    registra('T12', err12 < 1e-6 and err12b < 1e-6, 'convenzione tangenziale', '1e-6',
             '%.2e' % max(err12, err12b),
             'derivata dentro il simplesso = componente tangenziale del gradiente ambientale')

    # ── T2 Teorema 3.2(2): L-statistica lineare ─────────────────────────────────────────────
    c = np.array([0.0, 1.0, 0.0, 0.0])         # secondo piu' piccolo: un quantile

    def L_stat(x, b, pesi):
        return float(pesi @ np.sort(u_di_w(x, A, B) - b))

    ordine = np.argsort(u - b0)
    atteso = J[ordine[1], :]
    g_num = grad_num(lambda x: L_stat(x, b0, c), w)
    e2a = np.abs(g_num - atteso).max()
    registra('T2a', e2a < 1e-5, 'Teorema 3.2(2)', '1e-5', '%.2e' % e2a,
             'gradiente = riga dello Jacobiano del rango selezionato')

    b_cella = b0 + np.array([0.0005, -0.0004, 0.0003, 0.0002])
    assert (np.argsort(u_di_w(w, A, B) - b_cella) == ordine).all(), 'perturbazione fuori cella'
    e2b = np.abs(grad_num(lambda x: L_stat(x, b_cella, c), w) - g_num).max()
    registra('T2b', e2b < 1e-5, 'Teorema 3.2(2) invarianza', '1e-5', '%.2e' % e2b,
             'gradiente invariato dentro la cella dell arrangiamento')

    # ── T3/T4 Teorema 3.2(3): formula del salto ─────────────────────────────────────────────
    i_blk, j_blk = ordine[1], ordine[2]
    b_wall = b0.copy()
    b_wall[j_blk] = u[j_blk] - (u[i_blk] - b0[i_blk])          # a_i = a_j: sulla parete
    eps = 1e-5
    b_pre, b_post = b_wall.copy(), b_wall.copy()
    b_pre[j_blk] += eps
    b_post[j_blk] -= eps
    salto = (grad_num(lambda x: L_stat(x, b_post, c), w)
             - grad_num(lambda x: L_stat(x, b_pre, c), w))
    salto_atteso = (c[1] - c[2]) * (J[i_blk, :] - J[j_blk, :])
    e3 = np.abs(salto - salto_atteso).max()
    registra('T3', e3 < 1e-4, 'Teorema 3.2(3) salto', '1e-4', '%.2e' % e3,
             'salto osservato = (c_m - c_m\') (J_j - J_i), da entrambi i lati')

    c_med = np.array([0.0, 0.5, 0.5, 0.0])
    e4 = np.abs(grad_num(lambda x: L_stat(x, b_post, c_med), w)
                - grad_num(lambda x: L_stat(x, b_pre, c_med), w)).max()
    registra('T4', e4 < 1e-4, 'Teorema 3.2(3) salto nullo', '1e-4', '%.2e' % e4,
             'ranghi con peso uguale (mediana): nessun salto')

    # ── T11 Teorema 3.2(4): rango, degenerazione puntuale, inerzia sulla cella ──────────────
    # Tre sotto-controlli distinti, perche' il paper distingue tre affermazioni diverse.
    N2, K2 = 3, 4

    # (a) SUFFICIENTE: con rango pieno K e Lambda non affine l'obiettivo e' attivo.
    r = np.linalg.matrix_rank(J, tol=1e-10)

    def L_quad(x, b):
        return float(((np.sort(u_di_w(x, A, B) - b)) ** 2).sum())

    cm_quad = max(abs(curvatura_mista(L_quad, w, b0, l, i))
                  for l in range(K) for i in range(N))
    ok11a = (r == K) and cm_quad > 1e-3

    # (b) PUNTUALE: con K > N il nucleo di J(w)' e' non banale in OGNI punto, ma il vettore che
    #     lo genera dipende da w. Un obiettivo costruito su un c fisso preso nel nucleo in w0 e'
    #     degenere SOLO in w0: allontanandosi nella stessa cella il gradiente torna non nullo.
    #     Questo sotto-controllo verifica entrambe le cose, ed e' dichiarato puntuale.
    A2 = rendimenti(n_asset=N2, seme=SEME + 7)
    w2 = softmax(np.linspace(-0.3, 0.4, N2))
    J2 = jacobiano(w2, A2, B)                      # 4 x 3: righe necessariamente dipendenti
    _, _, Vt = np.linalg.svd(J2.T)
    c_ker = Vt[-1]
    residuo_ker = np.abs(J2.T @ c_ker).max()

    def L_punt(x, b):
        return float((c_ker @ (u_di_w(x, A2, B) - b)) ** 2)

    g_in_w0 = np.abs(grad_num(lambda x: L_punt(x, b0), w2)).max()
    ord2 = np.argsort(u_di_w(w2, A2, B) - b0)
    rng11 = np.random.default_rng(SEME + 13)
    w2_vicino = softmax(np.linspace(-0.3, 0.4, N2) + rng11.normal(0, 0.05, N2))
    stessa_cella_b = (np.argsort(u_di_w(w2_vicino, A2, B) - b0) == ord2).all()
    g_vicino = np.abs(grad_num(lambda x: L_punt(x, b0), w2_vicino)).max()
    ok11b = (residuo_ker < 1e-10 and g_in_w0 < 1e-8
             and stessa_cella_b and g_vicino > 1e-7)   # NON inerte fuori da w0

    # (c) SULLA CELLA: costruzione strutturata. Due sotto-periodi con sequenze di rendimento
    #     identiche danno u_1 = u_2 e J_1 = J_2 per OGNI w, quindi a_1 - a_2 = b_2 - b_1 non
    #     dipende da w. Con Lambda = (a_(r) - a_(s))^2 sui due ranghi occupati da quei blocchi
    #     l'obiettivo e' costante sulla cella: gradiente e curvatura mista nulli ovunque in essa.
    A3 = rendimenti(n_asset=N2, seme=SEME + 11).copy()
    A3[B[1], :] = A3[B[0], :]                      # blocco 2 identico al blocco 1, giorno per giorno
    w3 = softmax(np.linspace(-0.3, 0.4, N2))
    u3 = u_di_w(w3, A3, B)
    b3 = u3 - np.array([0.030, -0.020, 0.055, -0.010])   # b_1 != b_2: ranghi distinti
    ord3 = np.argsort(u3 - b3)
    rango_r = int(np.where(ord3 == 0)[0][0])       # rango occupato dal blocco 1
    rango_s = int(np.where(ord3 == 1)[0][0])       # rango occupato dal blocco 2

    def L_cella(x, b):
        a_ord = np.sort(u_di_w(x, A3, B) - b)
        return float((a_ord[rango_r] - a_ord[rango_s]) ** 2)

    valore_atteso = (b3[1] - b3[0]) ** 2
    punti = [w3,
             softmax(np.array([0.2, -0.5, 0.9])),
             softmax(np.array([-1.0, 0.3, 0.1])),
             softmax(np.array([0.05, 0.05, 0.05]))]
    err_val, g_max, cm_max, tutte_in_cella = 0.0, 0.0, 0.0, True
    for x in punti:
        tutte_in_cella &= bool((np.argsort(u_di_w(x, A3, B) - b3) == ord3).all())
        err_val = max(err_val, abs(L_cella(x, b3) - valore_atteso))
        g_max = max(g_max, np.abs(grad_num(lambda y: L_cella(y, b3), x)).max())
        cm_max = max(cm_max, max(abs(curvatura_mista(L_cella, x, b3, l, i))
                                 for l in range(K2) for i in range(N2)))
    rango_c = np.linalg.matrix_rank(jacobiano(w3, A3, B), tol=1e-10)
    ok11c = (tutte_in_cella and rango_c < K2 and err_val < 1e-12
             and g_max < 1e-10 and cm_max < 1e-10)

    registra('T11', ok11a and ok11b and ok11c, 'Teorema 3.2(4) rango', '1e-10',
             '%.2e' % max(g_max, cm_max),
             'rango pieno -> attivo (curv %.3f); c fisso nel nucleo: degenere in w0 (%.1e) ma '
             'NON nella cella (%.1e); costruzione a blocchi ripetuti: inerte in %d punti della '
             'stessa cella, rango %d < K' % (cm_quad, g_in_w0, g_vicino, len(punti), rango_c))

    # ── T5/T6 Teorema 3.3(1) e (3) ──────────────────────────────────────────────────────────
    arg = (b0 - u) / TAU
    grad_an = -(J * sigma(arg)[:, None]).sum(axis=0) / (K * TAU)
    e5 = np.abs(grad_an - grad_num(lambda x: L_attivo(x, b0), w)).max()
    registra('T5', e5 < 1e-5, 'Teorema 3.3(1)', '1e-5', '%.2e' % e5,
             'gradiente dell obiettivo lisciato, analitico contro numerico')

    e6 = 0.0
    for l in range(K):
        gpp = sigma(arg[l]) * (1 - sigma(arg[l]))
        for i in range(N):
            e6 = max(e6, abs(-gpp * J[l, i] / (K * TAU ** 2)
                             - curvatura_mista(L_attivo, w, b0, l, i)))
    registra('T6', e6 < 5e-3, 'Teorema 3.3(3) modulatore', '5e-3', '%.2e' % e6,
             'forma a modulatore, verificata su tutti i K x N incroci')

    # ── T7 Teorema 3.3(4): variazione totale ────────────────────────────────────────────────
    l, i = 2, 3
    def tv_di_tau(tau):
        griglia = np.linspace(-60 * tau, 60 * tau, 40001)      # a_l centrato in zero
        val = np.abs(sigma(-griglia / tau) * (1 - sigma(-griglia / tau))
                     * J[l, i] / (K * tau ** 2))
        return float(np.trapezoid(val, griglia)) if hasattr(np, 'trapezoid') \
            else float(np.trapz(val, griglia))

    tv = tv_di_tau(TAU)
    tv_att = abs(J[l, i]) / (K * TAU)
    e7 = abs(tv - tv_att) / tv_att
    registra('T7', e7 < 1e-3, 'Teorema 3.3(4) variazione', '1e-3 rel', '%.2e' % e7,
             'integrale %.5f contro |J|/(K tau) = %.5f' % (tv, tv_att))

    # ── T7b legge di scala in tau, due ordini di grandezza (nuovo) ──────────────────────────
    taus = np.array([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    sup_oss = np.array([abs(J[l, i]) / (4 * K * t ** 2) for t in taus])   # a_l = 0
    sup_num = []
    for t in taus:
        griglia = np.linspace(-20 * t, 20 * t, 8001)
        sup_num.append(np.abs(sigma(-griglia / t) * (1 - sigma(-griglia / t))
                              * J[l, i] / (K * t ** 2)).max())
    e7b = np.abs(np.array(sup_num) / sup_oss - 1).max()
    pend_sup = np.polyfit(np.log(taus), np.log(np.array(sup_num)), 1)[0]
    pend_tv = np.polyfit(np.log(taus), np.log([tv_di_tau(t) for t in taus]), 1)[0]
    # a b FISSATO con eccesso non nullo la sensibilita' va a ZERO, non all'infinito
    a_fisso = 0.05
    punt = np.array([sigma(-a_fisso / t) * (1 - sigma(-a_fisso / t))
                     * abs(J[l, i]) / (K * t ** 2) for t in taus])
    decresce = punt[0] < punt[-1] * 1e-6
    ok7b = e7b < 1e-3 and abs(pend_sup + 2) < 0.02 and abs(pend_tv + 1) < 0.02 and decresce
    registra('T7b', ok7b, 'Teorema 3.3(4) scaling', '0.02 pend.', '%.3f' % e7b,
             'pendenze log-log: sup %.3f (attesa -2), variazione %.3f (attesa -1); '
             'a b fisso la sensibilita va a zero' % (pend_sup, pend_tv))

    # ── T7c la legge di scala dipende dalla NORMALIZZAZIONE (nuovo) ─────────────────────────
    # Obiettivo generale L = (kappa/K) sum g(a_k/tau). Due convenzioni canoniche:
    #   kappa = 1     penalita' di forma fissa: sup ~ tau^-2, variazione totale ~ tau^-1
    #   kappa = tau   lisciatura alla Nesterov, converge alla cerniera: sup ~ tau^-1,
    #                 variazione totale INDIPENDENTE da tau
    # Invariante rispetto alla normalizzazione: il rapporto sup/variazione vale 1/(4 tau).
    def sup_e_tv(tau, kappa):
        griglia = np.linspace(-60 * tau, 60 * tau, 40001)
        val = np.abs(kappa * sigma(-griglia / tau) * (1 - sigma(-griglia / tau))
                     * J[l, i] / (K * tau ** 2))
        tv = float(np.trapezoid(val, griglia)) if hasattr(np, 'trapezoid') \
            else float(np.trapz(val, griglia))
        return val.max(), tv

    sup1 = np.array([sup_e_tv(t, 1.0)[0] for t in taus])
    tv1 = np.array([sup_e_tv(t, 1.0)[1] for t in taus])
    supT = np.array([sup_e_tv(t, t)[0] for t in taus])
    tvT = np.array([sup_e_tv(t, t)[1] for t in taus])
    p_sup1 = np.polyfit(np.log(taus), np.log(sup1), 1)[0]
    p_tv1 = np.polyfit(np.log(taus), np.log(tv1), 1)[0]
    p_supT = np.polyfit(np.log(taus), np.log(supT), 1)[0]
    p_tvT = np.polyfit(np.log(taus), np.log(tvT), 1)[0]
    tv_costante = np.abs(tvT / tvT[0] - 1).max()
    tv_atteso = abs(J[l, i]) / K
    rapporto = np.abs((sup1 / tv1) * (4 * taus) - 1).max()
    ok7c = (abs(p_sup1 + 2) < 0.02 and abs(p_tv1 + 1) < 0.02
            and abs(p_supT + 1) < 0.02 and abs(p_tvT) < 0.02
            and tv_costante < 1e-3 and abs(tvT[0] - tv_atteso) / tv_atteso < 1e-3
            and rapporto < 1e-3)
    registra('T7c', ok7c, 'Teorema 3.3(4) normalizzazione', '0.02 pend.', '%.3f' % rapporto,
             'kappa=1: sup %.3f, tv %.3f; kappa=tau: sup %.3f, tv %.3f (costante, = |J|/K); '
             'rapporto sup/tv = 1/(4 tau) invariante'
             % (p_sup1, p_tv1, p_supT, p_tvT))

    # ── T8 Lemma 3.4 ─────────────────────────────────────────────────────────────────
    v = np.linspace(-1, 2, N)
    sim = np.abs(S - S.T).max()
    nucleo = np.abs(S @ np.ones(N)).max()
    rango = np.linalg.matrix_rank(S, tol=1e-12)
    quad = float(v @ S @ v)
    varw = float(w @ (v ** 2) - (w @ v) ** 2)
    autoval = np.linalg.eigvalsh(S)
    ok8 = (sim < 1e-15 and nucleo < 1e-15 and rango == N - 1
           and abs(quad - varw) < 1e-12 and autoval.min() > -1e-15)
    registra('T8', ok8, 'Lemma 3.4', '1e-12', '%.1e' % abs(quad - varw),
             'simmetrica, semidefinita positiva, nucleo = span{1}, rango %d, forma = Var_w'
             % rango)

    # ── T9 Corollary 3.5: se e solo se, e limitazione bilaterale ───────────────────────────
    J_pes = (J * sigma(arg)[:, None]).sum(axis=0)
    ok9 = (np.abs(S @ (np.ones(N) * 3.7)).max() < 1e-15
           and np.abs(S @ J_pes).max() > 1e-6)
    registra('T9', ok9, 'Corollary 3.5(1)', '1e-15', '%.1e' % np.abs(S @ np.ones(N)).max(),
             'input uniforme annullato; Jacobiano pesato reale non annullato')

    def limitazione(x):
        mu = float(w @ x)
        varx = float(w @ (x - mu) ** 2)
        n_sx = float(np.sqrt(w.min() * varx))
        n_dx = float(np.sqrt(w.max() * varx))
        return np.linalg.norm(S @ x), n_sx, n_dx, varx

    prove = [J_pes, np.linspace(-2, 3, N), np.ones(N) * 5.0, np.eye(N)[0]]
    ok9b = all(sx - 1e-12 <= n <= dx + 1e-12 for n, sx, dx, _ in map(limitazione, prove))
    registra('T9b', ok9b, 'Corollary 3.5(2) limitazione', '1e-12', '0',
             'sqrt(w_min Var_w) <= ||S x|| <= sqrt(w_max Var_w) su 4 vettori di prova')

    # ── T9c caso limite: numeratore e denominatore entrambi piccoli (nuovo) ─────────────────
    base = np.ones(N) + 0.3 * np.array([1, -1, 0.5, -0.5, 0.2, -0.7])
    scale = np.array([1.0, 1e-3, 1e-6, 1e-9])
    rho = np.array([np.linalg.norm(S @ (s * base)) / np.linalg.norm(s * base) for s in scale])
    delta = np.sqrt(float(w @ (base - w @ base) ** 2)) / np.linalg.norm(base)
    costante = np.abs(rho / rho[0] - 1).max() < 1e-9
    incorniciato = np.sqrt(w.min()) * delta - 1e-12 <= rho[0] <= np.sqrt(w.max()) * delta + 1e-12
    registra('T9c', costante and incorniciato, 'Corollary 3.5(3)', '1e-9',
             '%.1e' % np.abs(rho / rho[0] - 1).max(),
             'rho invariante di scala su 9 ordini: NON tende a zero se solo la scala svanisce')

    # ── T10 Corollary 3.5(4): asset identici ───────────────────────────────────────────────
    A_id = rendimenti(identici=True)
    J_id = jacobiano(w, A_id, B)
    arg_id = (b0 - u_di_w(w, A_id, B)) / TAU
    gz_id = S @ (-(J_id * sigma(arg_id)[:, None]).sum(axis=0) / (K * TAU))
    disp = np.abs(J_id - J_id[:, [0]]).max()
    registra('T10', disp < 1e-12 and np.abs(gz_id).max() < 1e-12, 'Corollary 3.5(4)', '1e-12',
             '%.1e' % np.abs(gz_id).max(),
             'percorsi identici -> Jacobiano uniforme -> gradiente nei punteggi nullo')

    # ── esito ───────────────────────────────────────────────────────────────────────────────
    falliti = [n for n, ok, *_ in esiti if not ok]
    print()
    print('%d test su %d superati' % (len(esiti) - len(falliti), len(esiti)))
    if falliti:
        print('FALLITI:', ', '.join(falliti))
        raise SystemExit(1)
    print('tutti gli enunciati corretti sono confermati (semi %d/%d, N=%d, T=%d, K=%d, tau=%.3f)'
          % (SEME, SEME + 7, N, T, K, TAU))


if __name__ == '__main__':
    main()
