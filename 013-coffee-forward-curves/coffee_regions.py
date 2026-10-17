#-------- coffee_regions.py --------------------------------------------------------
#-----------------------------------------------------------------------------
# Brazil's arabica belt as a handful of weighted points, one per producing region.
#
# Weights are rough shares of Brazil's arabica output, renormalised over the regions
# listed. TO CHECK against CONAB's harvest survey (Acompanhamento da Safra Brasileira
# de Café) before any result is quoted. Frost and drought indices are
# weighted means over these points, so a frost in the Sul de Minas counts for more
# than one in Paraná.

# name: (latitude, longitude, weight)
REGIONS = {
    "Sul de Minas (Varginha)":         (-21.55, -45.43, 0.30),
    "Sul de Minas (Guaxupe)":          (-21.30, -46.71, 0.10),
    "Cerrado Mineiro (Patrocinio)":    (-18.94, -46.99, 0.15),
    "Matas de Minas (Manhuacu)":       (-20.26, -42.03, 0.17),
    "Mogiana (Franca)":                (-20.54, -47.40, 0.08),
    "Espirito Santo highlands":        (-20.34, -41.13, 0.10),
    "Bahia plateau (Vitoria da Conquista)": (-14.86, -40.84, 0.05),
    "Parana (Londrina)":               (-23.31, -51.16, 0.05),
}

_total = sum(w for _, _, w in REGIONS.values())
REGIONS = {k: (la, lo, w / _total) for k, (la, lo, w) in REGIONS.items()}
