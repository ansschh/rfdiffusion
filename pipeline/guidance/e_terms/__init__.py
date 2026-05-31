# Decomposed catalytic energy terms. Each term takes (protein_atoms, ACatFields)
# and returns a scalar energy. Sign convention: higher E_term = worse for the
# corresponding catalytic fact. E_cat = sum_t lambda_t * E_t with lambda_t set
# by the user/PI per term after calibration.
