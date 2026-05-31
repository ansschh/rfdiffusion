# OrganoEnzymeGen Level-2 catalytic guidance: A_cat as fields + decomposed E_cat
# as catalytic likelihood, ready to be used as `exp(-lambda * E_cat)` posterior
# weighting in an SMC / resampling scheme around RFD2.
#
# Stage 0 (this module): field representation + per-term scorers + damaged controls
# + discriminativity test. NO RFD2 modification yet — all evaluation on completed
# protein conformations.
