"""runcfg.py -- A3: training seed switch. FULLSYS_SEED (default 42 = all earlier results) selects the seed;
any other seed writes models to fullsys_logs/models_seed<k>/ and results to results/seed<k>/ (nothing overwritten)."""
import os

SEED = int(os.environ.get('FULLSYS_SEED', '42'))
SEED_SFX = '' if SEED == 42 else f'_seed{SEED}'
RES_SUB = '' if SEED == 42 else f'seed{SEED}'
