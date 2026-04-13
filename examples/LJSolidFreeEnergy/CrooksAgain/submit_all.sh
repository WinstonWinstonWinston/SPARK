#!/bin/bash
# Submit all 6 permutations (3 alphas x 2 schedules) as independent jobs

sbatch --job-name=crooks_a150_cos --export=ALPHA=150,SCHED=cosine,STEPS_SWITCH=2000,STEPS_EQ=2000 run_crooks.sbatch
sbatch --job-name=crooks_a200_cos --export=ALPHA=200,SCHED=cosine,STEPS_SWITCH=2000,STEPS_EQ=2000 run_crooks.sbatch
sbatch --job-name=crooks_a250_cos --export=ALPHA=250,SCHED=cosine,STEPS_SWITCH=2000,STEPS_EQ=2000 run_crooks.sbatch
sbatch --job-name=crooks_a150_lin --export=ALPHA=150,SCHED=linear,STEPS_SWITCH=3000,STEPS_EQ=2000 run_crooks.sbatch
sbatch --job-name=crooks_a200_lin --export=ALPHA=200,SCHED=linear,STEPS_SWITCH=3000,STEPS_EQ=2000 run_crooks.sbatch
sbatch --job-name=crooks_a250_lin --export=ALPHA=250,SCHED=linear,STEPS_SWITCH=3000,STEPS_EQ=2000 run_crooks.sbatch
