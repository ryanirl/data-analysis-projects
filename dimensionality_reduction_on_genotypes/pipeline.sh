python3 parse_vcf.py \
    --vcf_dir ./data/ALL.chr21.phase1_release_v3.20101123.snps_indels_svs.genotypes.vcf.gz \
    --panel_dir ./data/phase1_integrated_calls.20101123.ALL.panel \
    --skip_every 25

python3 fit_model.py \
    --data_dir ./clean_data.csv \
    --PCA True \
    --tSNE True \
    --UMAP True 

python3 inference.py \
    --vis_dir ./vis_data.csv

