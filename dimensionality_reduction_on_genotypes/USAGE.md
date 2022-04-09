# Usage

Assuming you have a mac, Python3, and all of the dependencies installed, you
should be able to reproduce my resutls with the following commands: 

**Dependencies:** Altair, Pysam, UMAP (umap-learn), Pandas, NumPy, 
SKLearn, SciPy, tqdm, and Seaborn.


```
git clone https://github.com/ryanirl/data-analysis-projects

cd data-analysis-projects/dimensionality_reduction_on_genotypes

mkdir data && cd data

curl -O https://1000genomes.s3.amazonaws.com/release/20110521/ALL.chr21.phase1_release_v3.20101123.snps_indels_svs.genotypes.vcf.gz

curl -O https://1000genomes.s3.amazonaws.com/release/20110521/phase1_integrated_calls.20101123.ALL.panel

cd ..

# This should take about 2 minutes depended on 
# what kind of PC your running
sh pipeline.sh
```

---


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<br />


