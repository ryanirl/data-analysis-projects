from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import warnings

from pysam import VariantFile

warnings.filterwarnings("ignore")


def allele_indices(vcf_dir, skip_every = 5):
    """
    Given a VCF file, this returns each samples genotype 
    represented as allele indices along with the corresponding
    variant IDs and list of samples.

    Args:
        vcf_dir (string): Location of VCF file.
        skip_every (int): Only return every other "skip_every"
            variant. This may be useful if you are running this
            with less than 16GB of RAM.

    Returns:
        list: List of each sample.
        list: Genotypes of shape: (samples, num_variants // skip_every, 2).
        list: Variant IDs for each variant.

    """
    vcf_file = VariantFile(vcf_dir)

    samples = list(vcf_file.header.samples)

    genotypes = []
    variant_ids = []
    for counter, record in tqdm(enumerate(vcf_file.fetch())):
        if counter % skip_every == 0:
            alleles = [record.samples[x].allele_indices for x in samples]

            genotypes.append(alleles)
            variant_ids.append(record.id)

    return samples, genotypes, variant_ids

def parse_panel(panel_dir):
    """
    Returns python dictionaries that maps each sample
    to a population and superpopulation. 

    Args:
        panel_dir (string): Location of the panel file 
            containing information about each sample.

    Returns:
        dict: Population map for each sample. For
            example, {sample_id, population}.

        dict: Superpopulation map for each sample. For
            example, {sample_id, superpopulation}.

    """
    population_map = {}  
    superpopulation_map = {}  
    with open(args.panel_dir) as panel_file:
        for sample in panel_file:
            """
            EX: ["HG00607", "CHS", "ASN", "ILLUMINA"]

            sample[0] = Sample ID.
            sample[1] = Population.
            sample[2] = Superpopulation.

            """
            sample = sample.strip().split('\t')

            population_map[sample[0]] = sample[1]
            superpopulation_map[sample[0]] = sample[2]
    
    return population_map, superpopulation_map 


def main(args):
    """
    Given a VCF file, this will put each samples allele indices 
    for each veriant at some stride args.skip_every. 
    Then place it in a "./clean_data.csv" file.

    Args:
        --vcf_dir (string): Location of the VCF file.
        --panel_dir (string): Location of the Panel file containing
            sample information.
        --skip_every (int): Recommend using if you are limited in RAM. 
            Samples every other N variants therefore cutting down on 
            memory usage. 

    Returns:
        None

    """
    samples, genotypes, variant_ids = allele_indices(args.vcf_dir, args.skip_every)
    population_map, superpopulation_map = parse_panel(args.panel_dir)

    # Prepare the data for dimensionality reduction.
    # shape: (samples, variant_count // skip_every, 2)
    genotypes = np.array(genotypes)
    genotypes = np.sum(genotypes, axis = 2).T

    # Create the cleaned data csv.
    df = pd.DataFrame(genotypes, columns = variant_ids, index = samples)
    df = df.reset_index().rename(columns = {"index": "sample"})

    df["population_code"]      = df["sample"].map(population_map)
    df["superpopulation_code"] = df["sample"].map(superpopulation_map) 

    df.to_csv("./clean_data.csv", index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--vcf_dir", type = str)
    parser.add_argument("--panel_dir", type = str)
    parser.add_argument("--skip_every", type = int, default = 5)

    args = parser.parse_args()
    
    main(args)









