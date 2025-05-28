# How to install datasets

We suggest putting all datasets under the same folder (say `$DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like

```
$DATA/
|–– dtd/
|–– eurosat/
|–– fgvc-aircraft-2013b/
|–– food-101/
```

If you have some datasets already installed somewhere else, you can create symbolic links in `$DATA/dataset_name` that point to the original data to avoid duplicate download.

Datasets list:
- [DTD](#dtd)
- [EuroSAT](#eurosat)
- [FGVCAircraft](#fgvcaircraft)
- [Food101](#food101)
- [UCF101](#ucf101)

The instructions to prepare each dataset are detailed below. To ensure reproducibility and fair comparison for future work, we provide fixed train/val/test splits for all datasets except ImageNet where the validation set is used as test set. The fixed splits are either from the original datasets (if available) or created by us.

### DTD
- Download the dataset from https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz and extract it to `$DATA`. This should lead to `$DATA/dtd/`.
- Download `split_zhou_DescribableTextures.json` from this [link](https://drive.google.com/file/d/1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x/view?usp=sharing).

The directory structure should look like
```
dtd/
|–– images/
|–– imdb/
|–– labels/
|–– split_zhou_DescribableTextures.json
```

### EuroSAT
- Create a folder named `eurosat/` under `$DATA`.
- Download the dataset from http://madm.dfki.de/files/sentinel/EuroSAT.zip and extract it to `$DATA/eurosat/`.
- Download `split_zhou_EuroSAT.json` from [here](https://drive.google.com/file/d/1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o/view?usp=sharing).

The directory structure should look like
```
eurosat/
|–– 2750/
|–– split_zhou_EuroSAT.json
```

### FGVCAircraft
- Download the data from https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz.
- Extract `fgvc-aircraft-2013b.tar.gz` and keep only `data/`.
- Move `data/` to `$DATA` and rename the folder to `fgvc_aircraft/`.

The directory structure should look like
```
fgvc_aircraft/
|–– images/
|–– ... # a bunch of .txt files
```

### Food101
- Download the dataset from https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ and extract the file `food-101.tar.gz` under `$DATA`, resulting in a folder named `$DATA/food-101/`.
- Download `split_zhou_Food101.json` from [here](https://drive.google.com/file/d/1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl/view?usp=sharing).

The directory structure should look like
```
food-101/
|–– images/
|–– license_agreement.txt
|–– meta/
|–– README.txt
|–– split_zhou_Food101.json
```

### UCF101
- Create a folder named `ucf101/` under `$DATA`.
- Download the zip file `UCF-101-midframes.zip` from [here](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing) and extract it to `$DATA/ucf101/`. This zip file contains the extracted middle video frames.
- Download `split_zhou_UCF101.json` from this [link](https://drive.google.com/file/d/1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y/view?usp=sharing).

The directory structure should look like
```
ucf101/
|–– UCF-101-midframes/
|–– split_zhou_UCF101.json
```
