#  :chart_with_upwards_trend: Anticipating trends in impact investing

**_Exploring predictive analytics and causal inference methods_**

## :wave: Welcome!

This was an exploratory project led by Nesta's [Discovery Hub](https://www.nesta.org.uk/project/discovery-hub/), to trial methods for anticipating future trends in venture capital investment into startups. We are especially interested in startups working on solutions relevant to Nesta’s missions, for example, green tech, food tech and early years education.

### :books: Read the blogs
We have described the results of this exploration in the following Medium articles:
- [Predicting start-up success using machine learning](https://medium.com/discovery-at-nesta/predicting-start-up-success-using-machine-learning-f3f53871bd22)
- [Causal inference for discerning the impact of research grants on company success](https://medium.com/discovery-at-nesta/causal-inference-634fe782e00c)


### :floppy_disk: Datasets
The input dataset unfortunately contains proprietary data from the Crunchbase database, which is why it can't be shared openly. However,the [Gateway to Research](https://gtr.ukri.org/) and [BEIS/Nesta R&D spatial data](https://access-research-development-spatial-data.beis.gov.uk/) are openly accessible.

You can also [see here the scripts](https://github.com/nestauk/innovation_sweet_spots/tree/dev/innovation_sweet_spots/pipeline/pilot/investment_predictions) used for joining up the three datasets, and preparing them for the prediction model and causal inference.

### :hammer: Analysis code

You will find [here the code notebooks](https://github.com/nestauk/iss_forecasting/tree/dev/iss_forecasting/analysis/company_level) used for both supervised machine learning and causal inference analysis.

We aim to implement these analyses into our future projects, whereas this repo will likely stay here as a reference. If you have questions and wish to talk with us, please feel free to reach out to [@KarlisKanders](https://twitter.com/kanderskarlis) or [@JackRasala](https://twitter.com/JackRasala).

## Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `git-crypt`, `direnv`, and `conda`
  - Have a Nesta AWS account configured with `awscli`
- Run `brew install cmake` to install CMake (needed to install LightGBM)
- Run `brew install libomp` to install OpenMP (needed to install LightGBM)
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure pre-commit
  - Configure metaflow to use AWS
- Run `make inputs-pull` to download inputs from S3
- Run `aws s3 sync s3://iss-forecasting/outputs/` to download outputs from S3
- Run `brew install graphviz` and `conda install -c conda-forge pygraphviz`
  to install graphviz to be able to visualise the causal graph when using DoWhy

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
