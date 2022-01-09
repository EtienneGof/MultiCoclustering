# MultiCoclustering

This code implements three model based block clustering methods. These models are based on the Dirichlet Process Mixture Model (DPMM, used for univariate dataset clustering) and extends it to multivariate datasets.

### What's inside 

The first approach, named "Clustering", performs the inference of a Dirichet Process Mixture Model on a univariate or multivariate dataset. Optionnally, it enables the user to specify a partition of the variables, thus allowing the creation of "blocks", i.e., groups of variables and simulations that independently follow the same distribution. In the "Clustering" case, this partition is not updated.

The second approach, named "Coclustering", estimates both the row-partition (as in the Clustering), but also estimates the variable partition. In this case, the dataset elements that belong to the same row and column cluster (i.e., the same block) follow independently the same distribution.

The last approach, named "Multi-Coclustering", assumes the presence of several coclustering structures. This approach first estimates a partition of variable and, in each dataset corresponding to one variable cluster, estimates a coclustering structure, such that elements in the same coclustering structure and, inside of it, in the same block, follow the same distribution independently. 

The three methods and their inter-connection is illustrated below

<p align="center">
  <img src="https://github.com/EtienneGof/MultiCoclustering/blob/main/illustration.gif" />
</p>


Fo each methods, the inference is performed with a Collapsed Gibbs Sampler (cf. [1]).

Also, in each case, the user can either choose to provide the Dirichlet Process concentration parameter value or a Gamma hyperprior for this parameter. In the latter case, alpha's value is updated following [2]. Using an hyper-prior is mandatory in the multi-coclustering case.

### Quick Setup

The script build.sh is provided to build the scala sources. 

See src/pom.xml file for Scala dependencies.

The scripts clusteringExample.scala, coclusteringExample.scala and multiCoclusteringExample.scala illustrate the use of each methods, based on toy datasets.

[1] Neal, R. M. (2000). Markov chain sampling methods for Dirichlet process mixture models. Journal of computational and graphical statistics, 9(2), 249-265.

[2] West, M. (1992). Hyperparameter estimation in Dirichlet process mixture models. ISDS Discussion Paper# 92-A03: Duke University.
