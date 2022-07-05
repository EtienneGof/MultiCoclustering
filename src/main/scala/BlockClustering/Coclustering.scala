package BlockClustering

import Common.NormalInverseWishart
import Common.Tools._
import breeze.linalg.DenseVector
import breeze.stats.distributions.{Gamma, MultivariateGaussian}

import scala.annotation.tailrec

/**
  * Implements the inference of a bi-dimensional Dirichlet Process Mixture Model, similar to a Bayesian Non Parametric Latent Block
  * Model. The inference, launched by the 'run' method, is composed of a Gibbs sampler that alternates the updates of the
  * row-partition conditional to the column-partition, and then the column-partition conditional to the row-partition
  * @param DataByRow Dataset as list of list, with the first list indexing the rows, and the second the columns
  * @param alphaRow (optional) Concentration parameter. Must be assigned a value if alphaRowPrior has none.
  * @param alphaRowPrior (optional) Observation partition concentration parameter prior. Must be assigned a value if alphaRow has none.
  * @param alphaCol (optional) Concentration parameter. Must be assigned a value if alphaColPrior has none.
  * @param alphaColPrior (optional) Variable partition concentration parameter prior. Must be assigned a value if alphaCol has none.
  * @param initByUserPrior (optional) Prior distribution (Normal Inverse Wishart) on the mixture components parameters.
  * @param initByUserRowPartition (optional) Initial observation partition that acts as a starting point for the inference.
  * @param initByUserColPartition (optional) Initial variable partition that acts as a starting point for the inference.
  */
class Coclustering(val DataByRow: List[List[DenseVector[Double]]],
                   var alphaRow: Option[Double] = None,
                   var alphaRowPrior: Option[Gamma] = None,
                   var alphaCol: Option[Double] = None,
                   var alphaColPrior: Option[Gamma] = None,
                   var initByUserPrior: Option[NormalInverseWishart] = None,
                   var initByUserRowPartition: Option[List[Int]] = None,
                   var initByUserColPartition: Option[List[Int]] = None) {

  val prior: NormalInverseWishart = initByUserPrior match {
    case Some(pr) => pr
    case None => new NormalInverseWishart(DataByRow)
  }

  var rowClustering = new Clustering(DataByRow, alphaRow, alphaRowPrior, Some(prior), initByUserRowPartition, initByUserColPartition)
  var colClustering = new Clustering(DataByRow.transpose, alphaCol, alphaColPrior,
    Some(prior),
    Some(rowClustering.colPartition),
    Some(rowClustering.rowPartition),
    Some(rowClustering.NIWParamsByRow.transpose))

  /** Computes the model completed likelihood.
    *
    */
  def likelihood(): Double = {
    val rowLikelihood = rowClustering.likelihood()

    rowLikelihood + colClustering.probabilityPartition
  }

  /** Returns the expectation of the block component parameters.
    *
    */
  def componentsEstimation: List[List[MultivariateGaussian]] = {
    rowClustering.componentsEstimation
  }

  /** launches the inference process
    *
    * @param nIter Iteration number
    * @param verbose Boolean activating the output of additional information (cluster count evolution)
    * @return
    */
  def run(nIter: Int = 10,
          verbose: Boolean = false): (List[List[Int]], List[List[Int]], List[List[List[MultivariateGaussian]]], List[Double]) = {

    var likelihoodEveryIteration = List(likelihood())

    @tailrec def go(iter: Int): Unit = {

      if(verbose){
        println("\n>>>>>> Coclustering Iteration: " + iter.toString)
        Common.Tools.prettyPrintLBM(rowClustering.countRowCluster.toList, rowClustering.countColCluster.toList)
        Common.Tools.prettyPrintLBM(colClustering.countRowCluster.toList, colClustering.countColCluster.toList)
      }

      if (iter <= nIter) {

        var t0 = System.nanoTime()

        rowClustering.setColPartitionAndNIWParams(colClustering.rowPartition, colClustering.NIWParamsByRow.transpose)
        rowClustering.updatePartition()

        if(verbose){
          t0 = printTime(t0, "draw row Partition")
        }

        colClustering.setColPartitionAndNIWParams(rowClustering.rowPartition, rowClustering.NIWParamsByRow.transpose)
        colClustering.updatePartition()

        if(verbose){
          t0 = printTime(t0, "draw col Partition")
        }

        likelihoodEveryIteration =  likelihoodEveryIteration ++
          List(likelihood())

        go(iter + 1)
      }
    }

    go(1)

    (rowClustering.rowPartitionEveryIteration,
      colClustering.rowPartitionEveryIteration,
      rowClustering.componentsEveryIterations,
      likelihoodEveryIteration)

  }

}
