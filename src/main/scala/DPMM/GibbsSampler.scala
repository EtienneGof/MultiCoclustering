package DPMM

import Common.NormalInverseWishart
import Common.ProbabilisticTools._
import breeze.linalg.DenseVector
import breeze.stats.distributions.{Gamma, MultivariateGaussian}

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer

class GibbsSampler(val Data: Array[DenseVector[Double]],
                   var prior: NormalInverseWishart = new NormalInverseWishart(),
                   var alphaPrior: Gamma,
                   var initByUserPartition: Option[Array[Int]] = None
                  ) extends Default(Data, prior, alphaPrior, initByUserPartition) {

  var components: ArrayBuffer[MultivariateGaussian] = prior.posteriorSample(Data, partition).to[ArrayBuffer]

  override def posteriorPredictive(observation: DenseVector[Double], cluster: Int): Double = {
    components(cluster).logPdf(observation)
  }

  private def removeElementFromComponents(idx: Int): Unit = {
    val currentPartition = partition(idx)
    if (countCluster(currentPartition) == 1) {
      components.remove(currentPartition)
    }
  }

  private def addElementToComponents(idx: Int): Unit = {
    if(partition(idx) == components.length){
      val newDensity = prior.update(Array(Data(idx))).sample()
      components = components ++ ArrayBuffer(newDensity)
    }
  }

  override def updatePartition(verbose: Boolean = false): Unit = {
    for (i <- 0 until n) {
      removeElementFromComponents(i)
      removeElementFromCluster(i)
      drawMembership(i)
      addElementToComponents(i)
      addElementToCluster(i)
    }
  }

  def run(maxIter: Int = 10,
          verbose: Boolean = false): (Array[Array[Int]], Array[Array[MultivariateGaussian]], Array[Double]) = {

    var partitionEveryIteration = Array(partition)
    var componentEveryIteration = Array(components.toArray)
    var likelihoodEveryIteration = Array(prior.DPMMLikelihood(actualAlpha, Data, partition, countCluster.toArray, components.toArray))

    @tailrec def go(iter: Int): Unit = {

      if(verbose){
        println("\n>>>>>> Iteration: " + iter.toString)
        println("\u03B1 = " + actualAlpha.toString)
        println("Cluster sizes: "+ countCluster.mkString(" "))
      }

      if (iter > maxIter) {

      } else {
        updatePartition()
        components = prior.posteriorSample(Data, partition).to[ArrayBuffer]

        if(updateAlphaFlag){actualAlpha = updateAlpha(actualAlpha, alphaPrior, countCluster.length, n)}

        val likelihood = prior.DPMMLikelihood(actualAlpha, Data, partition, countCluster.toArray, components.toArray)

        partitionEveryIteration = partitionEveryIteration :+ partition
        componentEveryIteration = componentEveryIteration :+ components.toArray
        likelihoodEveryIteration = likelihoodEveryIteration :+ likelihood
        go(iter + 1)
      }
    }

    go(1)

    (partitionEveryIteration, componentEveryIteration, likelihoodEveryIteration)
  }
}
