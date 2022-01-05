package BlockClustering

import Common.NormalInverseWishart
import Common.ProbabilisticTools._
import Common.Tools._
import breeze.linalg.DenseVector
import breeze.numerics.log
import breeze.stats.distributions.{Gamma, MultivariateGaussian}

import scala.annotation.tailrec
import scala.collection.mutable.ListBuffer

class Clustering(DataByRow: List[List[DenseVector[Double]]],
                 alpha: Option[Double] = None,
                 alphaPrior: Option[Gamma] = None,
                 initByUserPrior: Option[NormalInverseWishart] = None,
                 initByUserRowPartition: Option[List[Int]] = None,
                 initByUserColPartition: Option[List[Int]] = None,
                 initByUserNIW: Option[ListBuffer[ListBuffer[NormalInverseWishart]]] = None) extends Serializable {


  val n: Int = DataByRow.length
  val p: Int = DataByRow.head.length

  var prior: NormalInverseWishart = initByUserPrior match {
    case Some(pr) => pr
    case None => new NormalInverseWishart(DataByRow)
  }

  val d: Int = prior.d
  require(prior.d == DataByRow.head.head.length, "Prior and data dimensions differ")

  var rowPartition: List[Int] = initByUserRowPartition match {
    case Some(m) =>
      require(m.length == n, s"Inputted row partition length ${m.length} does not equal the dataset row number $n")
      m
    case None => List.fill(n)(0)
  }

  var colPartition: List[Int] = initByUserColPartition match {
    case Some(m) =>
      require(m.length == p)
      m
    case None => List.fill(p)(0)
  }

  var rowPartitionEveryIteration: List[List[Int]] = List(rowPartition)
  var componentsEveryIterations: List[List[List[MultivariateGaussian]]] = List(parametersEstimation)

  var countRowCluster: ListBuffer[Int] = partitionToOrderedCount(rowPartition).to[ListBuffer]
  var countColCluster: ListBuffer[Int] = partitionToOrderedCount(colPartition).to[ListBuffer]

  var NIWParamsByRow: ListBuffer[ListBuffer[NormalInverseWishart]] = initByUserNIW match {
    case Some(m) =>
      require(initByUserColPartition.isDefined & initByUserRowPartition.isDefined, "When supplying an initial NIW set, the colPartition and rowPartition must also be supplied")
      require(m.length == countRowCluster.length, "The cluster number in the supplied colPartition does not match the first dimension of supplied NIW set")
      require(m.head.length == countColCluster.length, "The cluster number in the supplied rowPartition does not match the second dimension of supplied NIW set")
      m
    case None => initializeNIW
  }

  var updateAlphaFlag: Boolean = checkAlphaPrior(alpha, alphaPrior)

  var actualAlphaPrior: Gamma = alphaPrior match {
    case Some(g) => g
    case None => new Gamma(1D,1D)
  }

  var actualAlpha: Double = alpha match {
    case Some(a) =>
      require(a > 0, s"AlphaRow parameter is optional and should be > 0 if provided, but got $a")
      a
    case None => actualAlphaPrior.mean
  }


  def setColPartitionAndNIWParams(newColPartition: List[Int], newNIWParamByRow: ListBuffer[ListBuffer[NormalInverseWishart]]): Unit ={
    require(newColPartition.length == p)
    countColCluster = partitionToOrderedCount(newColPartition).to[ListBuffer]
    colPartition = newColPartition
    require(newNIWParamByRow.head.length == countColCluster.length, "The cluster number in the supplied colPartition does not match the first dimension of supplied NIW set")
    NIWParamsByRow = newNIWParamByRow
  }

  def initializeNIW: ListBuffer[ListBuffer[NormalInverseWishart]] = {
    (DataByRow zip rowPartition).groupBy(_._2).values.map(e => {
      val dataPerRowCluster = e.map(_._1).transpose
      val k = e.head._2
      (k, (dataPerRowCluster zip colPartition).groupBy(_._2).values.map(f => {
        val dataPerBlock = f.map(_._1).reduce(_++_)
        val l = f.head._2
        (l, prior.update(dataPerBlock))
      }).toList.sortBy(_._1).map(_._2).to[ListBuffer])
    }).toList.sortBy(_._1).map(_._2).to[ListBuffer]
  }

  def checkAlphaPrior(alpha: Option[Double], alphaPrior: Option[Gamma]): Boolean = {
    require(!(alpha.isEmpty & alphaPrior.isEmpty),"Either alphaRow or alphaRowPrior must be provided: please provide one of the two parameters.")
    require(!(alpha.isDefined & alphaPrior.isDefined), "Providing both alphaRow or alphaRowPrior is not supported: remove one of the two parameters.")
    alphaPrior.isDefined
  }

  def priorPredictive(idx: Int): Double = {
    val row = DataByRow(idx)
    (row zip colPartition).groupBy(_._2).values.par.map(e => {
      val currentData = e.map(_._1)
      prior.jointPriorPredictive(currentData)
    }).toList.sum
  }

  def computeClusterMembershipProbabilities(idx: Int,
                                            verbose: Boolean=false): List[Double] = {

    val row = DataByRow(idx)
    val rowByCol = (row zip colPartition).groupBy(_._2).map(v => (v._1, v._2.map(_._1)))
    NIWParamsByRow.indices.par.map(k => {
      (k, NIWParamsByRow.head.indices.par.map(l => {
        NIWParamsByRow(k)(l).jointPriorPredictive(rowByCol(l))
      }).sum + log(countRowCluster(k)))
    }).toList.sortBy(_._1).map(_._2)
  }

  def drawMembership(idx: Int, verbose : Boolean = false): Int = {

    val probPartition = computeClusterMembershipProbabilities(idx, verbose)
    val posteriorPredictiveXi = priorPredictive(idx)
    val probs = probPartition :+ (posteriorPredictiveXi + log(actualAlpha))
    val normalizedProbs = normalizeLogProbability(probs)
    sample(normalizedProbs)
  }

  private def removeElementFromRowCluster(row: List[DenseVector[Double]], currentPartition: Int): Unit = {
    if (countRowCluster(currentPartition) == 1) {
      countRowCluster.remove(currentPartition)
      NIWParamsByRow.remove(currentPartition)
      rowPartition = rowPartition.map(c => { if( c > currentPartition ){ c - 1 } else c })
    } else {
      countRowCluster.update(currentPartition, countRowCluster.apply(currentPartition) - 1)
      (row zip colPartition).groupBy(_._2).values.foreach(e => {
        val l = e.head._2
        val dataInCol = e.map(_._1)
        NIWParamsByRow(currentPartition).update(l, NIWParamsByRow(currentPartition)(l).removeObservations(dataInCol))
      })
    }
  }

  private def addElementToRowCluster(row: List[DenseVector[Double]],
                                     newPartition: Int): Unit = {

    if (newPartition == countRowCluster.length) {
      countRowCluster = countRowCluster ++ ListBuffer(1)
      NIWParamsByRow = NIWParamsByRow :+ (row zip colPartition).groupBy(_._2).values.map(e => {
        val l = e.head._2
        val dataInCol = e.map(_._1)
        (l, this.prior.update(dataInCol))
      }).to[ListBuffer].sortBy(_._1).map(_._2)
    } else {
      countRowCluster.update(newPartition, countRowCluster.apply(newPartition) + 1)
      (row zip colPartition).groupBy(_._2).values.foreach(e => {
        val l = e.head._2
        val dataInCol = e.map(_._1)
        NIWParamsByRow(newPartition).update(l,
          NIWParamsByRow(newPartition)(l).update(dataInCol))
      })
    }
  }

  def updatePartition(): Unit = {
    for (idx <- DataByRow.indices) {
      val currentData = DataByRow(idx)
      val currentPartition = rowPartition(idx)
      removeElementFromRowCluster(currentData, currentPartition)
      val newPartition = drawMembership(idx)
      rowPartition = rowPartition.updated(idx, newPartition)
      addElementToRowCluster(currentData, newPartition)
    }

    rowPartitionEveryIteration = rowPartitionEveryIteration :+ rowPartition

  }

  def likelihood(componentsByRow: List[List[MultivariateGaussian]]): Double = {

    val rowPartitionDensity = probabilityPartition

    require(countRowCluster.length == componentsByRow.length)
    require(countColCluster.length == componentsByRow.head.length)

    val dataLikelihood = DataByRow.indices.par.map(i => {
      DataByRow.head.indices.map(j => {
        componentsByRow(rowPartition(i))(colPartition(j)).logPdf(DataByRow(i)(j))
      }).sum
    }).sum

    val paramsDensity = componentsByRow.reduce(_++_).map(prior.logPdf).sum
    rowPartitionDensity + paramsDensity + dataLikelihood
  }

  def parametersEstimation: List[List[MultivariateGaussian]] = {

    (DataByRow zip rowPartition).groupBy(_._2).values.par.map(e => {
      val dataInRowCluster = e.map(_._1)
      val k = e.head._2
      (k,
        (dataInRowCluster.transpose zip colPartition).groupBy(_._2).values.par.map(f => {
          val dataInBlock = f.map(_._1).reduce(_++_)
          val l = f.head._2
          val sufficientStatistic = prior.update(dataInBlock)
          (l, sufficientStatistic.expectation())
        }).toList.sortBy(_._1).map(_._2))

    }).toList.sortBy(_._1).map(_._2)
  }

  def probabilityPartition: Double = {
    countRowCluster.length * log(actualAlpha) +
      countRowCluster.map(c => Common.Tools.logFactorial(c - 1)).sum -
      (0 until n).map(e => log(actualAlpha + e.toDouble)).sum
  }


  def run(nIter: Int = 10,
          verbose: Boolean = false,
          printLikelihood: Boolean = false): (List[List[Int]], List[List[List[MultivariateGaussian]]], List[Double]) = {

    var likelihoodEveryIteration = List(likelihood(componentsEveryIterations.head))

    @tailrec def go(iter: Int): Unit = {

      if(verbose){
        println("\n Clustering >>>>>> Iteration: " + iter.toString)
        Common.Tools.prettyPrintLBM(countRowCluster.toList, countColCluster.toList)
      }

      if (iter <= nIter) {

        var t0 = System.nanoTime()

        updatePartition()

        if(verbose){
          t0 = printTime(t0, "draw row Partition")
        }

        if(updateAlphaFlag){actualAlpha = updateAlpha(actualAlpha, actualAlphaPrior, countRowCluster.length, n)}
        val components = parametersEstimation

        componentsEveryIterations = componentsEveryIterations ++ List(components)

        likelihoodEveryIteration =  likelihoodEveryIteration ++ List(likelihood(components))

        go(iter + 1)
      }
    }

   go(1)


    (rowPartitionEveryIteration,  componentsEveryIterations, likelihoodEveryIteration)

  }


}
