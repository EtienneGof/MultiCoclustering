package NPLBM

import Common.NormalInverseWishart
import Common.ProbabilisticTools._
import Common.Tools._
import breeze.linalg.DenseVector
import breeze.numerics.log
import breeze.stats.distributions.{Gamma, MultivariateGaussian}

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer

class CollapsedGibbsSampler(val Data: Array[Array[DenseVector[Double]]],
                            var alpha: Option[Double] = None,
                            var beta: Option[Double] = None,
                            var alphaPrior: Option[Gamma] = None,
                            var betaPrior: Option[Gamma] = None,
                            var initByUserPrior: Option[NormalInverseWishart] = None,
                            var initByUserRowPartition: Option[Array[Int]] = None,
                            var initByUserColPartition: Option[Array[Int]] = None) extends Serializable {

  val DataByRow: Array[Array[DenseVector[Double]]] = Data.transpose
  val n: Int = Data.head.length
  val p: Int = Data.length

  val prior: NormalInverseWishart = initByUserPrior match {
    case Some(pr) => pr
    case None => new NormalInverseWishart(Data)
  }

  val d: Int = prior.d
  require(prior.d == Data.head.head.length, "Prior and data dimensions differ")

  var rowPartition: Array[Int] = initByUserRowPartition match {
    case Some(m) =>
      require(m.length == Data.head.length)
      m
    case None => Array.fill(n)(0)
  }

  var colPartition: Array[Int] = initByUserColPartition match {
    case Some(m) =>
      require(m.length == Data.length)
      m
    case None => Array.fill(p)(0)
  }

  var countRowCluster: ArrayBuffer[Int] = partitionToOrderedCount(rowPartition).to[ArrayBuffer]
  var countColCluster: ArrayBuffer[Int] = partitionToOrderedCount(colPartition).to[ArrayBuffer]

  var NIWParamsByCol: ArrayBuffer[ArrayBuffer[NormalInverseWishart]] = (Data zip colPartition).groupBy(_._2).values.map(e => {
    val dataPerColCluster = e.map(_._1).transpose
    val l = e.head._2
    (l, (dataPerColCluster zip rowPartition).groupBy(_._2).values.map(f => {
      val dataPerBlock = f.map(_._1).reduce(_++_)
      val k = f.head._2
      (k, prior.update(dataPerBlock))
    }).toArray.sortBy(_._1).map(_._2).to[ArrayBuffer])
  }).toArray.sortBy(_._1).map(_._2).to[ArrayBuffer]

  val updateAlphaFlag: Boolean = checkAlphaPrior(alpha, alphaPrior)
  val updateBetaFlag: Boolean = checkAlphaPrior(beta, betaPrior)

  val actualAlphaPrior: Gamma = alphaPrior match {
    case Some(g) => g
    case None => new Gamma(1D, 1D)
  }
  val actualBetaPrior: Gamma = betaPrior match {
    case Some(g) => g
    case None => new Gamma(1D, 1D)
  }

  var actualAlpha: Double = alpha match {
    case Some(a) =>
      require(a > 0, s"AlphaRow parameter is optional and should be > 0 if provided, but got $a")
      a
    case None => actualAlphaPrior.mean
  }

  var actualBeta: Double = beta match {
    case Some(a) =>
      require(a > 0, s"AlphaCol parameter is optional and should be > 0 if provided, but got $a")
      a
    case None => actualBetaPrior.mean
  }

  def getRowPartition: Array[Int] = rowPartition
  def getColPartition: Array[Int] = colPartition

  def getNIWParams: Seq[ArrayBuffer[NormalInverseWishart]] = NIWParamsByCol

  def checkAlphaPrior(alpha: Option[Double], alphaPrior: Option[Gamma]): Boolean = {
    require(!(alpha.isEmpty & alphaPrior.isEmpty),"Either alphaRow or alphaRowPrior must be provided: please provide one of the two parameters.")
    require(!(alpha.isDefined & alphaPrior.isDefined), "Providing both alphaRow or alphaRowPrior is not supported: remove one of the two parameters.")
    alphaPrior.isDefined
  }

  def priorPredictive(line: Array[DenseVector[Double]],
                      partitionOtherDim: Array[Int]): Double = {

    (line zip partitionOtherDim).groupBy(_._2).values.par.map(e => {
      val currentData = e.map(_._1)
      prior.jointPriorPredictive(currentData)
    }).toArray.sum
  }

  def computeClusterMembershipProbabilities(x: Array[DenseVector[Double]],
                                            partitionOtherDimension: Array[Int],
                                            countCluster: ArrayBuffer[Int],
                                            NIWParams: ArrayBuffer[ArrayBuffer[NormalInverseWishart]],
                                            verbose: Boolean=false): Array[Double] = {

    val xByRow = (x zip partitionOtherDimension).groupBy(_._2).map(v => (v._1, v._2.map(_._1)))
    NIWParams.indices.par.map(l => {
      (l, NIWParams.head.indices.par.map(k => {
        NIWParams(l)(k).jointPriorPredictive(xByRow(k))
      }).sum + log(countCluster(l)))
    }).toArray.sortBy(_._1).map(_._2)
  }

  def drawMembership(x: Array[DenseVector[Double]],
                     partitionOtherDimension: Array[Int],
                     countCluster: ArrayBuffer[Int],
                     NIWParams: ArrayBuffer[ArrayBuffer[NormalInverseWishart]],
                     alpha: Double,
                     verbose : Boolean = false): Int = {

    val probPartition = computeClusterMembershipProbabilities(x, partitionOtherDimension, countCluster, NIWParams, verbose)
    val posteriorPredictiveXi = priorPredictive(x, partitionOtherDimension)
    val membershipProbabilities = probPartition :+ (posteriorPredictiveXi + log(alpha))
    val normalizedProbs = normalizeLogProbability(membershipProbabilities)
    sample(normalizedProbs)
  }

  private def removeElementFromRowCluster(row: Array[DenseVector[Double]], currentPartition: Int): Unit = {
    if (countRowCluster(currentPartition) == 1) {
      countRowCluster.remove(currentPartition)
      NIWParamsByCol.map(_.remove(currentPartition))
      rowPartition = rowPartition.map(c => { if( c > currentPartition ){ c - 1 } else c })
    } else {
      countRowCluster.update(currentPartition, countRowCluster.apply(currentPartition) - 1)
      (row zip colPartition).groupBy(_._2).values.foreach(e => {
        val l = e.head._2
        val dataInCol = e.map(_._1)
        NIWParamsByCol(l).update(currentPartition, NIWParamsByCol(l)(currentPartition).removeObservations(dataInCol))
      })
    }
  }

  private def removeElementFromColCluster(column: Array[DenseVector[Double]], currentPartition: Int): Unit = {
    if (countColCluster(currentPartition) == 1) {
      countColCluster.remove(currentPartition)
      NIWParamsByCol.remove(currentPartition)
      colPartition = colPartition.map(c => { if( c > currentPartition ){ c - 1 } else c })
    } else {
      countColCluster.update(currentPartition, countColCluster.apply(currentPartition) - 1)
      (column zip rowPartition).groupBy(_._2).values.foreach(e => {
        val k = e.head._2
        val dataInCol = e.map(_._1)
        NIWParamsByCol(currentPartition).update(k, NIWParamsByCol(currentPartition)(k).removeObservations(dataInCol))
      })
    }
  }

  private def addElementToRowCluster(row: Array[DenseVector[Double]],
                                     newPartition: Int): Unit = {

    if (newPartition == countRowCluster.length) {
      countRowCluster = countRowCluster ++ ArrayBuffer(1)
      (row zip colPartition).groupBy(_._2).values.foreach(e => {
        val l = e.head._2
        val dataInCol = e.map(_._1)
        val newNIWparam = this.prior.update(dataInCol)
        NIWParamsByCol(l) = NIWParamsByCol(l) ++ ArrayBuffer(newNIWparam)
      })
    } else {
      countRowCluster.update(newPartition, countRowCluster.apply(newPartition) + 1)
      (row zip colPartition).groupBy(_._2).values.foreach(e => {
        val l = e.head._2
        val dataInCol = e.map(_._1)
        NIWParamsByCol(l).update(newPartition,
          NIWParamsByCol(l)(newPartition).update(dataInCol))
      })
    }
  }

  private def addElementToColCluster(column: Array[DenseVector[Double]],
                                     newPartition: Int): Unit = {

    if (newPartition == countColCluster.length) {
      countColCluster = countColCluster ++ ArrayBuffer(1)
      val newCluster = (column zip rowPartition).groupBy(_._2).values.map(e => {
        val k = e.head._2
        val dataInRow = e.map(_._1)
        (k, this.prior.update(dataInRow))
      }).toArray.sortBy(_._1).map(_._2).to[ArrayBuffer]
      NIWParamsByCol = NIWParamsByCol ++ ArrayBuffer(newCluster)
    } else {
      countColCluster.update(newPartition, countColCluster.apply(newPartition) + 1)
      (column zip rowPartition).groupBy(_._2).values.foreach(e => {
        val k = e.head._2
        val dataInCol = e.map(_._1)
        NIWParamsByCol(newPartition).update(k,
          NIWParamsByCol(newPartition)(k).update(dataInCol))
      })
    }
  }

  def updateRowPartition(verbose: Boolean = false): Unit = {
    for (i <- DataByRow.indices) {
      val currentData = DataByRow(i)
      val currentPartition = rowPartition(i)
      removeElementFromRowCluster(currentData, currentPartition)
      val newPartition = drawMembership(currentData, colPartition, countRowCluster, NIWParamsByCol.transpose, actualAlpha)
      rowPartition = rowPartition.updated(i, newPartition)
      addElementToRowCluster(currentData, newPartition)
    }
  }

  def updateColPartition(verbose: Boolean = false): Unit = {
    for (i <- Data.indices) {
      val currentData = Data(i)
      val currentPartition = colPartition(i)
      removeElementFromColCluster(currentData, currentPartition)
      val newMembership = drawMembership(currentData, rowPartition, countColCluster, NIWParamsByCol, actualBeta)
      colPartition = colPartition.updated(i, newMembership)
      addElementToColCluster(currentData, newMembership)
    }
  }

  def run(nIter: Int = 10,
          verbose: Boolean = false): (Array[Array[Int]], Array[Array[Int]], Array[Array[Array[MultivariateGaussian]]]) = {

    @tailrec def go(rowPartitionEveryIteration: Array[Array[Int]],
                    colPartitionEveryIteration: Array[Array[Int]],
                    iter: Int):
    (Array[Array[Int]], Array[Array[Int]])= {

      if(verbose){
        println("\n>>>>>> Iteration: " + iter.toString)
        Common.Tools.prettyPrintLBM(countRowCluster.toArray, countColCluster.toArray)
      }

      if (iter > nIter) {

        (rowPartitionEveryIteration, colPartitionEveryIteration)

      } else {

        var t0 = System.nanoTime()

        updateRowPartition()

        if(verbose){
          t0 = printTime(t0, "draw row Partition")
        }

        updateColPartition()

        if(verbose){
          t0 = printTime(t0, "draw col Partition")
        }

        if(updateAlphaFlag){actualAlpha = updateAlpha(actualAlpha, actualAlphaPrior, countRowCluster.length, n)}
        if(updateBetaFlag){actualBeta = updateAlpha(actualBeta, actualBetaPrior, countColCluster.length, p)}

        go(rowPartitionEveryIteration :+ rowPartition,
          colPartitionEveryIteration :+ colPartition,
          iter + 1)
      }
    }

    val (rowPartitionEveryIterations, colPartitionEveryIterations) = go(Array(rowPartition), Array(colPartition), 1)

    val componentsEveryIterations = rowPartitionEveryIterations.indices.map(i => {
      clustersParametersEstimation(Data, prior, rowPartitionEveryIterations(i), colPartitionEveryIterations(i))
    }).toArray
    (rowPartitionEveryIterations, colPartitionEveryIterations,  componentsEveryIterations)

  }


  def runWithFixedPartitions(nIter: Int = 10,
                              updateCol: Boolean = false,
                              updateRow: Boolean = true,
                              verbose: Boolean = false): (Array[Array[Int]], Array[Array[Int]], Array[Array[Array[MultivariateGaussian]]]) = {

    @tailrec def go(rowPartitionEveryIteration: Array[Array[Int]],
                    colPartitionEveryIteration: Array[Array[Int]],
                    iter: Int):
    (Array[Array[Int]], Array[Array[Int]])= {

      if(verbose){
        println("\n>>>>>> Iteration: " + iter.toString)
        Common.Tools.prettyPrintLBM(countRowCluster.toArray, countColCluster.toArray)
      }

      if (iter > nIter) {

        (rowPartitionEveryIteration, colPartitionEveryIteration)

      } else {

        var t0 = System.nanoTime()

        if(updateRow){

          updateRowPartition()

          if(verbose){
            t0 = printTime(t0, "draw row Partition")
          }

          if(updateAlphaFlag){actualAlpha = updateAlpha(actualAlpha, actualAlphaPrior, countRowCluster.length, n)}

        }

        if(updateCol){

          updateColPartition()

          if(verbose){
            t0 = printTime(t0, "draw col Partition")
          }

          if(updateBetaFlag){actualBeta = updateAlpha(actualBeta, actualBetaPrior, countColCluster.length, p)}

        }

        go(rowPartitionEveryIteration :+ rowPartition,
          colPartitionEveryIteration :+ colPartition,
          iter + 1)
      }
    }

    val (rowPartitionEveryIterations, colPartitionEveryIterations) = go(Array(rowPartition), Array(colPartition), 1)

    val componentsEveryIterations = rowPartitionEveryIterations.indices.map(i => {
      clustersParametersEstimation(Data, prior, rowPartitionEveryIterations(i), colPartitionEveryIterations(i))
    }).toArray
    (rowPartitionEveryIterations, colPartitionEveryIterations,  componentsEveryIterations)

  }
}
