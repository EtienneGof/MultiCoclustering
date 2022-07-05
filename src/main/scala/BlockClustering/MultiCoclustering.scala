package BlockClustering

import Common.NormalInverseWishart
import Common.ProbabilisticTools.{normalizeLogProbability, sample, updateAlpha}
import Common.Tools._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.log
import scala.annotation.tailrec
import scala.collection.mutable.ListBuffer
import breeze.stats.distributions.{Gamma, MultivariateGaussian}


/**
  * Implements the inference of a multi-coclustering model, assuming the existing of several coclustering structures.
  * @param DataByRow Dataset as list of list, with the first list indexing the rows, and the second the columns
  * @param alphaRowPrior Observation partition concentration parameter prior.
  * @param alphaColPrior Variable partition concentration parameter prior.
  * @param alphaRedundantColPrior Variable assignement in the coclustering structures.
  * @param initByUserPrior (optional) Prior distribution (Normal Inverse Wishart) on the mixture components parameters.
  * @param initByUserRowPartitions (optional) Initial observation partitions that acts as a starting point for the inference.
  * @param initByUserColPartitions (optional) Initial variable partitions that acts as a starting point for the inference.
  * @param initByUserRedundantColPartition (optional) Initial variable partitions that acts as a starting point for the inference.
  */
class MultiCoclustering(val DataByRow: List[List[DenseVector[Double]]],
                        var alphaRowPrior: Gamma,
                        var alphaColPrior: Gamma,
                        var alphaRedundantColPrior: Gamma,
                        var initByUserPrior: Option[NormalInverseWishart] = None,
                        var initByUserRowPartitions: Option[List[List[Int]]] = None,
                        var initByUserColPartitions: Option[List[List[Int]]] = None,
                        var initByUserRedundantColPartition: Option[List[Int]] = None) {

  val DataByCol: List[List[DenseVector[Double]]] = DataByRow.transpose

  val prior: NormalInverseWishart = initByUserPrior match {
    case Some(pr) => pr
    case None => new NormalInverseWishart(DataByRow)
  }

  val p: Int = DataByCol.length
  val n: Int = DataByRow.length

  var redundantColPartition: List[Int] = initByUserRedundantColPartition match {
    case Some(m) =>
      require(m.length == p)
      m
    case None => List.fill(p)(0)
  }

  var countRedundantColCluster: ListBuffer[Int] = partitionToOrderedCount(redundantColPartition).to[ListBuffer]
  var nRedundantColCluster: Int = countRedundantColCluster.length


  var rowPartitions: ListBuffer[List[Int]] = initByUserRowPartitions match {
    case Some(m) =>
      require(m.length == nRedundantColCluster,
        "The number of supplied row partitions should equal the redundant column cluster number")
      m.to[ListBuffer]
    case None => ListBuffer.fill(nRedundantColCluster)(List.fill(n)(0))
  }


  var colPartitions: ListBuffer[List[Int]] = initByUserColPartitions match {
    case Some(m) =>
      require(m.length == nRedundantColCluster,
        "The number of supplied col partitions should equal the redundant column cluster number")
      m.to[ListBuffer]
    case None => countRedundantColCluster.map(List.fill(_)(0))
  }

  val columnsPriorPredictiveAndRowMembership: List[(Double, List[Int])] = DataByCol.indices.par.map(j =>
    (j, columnPriorPredictive(j))
  ).toList.sortBy(_._1).map(_._2).toArray.toList

  val columnsPriorPredictive: List[Double] = columnsPriorPredictiveAndRowMembership.map(_._1)
  val rowMembershipPriorPredictiveEachVar: List[List[Int]] = columnsPriorPredictiveAndRowMembership.map(_._2)
  val nClusterEachVar: List[Int] = rowMembershipPriorPredictiveEachVar.map(_.distinct.length)
  val countClusterEachVar: List[ListBuffer[Int]] = rowMembershipPriorPredictiveEachVar.map(partitionToOrderedCount(_).to[ListBuffer])

  var alphaRedundantCol: Double = updateAlpha(alphaRedundantColPrior.mean, alphaRedundantColPrior, countRedundantColCluster.length, p)

  def columnPriorPredictive(colIdx: Int): (Double, List[Int]) = {

    val column = DataByCol(colIdx).toArray.toList
    val dataMat = DenseMatrix(column.toArray).reshape(1, column.length)
    val globalMean = Common.ProbabilisticTools.meanListDV(column)
    val globalVariance = Common.ProbabilisticTools.covariance(column, globalMean) + DenseMatrix.eye[Double](globalMean.length) * 1E-8
    val d = dataMat(0, 0).length
    val prior = new NormalInverseWishart(globalMean, 1D, globalVariance, d + 1)
    val mm = new Clustering(column.map(List(_)), initByUserPrior = Some(prior), alphaPrior = Some(alphaRowPrior))
    val (membership, _, _) = mm.run(verbose = true)
    (priorDistributionInPartition(column, membership.last), membership.last)

  }

  def priorDistributionInPartition(column: List[DenseVector[Double]], partition: List[Int]): Double = {
    (column zip partition).groupBy(_._2).values.par.map(e => {
      val dataInBlock = e.map(_._1)
      prior.jointPriorPredictive(dataInBlock)
    }).sum
  }

  def computeRedundantColClusterMembershipProbabilities(colIdx: Int)(implicit d: DummyImplicit): List[Double] = {
    rowPartitions.indices.par.map(l => {
      (l, priorDistributionInPartition(DataByCol(colIdx), rowPartitions(l)) + log(countRedundantColCluster(l)))
    }).toList.sortBy(_._1).map(_._2)

  }

  def parametersEstimation: List[List[List[MultivariateGaussian]]] = {
      (DataByCol zip redundantColPartition).groupBy(_._2).values.par.map(e => {
        val dataInCol = e.map(_._1)
        val h = e.head._2
        val m = new Coclustering(dataInCol.transpose,
          alphaRowPrior = Some(alphaRowPrior),
          alphaColPrior = Some(alphaColPrior),
          initByUserPrior = Some(prior),
          initByUserRowPartition = Some(rowPartitions(h)),
          initByUserColPartition = Some(colPartitions(h)))
        (h, m.componentsEstimation)
      }).toList.sortBy(_._1).map(e => e._2)
  }

  def drawColMemberships(colIdx: Int,
                         verbose : Boolean = false): Int = {
    val probMembership = computeRedundantColClusterMembershipProbabilities(colIdx)
    val probs = probMembership :+ (columnsPriorPredictive(colIdx) + log(alphaRedundantCol))
    val normalizedProbs = normalizeLogProbability(probs)
    sample(normalizedProbs)
  }

  def removeElementFromRedundantCluster(column: List[DenseVector[Double]], currentColMembership: Int): Unit = {
    if (countRedundantColCluster(currentColMembership) == 1) {
      countRedundantColCluster.remove(currentColMembership)
      rowPartitions.remove(currentColMembership)
      colPartitions.remove(currentColMembership)
      redundantColPartition = redundantColPartition.map(c => { if( c > currentColMembership ){ c - 1 } else c })
    } else {
      countRedundantColCluster.update(currentColMembership, countRedundantColCluster.apply(currentColMembership) - 1)
    }
  }

  def addElementToRedundantCluster(j:Int, newColMembership: Int): Unit = {
    if (newColMembership == countRedundantColCluster.length) {
      countRedundantColCluster = countRedundantColCluster ++ ListBuffer(1)
      rowPartitions = rowPartitions ++ List(rowMembershipPriorPredictiveEachVar(j))
      colPartitions = colPartitions ++ List(List(0))
    } else {
      countRedundantColCluster.update(newColMembership, countRedundantColCluster.apply(newColMembership) + 1)
    }
  }

  def updateRedundantColMembership(): Unit = {
    for (j <- DataByCol.indices) {
      val currentData = DataByCol(j)
      val currentMembership = redundantColPartition(j)
      removeElementFromRedundantCluster(currentData, currentMembership)
      val newMembership = drawColMemberships(j)
      redundantColPartition = redundantColPartition.updated(j, newMembership)
      addElementToRedundantCluster(j, newMembership)
    }


    require(rowPartitions.length == colPartitions.length)

  }

  def updateCoclusteringStructures(): Unit = {
    val res = (DataByCol zip redundantColPartition).groupBy(_._2).values.par.map(e => {
      val dataInCol = e.map(_._1)
      val h = e.head._2
      val m = new Coclustering(dataInCol.transpose,
        alphaRowPrior = Some(alphaRowPrior),
        alphaColPrior = Some(alphaColPrior),
        initByUserPrior = Some(prior),
        initByUserRowPartition = Some(rowPartitions(h)))

      m.run(5)
      (h, m.rowClustering.rowPartition, m.colClustering.rowPartition)
    }).toList.sortBy(_._1).map(e => (e._2, e._3))
    rowPartitions = res.map(_._1).to[ListBuffer]
    colPartitions = res.map(_._2).to[ListBuffer]
  }

  def run(nIter: Int = 10,
          verbose: Boolean = false,
          printLikelihood: Boolean = false): (List[Int], List[List[Int]], List[List[Int]]) = {

    @tailrec def go(iter: Int): Unit = {

      if(verbose){
        println("\n MultiCoclustering >>>>>> Iteration: " + iter.toString)
      }

      if (iter <= nIter) {

        var t0 = System.nanoTime()

        updateRedundantColMembership()

        if(verbose){
          t0 = printTime(t0, "draw variable partition")
        }

        updateCoclusteringStructures()

        if(verbose){
          t0 = printTime(t0, "update coclustering structures")
        }

        go(iter + 1)
      }
    }

    go(1)

    (redundantColPartition, rowPartitions.toList, colPartitions.toList)

  }

}
