package MCC

import Common.NormalInverseWishart
import Common.ProbabilisticTools._
import Common.Tools._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.log
import breeze.stats.distributions.Gamma

import scala.annotation.tailrec
import scala.collection.immutable
import scala.collection.mutable.ArrayBuffer

class CollapsedGibbsSampler(val Data: Array[Array[DenseVector[Double]]],
                            var alphaPrior: Gamma,
                            var betaPrior : Gamma,
                            var gammaPrior: Gamma,
                            var initByUserPrior: Option[NormalInverseWishart] = None,
                            var initByUserRowMembership: Option[Array[Array[Int]]] = None,
                            var initByUserColMembership: Option[Array[Int]] = None) extends Serializable {

  val n  : Int = Data.head.length
  val p  : Int = Data.length
  val d  : Int = Data.head.head.length

  val prior: NormalInverseWishart = initByUserPrior match {
    case Some(pr) => pr
    case None => new NormalInverseWishart(Data)
  }

  var redundantColPartition: Array[Int] = initByUserColMembership match {
    case Some(m) =>
      require(m.length == p)
      m
    case None => Array.fill(p)(0)
  }

  var countRedundantColCluster: ArrayBuffer[Int] = partitionToOrderedCount(redundantColPartition).to[ArrayBuffer]

  var rowPartitions: ArrayBuffer[Array[Int]] = initByUserRowMembership match {
    case Some(m) =>
      require(m.length == countRedundantColCluster.length)
      require(m.map(_.length) sameElements Array.fill(countRedundantColCluster.length)(n))
      m.to[ArrayBuffer]
    case None => ArrayBuffer.fill(countRedundantColCluster.length)(Array.fill(n)(0))
  }

  var countRowCluster: ArrayBuffer[ArrayBuffer[Int]] = rowPartitions.map(partitionToOrderedCount(_).to[ArrayBuffer])

  var correlatedColPartitions: ArrayBuffer[Array[Int]] = initByUserRowMembership match {
    case Some(m) =>
      require(m.length == redundantColPartition.length)
      m.to[ArrayBuffer]
    case None => countRedundantColCluster.map(p_h => ArrayBuffer.fill(p_h)(0).toArray)
  }

  var countCorrelatedColCluster: ArrayBuffer[ArrayBuffer[Int]] = correlatedColPartitions.map(partitionToOrderedCount(_).to[ArrayBuffer])

  var alphas: ArrayBuffer[Double] = countRowCluster.indices.map(h => {
    updateAlpha(alphaPrior.mean, alphaPrior, countRowCluster(h).length, n)
  }).to[ArrayBuffer]

  var betas: ArrayBuffer[Double] = countCorrelatedColCluster.indices.map(h => {
    updateAlpha(betaPrior.mean, betaPrior, countCorrelatedColCluster(h).length, countRedundantColCluster(h))
  }).to[ArrayBuffer]

  var gamma: Double = updateAlpha(gammaPrior.mean, gammaPrior, countRedundantColCluster.length, p)

  val columnsPriorPredictiveAndRowMembership: Array[(Double, Array[Int])] = Data.indices.par.map(j =>
    (j, columnPriorPredictive(j))
  ).toArray.sortBy(_._1).map(_._2)

  val columnsPriorPredictive: Array[Double] = columnsPriorPredictiveAndRowMembership.map(_._1)
  val rowMembershipPriorPredictiveEachVar: Array[Array[Int]] = columnsPriorPredictiveAndRowMembership.map(_._2)
  val nClusterEachVar: Array[Int] = rowMembershipPriorPredictiveEachVar.map(_.distinct.length)
  val countClusterEachVar: Array[ArrayBuffer[Int]] = rowMembershipPriorPredictiveEachVar.map(partitionToOrderedCount(_).to[ArrayBuffer])

  var alphasEachVar: immutable.IndexedSeq[Double] = nClusterEachVar.indices.map(j =>
    updateAlpha(alphaPrior.mean, alphaPrior, nClusterEachVar(j), n)
  )

  def getRowPartitions: Array[Array[Int]] = rowPartitions.toArray

  def getColPartition: Array[Int] = redundantColPartition

  def checkAlphaPrior(alpha: Option[Double], alphaPrior: Option[Gamma]): Boolean = {
    require(!(alpha.isEmpty & alphaPrior.isEmpty),"Either alpha or alphaPrior must be provided: please provide one of the two parameters.")
    require(!(alpha.isDefined & alphaPrior.isDefined), "Providing both alpha or alphaPrior is not supported: remove one of the two parameters.")
    alphaPrior.isDefined
  }

  def columnPriorPredictive(colIdx: Int): (Double, Array[Int]) = {

    val column = Data(colIdx)
    val dataMat = DenseMatrix(column).reshape(1, column.length)
    val globalMean = Common.ProbabilisticTools.meanArrayDV(column)
    val globalVariance = Common.ProbabilisticTools.covariance(column, globalMean) + DenseMatrix.eye[Double](globalMean.length) * 1E-8
    val d = dataMat(0, 0).length
    val prior = new NormalInverseWishart(globalMean, 1D, globalVariance, d + 1)
    val mm = new DPMM.CollapsedGibbsSampler(column, prior, alphaPrior = alphaPrior)
    val (membership, _, _) = mm.run()
    (priorDistributionInPartition(column, membership.last), membership.last)

  }

  def priorDistributionInPartition(column: Array[DenseVector[Double]], partition: Array[Int]): Double = {
    (column zip partition).groupBy(_._2).values.par.map(e => {
      val dataInBlock = e.map(_._1)
      prior.jointPriorPredictive(dataInBlock)
    }).sum
  }

  def computeColClusterMembershipProbabilities(colIdx: Int,
                                               verbose: Boolean = false)(implicit d: DummyImplicit): Array[Double] = {
    rowPartitions.indices.par.map(l => {
      (l, priorDistributionInPartition(Data(colIdx), rowPartitions(l)) + log(countRedundantColCluster(l)))
    }).toArray.sortBy(_._1).map(_._2)

  }

  def drawColMemberships(colIdx: Int,
                         verbose : Boolean = false): Int = {

    val probMembership = computeColClusterMembershipProbabilities(colIdx, verbose)
    val membershipProbabilities = probMembership :+ (columnsPriorPredictive(colIdx) + log(gamma))
    val normalizedProbs = normalizeLogProbability(membershipProbabilities)
    sample(normalizedProbs)
  }

  private def removeElementFromColCluster(column: Array[DenseVector[Double]], currentColMembership: Int): Unit = {
    if (countRedundantColCluster(currentColMembership) == 1) {
      countRedundantColCluster.remove(currentColMembership)
      countRowCluster.remove(currentColMembership)
      rowPartitions.remove(currentColMembership)
      redundantColPartition = redundantColPartition.map(c => { if( c > currentColMembership ){ c - 1 } else c })
    } else {
      countRedundantColCluster.update(currentColMembership, countRedundantColCluster.apply(currentColMembership) - 1)
    }
  }

  private def addElementToColCluster(j:Int,
                                     newColMembership: Int): Unit = {

    if (newColMembership == countRedundantColCluster.length) {
      countRedundantColCluster = countRedundantColCluster ++ ArrayBuffer(1)
      val alphaNewCluster = updateAlpha(alphaPrior.mean, alphaPrior, nClusterEachVar(j), n)
      val betaNewCluster = updateAlpha(betaPrior.mean, betaPrior, 1, 1)
      alphas = alphas ++ Array(alphaNewCluster)
      betas = betas ++ Array(betaNewCluster)
      rowPartitions = rowPartitions ++ Array(rowMembershipPriorPredictiveEachVar(j))
      countRowCluster = countRowCluster ++ ArrayBuffer(countClusterEachVar(j))
    } else {
      countRedundantColCluster.update(newColMembership, countRedundantColCluster.apply(newColMembership) + 1)
    }
  }

  def updateRowMemberships(verbose: Boolean = false): Unit = {

    val res = (Data zip redundantColPartition).groupBy(_._2).values.par.map(e => {
      val dataInCol = e.map(_._1)
      val l = e.head._2
      val m = new NPLBM.CollapsedGibbsSampler(dataInCol,
        alpha = Some(alphas(l)),
        beta = Some(betas(l)),
        initByUserPrior = Some(prior),
        initByUserRowPartition = Some(rowPartitions(l)))
      m.run(5)
      (l, m.getRowPartition, m.getColPartition)
    }).toArray.sortBy(_._1).map(e => (e._2, e._3))

    rowPartitions = res.map(_._1).to[ArrayBuffer]
    correlatedColPartitions = res.map(_._2).to[ArrayBuffer]
    countRowCluster = rowPartitions.map(partitionToOrderedCount(_).to[ArrayBuffer])
    countCorrelatedColCluster = correlatedColPartitions.map(partitionToOrderedCount(_).to[ArrayBuffer])

  }

  def updateColMembership(verbose: Boolean = false): Unit = {
    for (j <- Data.indices) {
      val currentData = Data(j)
      val currentMembership = redundantColPartition(j)
      removeElementFromColCluster(currentData, currentMembership)
      val newMembership = drawColMemberships(j)
      redundantColPartition = redundantColPartition.updated(j, newMembership)
      addElementToColCluster(j, newMembership)
    }
  }

  def run(nIter: Int = 10,
          verbose: Boolean = false): (Array[Int], Array[Array[Int]], Array[Array[Int]], Double) = {

    var rowMembershipsEveryIteration = Array(rowPartitions.toArray)
    var colMembershipEveryIteration = Array(redundantColPartition)

    @tailrec def go(iter: Int): Unit = {

      if(verbose){
        println("\n>>>>>> Iteration: " + iter.toString)
        Common.Tools.prettyPrintCLBM(countRowCluster.map(_.toArray).toArray, countRedundantColCluster.toArray)
      }

      if (iter > nIter) {

      } else {

        var t0 = System.nanoTime()

        updateColMembership()

        if(verbose) {
          t0 = printTime(t0, "draw col Membership")
        }

        updateRowMemberships()

        if(verbose) {
          t0 = printTime(t0, "draw row Membership")
        }

        alphas = countRedundantColCluster.indices.map(l => updateAlpha(alphas(l), alphaPrior, countRowCluster(l).length, n)).to[ArrayBuffer]
        betas = countRedundantColCluster.indices.map(l => updateAlpha(betas(l), betaPrior, countCorrelatedColCluster(l).length, countCorrelatedColCluster(l).sum)).to[ArrayBuffer]
        gamma = updateAlpha(gamma, gammaPrior, countRedundantColCluster.length, p)

        rowMembershipsEveryIteration = rowMembershipsEveryIteration :+ rowPartitions.toArray
        colMembershipEveryIteration = colMembershipEveryIteration :+ redundantColPartition

        go(iter + 1)
      }
    }

    go(1)

    val likelihood = prior.MCCLikelihood(alphaPrior,
      betaPrior,
      gammaPrior,
      alphas.toList,
      betas.toList,
      gamma,
      Data,
      redundantColPartition,
      correlatedColPartitions.toArray,
      rowPartitions.toArray,
      countRedundantColCluster.toArray,
      countCorrelatedColCluster.map(_.toArray).toArray,
      countRowCluster.map(_.toArray).toArray)
    (redundantColPartition, correlatedColPartitions.toArray, rowPartitions.toArray, likelihood)
  }




}
