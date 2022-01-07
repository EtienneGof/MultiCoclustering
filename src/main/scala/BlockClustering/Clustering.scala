package BlockClustering

import Common.NormalInverseWishart
import Common.ProbabilisticTools._
import Common.Tools._
import breeze.linalg.DenseVector
import breeze.numerics.log
import breeze.stats.distributions.{Gamma, MultivariateGaussian}

import scala.annotation.tailrec
import scala.collection.mutable.ListBuffer

/** Implements the inference of a Dirichlet Process Mixture Model on a multivariate dataset of continuous observations.
  * The model assumes that several variables can share the same distribution (this information is given by initByUserColPartition)
  * and that elements in the same block cluster (i.e., the same observation cluster and the same column cluster) follow independently
  * the same multivariate Gaussian distribution.
  * If no variable partition is given, every variable is assumed to follow a different distribution.
  *
  * @param DataByRow Dataset as list of list, with the first list indexing the rows, and the second the columns
  * @param alpha (optional) Concentration parameter. Must be assigned a value if alphaPrior has none.
  * @param alphaPrior (Optional) Concentration parameter prior. Must be assigned a value if alpha has none.
  * @param initByUserPrior (optional) Prior distribution (Normal Inverse Wishart) on the mixture components parameters.
  * @param initByUserRowPartition (optional) Initial observation partition that acts as a starting point for the inference.
  * @param initByUserColPartition (optional) Variable partition, such that variables in the same variable cluster follow the same distribution.
  *                               and elements in the same
  * @param initByUserNIW (optional) Initial block component distribution hyper-parameters.
  */
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
    case None => (0 to p).toList
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


  /** Sets the variable partition and block component posterior predictive distribution parameters values
    *
    * @param newColPartition List of Int
    * @param newNIWParamByRow Nested List of Normal Inverse Wishart distributions.
    */
  def setColPartitionAndNIWParams(newColPartition: List[Int], newNIWParamByRow: ListBuffer[ListBuffer[NormalInverseWishart]]): Unit ={
    require(newColPartition.length == p)
    countColCluster = partitionToOrderedCount(newColPartition).to[ListBuffer]
    colPartition = newColPartition
    require(newNIWParamByRow.head.length == countColCluster.length, "The cluster number in the supplied colPartition does not match the first dimension of supplied NIW set")
    NIWParamsByRow = newNIWParamByRow
  }


  /** Default initialization of the nested list of posterior predictive distributions.
    *
    */
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

  /** Checks that either alpha or alphaPrior is assigned
    *
    * @param alpha
    * @param alphaPrior
    */
  def checkAlphaPrior(alpha: Option[Double], alphaPrior: Option[Gamma]): Boolean = {
    require(!(alpha.isEmpty & alphaPrior.isEmpty),"Either alphaRow or alphaRowPrior must be provided: please provide one of the two parameters.")
    require(!(alpha.isDefined & alphaPrior.isDefined), "Providing both alphaRow or alphaRowPrior is not supported: remove one of the two parameters.")
    alphaPrior.isDefined
  }

  /** Computes the prior predictive distribution of one observation
    *
    * @param idx Observation index
    */
  def priorPredictive(idx: Int): Double = {
    val row = DataByRow(idx)
    (row zip colPartition).groupBy(_._2).values.par.map(e => {
      val currentData = e.map(_._1)
      prior.jointPriorPredictive(currentData)
    }).toList.sum
  }

  /** Computes the cluster membership (existing clusters + new cluster discovery) probabilities.
    *
    * @param idx Index of the target observation
    */
  def computeClusterMembershipProbabilities(idx: Int): List[Double] = {

    val row = DataByRow(idx)
    val rowByCol = (row zip colPartition).groupBy(_._2).map(v => (v._1, v._2.map(_._1)))
    NIWParamsByRow.indices.par.map(k => {
      (k, NIWParamsByRow.head.indices.par.map(l => {
        NIWParamsByRow(k)(l).jointPriorPredictive(rowByCol(l))
      }).sum + log(countRowCluster(k)))
    }).toList.sortBy(_._1).map(_._2)
  }

  /** Update the membership of one observation
    *
    * @param idx Index of the target observation
    */
  def drawMembership(idx: Int): Int = {

    val probPartition = computeClusterMembershipProbabilities(idx)
    val posteriorPredictiveXi = priorPredictive(idx)
    val probs = probPartition :+ (posteriorPredictiveXi + log(actualAlpha))
    val normalizedProbs = normalizeLogProbability(probs)
    sample(normalizedProbs)
  }

  /** Before updating one observation's membership, decrements the cluster count and removes the information associated with
    * that observation from the corresponding block components (i.e. all the block components belonging to the associated row-cluster)
    *
    * @param idx Index of the target observation
    * @param formerMembership Previous membership of the target observation
    */
  private def removeElementFromRowCluster(idx: Int, formerMembership: Int): Unit = {
    if (countRowCluster(formerMembership) == 1) {
      countRowCluster.remove(formerMembership)
      NIWParamsByRow.remove(formerMembership)
      rowPartition = rowPartition.map(c => { if( c > formerMembership ){ c - 1 } else c })
    } else {
      countRowCluster.update(formerMembership, countRowCluster.apply(formerMembership) - 1)
      (DataByRow(idx) zip colPartition).groupBy(_._2).values.foreach(e => {
        val l = e.head._2
        val dataInCol = e.map(_._1)
        NIWParamsByRow(formerMembership).update(l, NIWParamsByRow(formerMembership)(l).removeObservations(dataInCol))
      })
    }
  }


  /** After having sampled a new membership for an observation, increments the cluster count and update the
    * associated block components with the information associated with that observation
    *
    * @param idx Index of the target observation
    * @param newMembership New membership of the target observation
    */
  private def addElementToRowCluster(idx: Int,
                                     newMembership: Int): Unit = {

    if (newMembership == countRowCluster.length) {
      countRowCluster = countRowCluster ++ ListBuffer(1)
      NIWParamsByRow = NIWParamsByRow :+ (DataByRow(idx) zip colPartition).groupBy(_._2).values.map(e => {
        val l = e.head._2
        val dataInCol = e.map(_._1)
        (l, this.prior.update(dataInCol))
      }).to[ListBuffer].sortBy(_._1).map(_._2)
    } else {
      countRowCluster.update(newMembership, countRowCluster.apply(newMembership) + 1)
      (DataByRow(idx) zip colPartition).groupBy(_._2).values.foreach(e => {
        val l = e.head._2
        val dataInCol = e.map(_._1)
        NIWParamsByRow(newMembership).update(l,
          NIWParamsByRow(newMembership)(l).update(dataInCol))
      })
    }
  }

  /** Update the membership of every observations in the dataset
    *
    */
  def updatePartition(): Unit = {
    for (idx <- DataByRow.indices) {
      val currentPartition = rowPartition(idx)
      removeElementFromRowCluster(idx, currentPartition)
      val newPartition = drawMembership(idx)
      rowPartition = rowPartition.updated(idx, newPartition)
      addElementToRowCluster(idx, newPartition)
    }

    rowPartitionEveryIteration = rowPartitionEveryIteration :+ rowPartition

  }

  /** Computes the model completed likelihood.
    *
    */
  def likelihood(): Double = {

    val rowPartitionDensity: Double = probabilityPartition

    val dataLikelihood: Double = (DataByRow zip rowPartition).groupBy(_._2).values.par.map(e => {
      val dataPerRowCluster = e.map(_._1).transpose
      (dataPerRowCluster zip colPartition).groupBy(_._2).values.map(f => {
        val dataPerBlock = f.map(_._1).reduce(_++_)
        prior.jointPriorPredictive(dataPerBlock)
      }).sum
    }).sum

    rowPartitionDensity + dataLikelihood
  }

  /** Returns the expectation of the block component parameters.
    *
    */
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

  /** Returns the density of the partition
    *
    */
  def probabilityPartition: Double = {
    countRowCluster.length * log(actualAlpha) +
      countRowCluster.map(c => Common.Tools.logFactorial(c - 1)).sum -
      (0 until n).map(e => log(actualAlpha + e.toDouble)).sum
  }


  /** launches the inference process
    *
    * @param nIter Iteration number
    * @param verbose Boolean activating the output of additional information (cluster count evolution)
    * @return
    */
  def run(nIter: Int = 10,
          verbose: Boolean = false): (List[List[Int]], List[List[List[MultivariateGaussian]]], List[Double]) = {

    var likelihoodEveryIteration = List(likelihood())

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

        likelihoodEveryIteration =  likelihoodEveryIteration ++ List(likelihood())

        go(iter + 1)
      }
    }

   go(1)


    (rowPartitionEveryIteration,  componentsEveryIterations, likelihoodEveryIteration)

  }


}
