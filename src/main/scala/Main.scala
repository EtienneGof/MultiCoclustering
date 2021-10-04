
import Common.DataGeneration.randomFunCLBMDataGeneration
import Common.NormalInverseWishart
import Common.TSSInterface.{getSeriesPeriodogramsAndPcaCoefs, toTSS}
import Common.Tools._
import Common.ToolsRDD.RDDToMatrix
import Common.functionPrototypes._
import breeze.linalg.{DenseMatrix, DenseVector, argmax, max}
import breeze.stats.distributions.Gamma
import com.github.unsupervise.spark.tss.core.TSS
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import smile.validation.adjustedRandIndex

import scala.util.Random

object Main {

  implicit val ss: SparkSession = SparkSession
    .builder()
    .master("local[*]")
    .appName("AnalysePlan")
    .config("spark.executor.cores", 2)
    //.config("spark.executor.memory", "30G")
    .config("spark.executor.heartbeatInterval", "20s")
    .config("spark.driver.memory", "10G")
    .getOrCreate()

  ss.sparkContext.setLogLevel("WARN")
  ss.sparkContext.setCheckpointDir("checkpointDir")

  def main(args: Array[String]) {

    def generateDataset(pU: Double, pM: Double): DenseMatrix[DenseVector[Double]]= {

      val sigma = 0.01
      val groundTruthPartition = List(30, 30, 30, 30, 30)
      val nVariableBasis = 30

      val groundTruthPrototypes: List[List[List[Double] => List[Double]]] =
        List(List(f1, f2, f3, f4, f5), List(f5, f6, f7, f8, f9), List(f1, f3, f5, f7, f9))

      val sizeRowPartitions = List.fill(3)(groundTruthPartition)
      val sizeColPartition = List.fill(3)(10)

      val misleadingPrototypes: List[List[List[Double] => List[Double]]] =
        List(List(f1, f3, f5, f7), List(f2, f4, f6, f8),List(f9, f8, f2, f1))

      val uninformativePrototypes: List[List[List[Double] => List[Double]]] = List(List(f5))

      var auxPrototypes = groundTruthPrototypes
      var auxSizeRowPartitions = sizeRowPartitions
      var auxSizeColPartition = sizeColPartition

      if(pM > 0) {
        val misleadingPartition = Random.shuffle(List(50, 20, 60, 20))
        auxPrototypes = auxPrototypes ++ (0 until 3).toList.map(j => {
          misleadingPrototypes(j % 3)
        })
        auxSizeRowPartitions = auxSizeRowPartitions ++ List.fill(3)(misleadingPartition)
        auxSizeColPartition = auxSizeColPartition ++ List.fill(3)((pM * 10).toInt)
      }

      if(pU > 0) {
        val uninformativePartition = Random.shuffle(List(150))
        auxPrototypes = auxPrototypes ++ (0 until 1).toList.map(_ => {
          uninformativePrototypes.head
        })
        auxSizeRowPartitions = auxSizeRowPartitions ++ List.fill(1)(uninformativePartition)
        auxSizeColPartition = auxSizeColPartition ++ List.fill(1)((pU * nVariableBasis).toInt)
      }

      val row = randomFunCLBMDataGeneration(auxPrototypes, sigma, auxSizeRowPartitions, auxSizeColPartition)
      val dataRDD: RDD[(Int, Int, List[Double], List[Double])] = ss.sparkContext.parallelize(row)
      val tss: TSS = toTSS(dataRDD)

      val (_, _, pcaCoefsByRow) = getSeriesPeriodogramsAndPcaCoefs(tss,
        20, 3, 0.9)

      RDDToMatrix(pcaCoefsByRow)

    }

    def bestAmongSeveralRun(dataList: Array[Array[DenseVector[Double]]],
                            alphaPrior: Gamma, betaPrior: Gamma, gammaPrior: Gamma,
                            prior: Option[NormalInverseWishart],
                            nIter: Int = 5,
                            nRun: Int = 3,
                            verbose: Boolean = false,
                            withLikeliHood: Boolean = false) = {

      val listRest = (0 until nRun).map(_ => {
        new MCC.CollapsedGibbsSampler(
          dataList, alphaPrior, betaPrior, gammaPrior, initByUserPrior = prior).run(nIter, verbose = verbose)
      })
      val allLikelihoods: DenseVector[Double] = DenseVector(listRest.map(_._4).toArray)
      if(verbose){println("best LogLikelihood: " + max(allLikelihoods).toString)}
      listRest(argmax(allLikelihoods))

    }

    val pU = 1D
    val pM = 0D
    val dataMatrix = generateDataset(pU, pM)

    val shape = 5
    val scale = 10

    val alphaPrior = Gamma(shape = shape, scale = scale) // lignes
    val betaPrior  = Gamma(shape = shape, scale = scale) // clusters redondants
    val gammaPrior = Gamma(shape = 1E-7, scale = 2E-7) // clusters corrélés

    val nVarBasis = 30
    val trueRowPartition = (0 until 5).map(k => Array.fill(30)(k)).reduce(_ ++ _)

    val proportionUninformative = pU
    val proportionMisleading = pM
    val trueColPartition = Array.fill(nVarBasis)(0) ++
      Array.fill((proportionUninformative * nVarBasis).toInt)(1) ++
      Array.fill((proportionMisleading * nVarBasis).toInt)(2)

    var t0 = System.nanoTime()

    require(trueRowPartition.length == dataMatrix.rows)

    val dataList = matrixToDataByCol(dataMatrix)

    val G0 = new NormalInverseWishart(dataList)

    val (ariMCC, ariColMCC, riMCC, nmiMCC, nClusterMCC) = {

      var t1 = System.nanoTime()

      val (redundantColPartition, _, rowPartitions, _) = bestAmongSeveralRun(dataList,
        alphaPrior, betaPrior, gammaPrior, Some(G0), verbose = true)

      val bestAriIdxTSMC = rowPartitions.map(rowPartition => {
        adjustedRandIndex(rowPartition, trueRowPartition)
      }).zipWithIndex.maxBy(_._1)._2
      val closestRowPartition = rowPartitions(bestAriIdxTSMC)

      t1 = printTime(t1, "MCC")

      println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
      println("True Column Partition:")
      println(trueColPartition.mkString(", "))
      println("Estimated Column partition:")
      println(redundantColPartition.mkString(", "))

      getScores(closestRowPartition, trueRowPartition, redundantColPartition, trueColPartition)
    }

    println(ariMCC, ariColMCC, riMCC, nmiMCC, nClusterMCC)
    t0 = printTime(t0, "all methods run")

  }
}