package MCC

import Common.DataGeneration.randomFunCLBMDataGeneration
import Common.TSSInterface.{getSeriesPeriodogramsAndPcaCoefs, toTSS}
import Common.ToolsRDD.RDDToMatrix
import Common.functionPrototypes._
import com.github.unsupervise.spark.tss.core.TSS
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.util.Random

object generateTS {

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


    val sigma = 0.01
    val prototypeList: List[List[Double] => List[Double]] = List(f1, f2, f3, f4, f5, f6, f7, f8, f9)
    val groundTruthPartition = List(30, 30, 30, 30, 30)
    val nVariableBasis = 30

    val prototypes = List(Random.shuffle(prototypeList).slice(0, 5),
      Random.shuffle(prototypeList).slice(0, 5),
      Random.shuffle(prototypeList).slice(0, 5))

//    val prototypes: List[List[List[Double] => List[Double]]] = List(List(f8, f2, f4, f1, f3),
//      List(f1, f9, f2, f3, f7),
//      List(f4, f5, f6, f2, f8))

    val sizeRowPartitions = List.fill(3)(groundTruthPartition)
    val sizeColPartition = List.fill(3)(10)

    val propUninformativeAndMisleading = (List(0D, 0D, 0D, 0D, 0.5, 1, 2), List(0D, 0.5, 1, 2, 0D, 0D, 0D))

    val misleadingPrototypes: List[List[List[Double] => List[Double]]] = List(List(f1, f3, f5, f7),
      List(f2, f4, f6, f8),
      List(f9, f8, f2, f1))

    val uninformativePrototypes: List[List[List[Double] => List[Double]]] = List(List(f5, f3),
      List(f5, f3),
      List(f5, f3))

//      propUninformativeAndMisleading._1.indices
      (4 to 6).foreach(idx => {
        println("--------------------------------------------------")
        println(idx)
        println(sizeRowPartitions)
        println(sizeColPartition)

        var auxPrototypes = prototypes
        var auxSizeRowPartitions = sizeRowPartitions
        var auxSizeColPartition = sizeColPartition
        val pU = propUninformativeAndMisleading._1(idx)
        val pM = propUninformativeAndMisleading._2(idx)

        if(pM > 0) {
        val misleadingPartition = Random.shuffle(List(50, 20, 60, 20))
        auxPrototypes = auxPrototypes ++ (0 until 3).toList.map(j => {
          misleadingPrototypes(j % 3)
        })
        auxSizeRowPartitions = auxSizeRowPartitions ++ List.fill(3)(misleadingPartition)
        auxSizeColPartition = auxSizeColPartition ++ List.fill(3)((pM * 10).toInt)
      }

      if(pU > 0) {
        val uninformativePartition = Random.shuffle(List(70, 80))
        auxPrototypes = auxPrototypes ++ (0 until 3).toList.map(j => {
          uninformativePrototypes(j % 2)
        })
        auxSizeRowPartitions = auxSizeRowPartitions ++ List.fill(3)(uninformativePartition)
        auxSizeColPartition = auxSizeColPartition ++ List.fill(3)((pU * 10).toInt)
        println(auxPrototypes.map(_.length))
      }

      println(auxSizeRowPartitions)
      println(auxSizeColPartition)

      val row = randomFunCLBMDataGeneration(auxPrototypes, sigma, auxSizeRowPartitions, auxSizeColPartition)
      val dataRDD: RDD[(Int, Int, List[Double], List[Double])] = ss.sparkContext.parallelize(row)
      val tss: TSS = toTSS(dataRDD)

      val (_, _, pcaCoefsByRow) = getSeriesPeriodogramsAndPcaCoefs(tss, 10, 10, 0.9)

        val dataset = RDDToMatrix(pcaCoefsByRow)

//      if(pM > 0) {
//        dataset = addDuplicate(dataset, nVariableBasisNonDuplicate + nUNonDuplicate + nMNonDuplicate -1, (proportionDuplicate * nM).toInt)
//        datasetDPV = addDuplicate(datasetDPV, nVariableBasisNonDuplicate + nUNonDuplicate + nMNonDuplicate -1, (proportionDuplicate * nM).toInt)
//      }
//
//      if(pU > 0) {
//        dataset    = addDuplicate(dataset, nVariableBasisNonDuplicate + nUNonDuplicate  - 1, (proportionDuplicate * nU).toInt)
//        datasetDPV = addDuplicate(datasetDPV, nVariableBasisNonDuplicate + nUNonDuplicate  - 1, (proportionDuplicate * nU).toInt)
//      }
//
//      dataset    = addDuplicate(dataset, nVariableBasisNonDuplicate - 1, (proportionDuplicate * nVariableBasis).toInt )
//      datasetDPV = addDuplicate(datasetDPV, nVariableBasisNonDuplicate - 1, (proportionDuplicate * nVariableBasis).toInt )

      println(dataset.cols)
      println( (1 + pU + pM) * nVariableBasis)
      require(dataset.cols == (1 + pU + pM) * nVariableBasis)
//
      Common.IO.writeMatrixDvDoubleToCsv("playground/MCC/benchmarkSpringer/dataset_" + idx + ".csv", dataset)

    })





    //
    //
    //
    //
    //    (1 to 1).foreach(idxU => {
    //      (0 to 0).indices.foreach(idxM => {
    //
    //        var auxDataBasis = dataBasis
    //        var auxDataBasisDPV = dataBasisDPV
    //        var pU = propUninformative(idxU)
    //        val pM = propMisleading(idxM)
    //
    //        println(auxDataBasis.rows, auxDataBasis.cols)
    //
    //        if(pU > 0){
    //          val nU = pU * nVariableBasis
    //          println(nVariableBasis, nU)
    //
    //          val nUNonDuplicate = (nU*(1D-proportionDuplicate)).toInt
    //          val nUDuplicate = (proportionDuplicate * nU).toInt
    //
    //          val prototypes: List[List[List[Double] => List[Double]]] = (0 until (nUNonDuplicate+1)).toList.map(j => {
    //            Random.shuffle(prototypeList).slice(0,1)}).toList
    //          val sizeRowPartition = List.fill(nUNonDuplicate+1)(List(90))
    //          val sizeColPartition = List.fill(nUNonDuplicate+1)(1)
    //          val row = randomFunCLBMDataGeneration(prototypes, sigma, sizeRowPartition, sizeColPartition, shuffle = false)
    //          val dataRDD: RDD[(Int, Int, List[Double], List[Double])] = ss.sparkContext.parallelize(row)
    //          val tss: TSS = toTSS(dataRDD)
    //
    //          val (_, _, pcaCoefsByRow) = getSeriesPeriodogramsAndPcaCoefs(tss,
    //            10, 10, 0.9)
    //          val (_, _, pcaCoefsByRowDPV) = getSeriesPeriodogramsAndPcaCoefsPerVariable(tss,
    //            10, 10, 0.9)
    //
    //          val datasetU = addDuplicate(RDDToMatrix(pcaCoefsByRow), nUNonDuplicate, nUDuplicate)
    //          val datasetUDPV = addDuplicate(RDDToMatrix(pcaCoefsByRowDPV), nUNonDuplicate, nUDuplicate)
    //
    //          auxDataBasis = DenseMatrix.vertcat(auxDataBasis.t, datasetU.t).t
    //          auxDataBasisDPV = DenseMatrix.vertcat(auxDataBasisDPV.t, datasetUDPV.t).t
    //        }
    //
    //
    //        if(pM > 0){
    //          val nM = pM * nVariableBasis
    //          val nMNonDuplicate = (nM*(1D-proportionDuplicate)).toInt
    //          val nMDuplicate = (proportionDuplicate * nM).toInt
    //
    //          val prototypes: List[List[List[Double] => List[Double]]] = (0 until (nMNonDuplicate+1)).toList.map(j => {
    //            Random.shuffle(prototypeList).slice(0,4)}).toList
    //          val sizeRowPartition = List.fill(nMNonDuplicate+1)(Random.shuffle(List(20, 30, 20, 20)))
    //          val sizeColPartition = List.fill(nMNonDuplicate+1)(1)
    //          val row = randomFunCLBMDataGeneration(prototypes, sigma, sizeRowPartition, sizeColPartition, shuffle = false)
    //          val dataRDD: RDD[(Int, Int, List[Double], List[Double])] = ss.sparkContext.parallelize(row)
    //          val tss: TSS = toTSS(dataRDD)
    //          val (_, _, pcaCoefsByRow) = getSeriesPeriodogramsAndPcaCoefs(tss,
    //            10, 10, 0.9)
    //          val (_, _, pcaCoefsByRowDPV) = getSeriesPeriodogramsAndPcaCoefsPerVariable(tss,
    //            10, 10, 0.9)
    //
    //          auxDataBasis = DenseMatrix.vertcat(auxDataBasis.t,
    //            addDuplicate(RDDToMatrix(pcaCoefsByRow), nMNonDuplicate, nMDuplicate).t).t
    //          auxDataBasisDPV = DenseMatrix.vertcat(auxDataBasisDPV.t,
    //            addDuplicate(RDDToMatrix(pcaCoefsByRowDPV), nMNonDuplicate, nMDuplicate).t).t
    //        }
    //
    ////        println(idxU, idxM, propUninformative(idxU), auxDataBasis.rows, auxDataBasis.rows)
    ////        Common.IO.writeMatrixDvDoubleToCsv("playground/benchmark/dataset_" + idxU + "_" + idxM + ".csv", auxDataBasis)
    ////        Common.IO.writeMatrixDvDoubleToCsv("playground/benchmark/dataset_" + idxU + "_" + idxM + "DPV.csv", auxDataBasisDPV)
    //
    //      })
    //    })

  }
}
