package Common

import java.io.File

import Common.Tools._
import Common.ToolsRDD._
import breeze.linalg.{diag, DenseMatrix => BzDenseMatrix, DenseVector => BzDenseVector}
import breeze.stats.distributions.MultivariateGaussian
import com.github.unsupervise.spark.tss.core.TSS
import com.github.unsupervise.spark.tss.{functions => tssFunctions}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.{DenseMatrix, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, SparkSession, functions}

object TSSInterface  {

  def getSeriesPeriodogramsAndPcaCoefs(tss: TSS, sizePeriodogram: Int = 10,
                                       nPcaAxisMax: Int = 5,
                                       thresholdVarExplained: Double = 0.99)(implicit ss: SparkSession): (RDD[(Int, Array[BzDenseVector[Double]])], RDD[(Int, Array[BzDenseVector[Double]])], RDD[(Int, Array[BzDenseVector[Double]])]) = {

    val periodogramAndSeries = getPeriodogramsAndSeries(tss, sizePeriodogram)
    val periodograms = periodogramAndSeries.map(row => (row._1, row._2, row._3))
    val series = periodogramAndSeries.map(row => (row._1, row._2, BzDenseVector(row._4.toArray)))
    val periodogramByRow = fromCellDistributionToRowDistribution(periodogramAndSeries.map(row => (row._1, row._2, BzDenseVector(row._3.toArray))))
    val seriesByRow = fromCellDistributionToRowDistribution(series)

    val p = periodogramByRow.take(1).head._2.length
    val dummyPartition = List.fill(p)(0)
    val partitionPerColBc: Broadcast[BzDenseVector[Int]] = ss.sparkContext.broadcast(BzDenseVector(dummyPartition: _*))

    val (meanList, covMat, _, _) = getMeansAndCovariances(periodogramByRow.map(r => (r._1, r._2, List(0))),
      colPartition = partitionPerColBc,
      KVec = List(1),
      fullCovariance = true)

    val (loadings,_) = getLoadingsAndVarianceExplained(covMat.flatten.head, nPcaAxisMax, thresholdVarExplained)
    val pcaCoefs = periodograms.map(r => (r._1, r._2,
      loadings * (BzDenseVector(r._3.toArray) - meanList.flatten.head)))

    val pcaCoefByRow = fromCellDistributionToRowDistribution(pcaCoefs)

    (seriesByRow, periodogramByRow, pcaCoefByRow)
  }

  def getSeriesPeriodogramsAndPcaCoefsPerVariable(tss: TSS, sizePeriodogram: Int = 10,
                                                  nPcaAxisMax: Int = 5,
                                                  thresholdVarExplained: Double = 0.99)(implicit ss: SparkSession): (RDD[(Int, Array[BzDenseVector[Double]])], RDD[(Int, Array[BzDenseVector[Double]])], RDD[(Int, Array[BzDenseVector[Double]])]) = {

    val periodogramAndSeries = getPeriodogramsAndSeries(tss, sizePeriodogram)
    val series = periodogramAndSeries.map(row => (row._1, row._2, BzDenseVector(row._4.toArray)))
    val periodogramByRow = fromCellDistributionToRowDistribution(periodogramAndSeries.map(row => (row._1, row._2, BzDenseVector(row._3.toArray))))
    val seriesByRow = fromCellDistributionToRowDistribution(series)

    val p = periodogramByRow.take(1).head._2.length
    val colPartition = (0 until p).toList
    val partitionPerColBc: Broadcast[BzDenseVector[Int]] = ss.sparkContext.broadcast(BzDenseVector(colPartition: _*))

    val periodogramByRowWithRowPartitions = periodogramByRow.map(r => (r._1, r._2, List.fill(p)(0)))

    val (meanList, covMat, _, _) = getMeansAndCovariances(periodogramByRowWithRowPartitions,
      colPartition = partitionPerColBc,
      KVec = List.fill(p)(1),
      fullCovariance = true)

    val newLoadings =
      (0 until p).map(l => {
        List.fill(p)(0).map({k =>
          getLoadingsAndVarianceExplained(covMat(l)(k), nPcaAxisMax, thresholdVarExplained)._1
        })
      }).toList

    val pcaCoefByRow = periodogramByRowWithRowPartitions.map(r =>
      (r._1, r._2.indices.map(j => {
        newLoadings(colPartition(j))(r._3(colPartition(j))) * (r._2(j) - meanList(colPartition(j))(r._3(colPartition(j))))
      }).toArray))

    (seriesByRow, periodogramByRow, pcaCoefByRow)

  }

  def getSeriesPeriodogramsAndPcaCoefsPerVariableWithVarExplained(tss: TSS, sizePeriodogram: Int = 10,
                                                                  nPcaAxisMax: Int = 5,
                                                                  thresholdVarExplained: Double = 0.99)(implicit ss: SparkSession): (RDD[(Int, Array[BzDenseVector[Double]])], RDD[(Int, Array[BzDenseVector[Double]])], RDD[(Int, Array[BzDenseVector[Double]])], Double) = {

    val periodogramAndSeries = getPeriodogramsAndSeries(tss, sizePeriodogram)
    val series = periodogramAndSeries.map(row => (row._1, row._2, BzDenseVector(row._4.toArray)))
    val periodogramByRow = fromCellDistributionToRowDistribution(periodogramAndSeries.map(row => (row._1, row._2, BzDenseVector(row._3.toArray))))
    val seriesByRow = fromCellDistributionToRowDistribution(series)

    val p = periodogramByRow.take(1).head._2.length
    val colPartition = (0 until p).toList
    val partitionPerColBc: Broadcast[BzDenseVector[Int]] = ss.sparkContext.broadcast(BzDenseVector(colPartition: _*))

    val periodogramByRowWithRowPartitions = periodogramByRow.map(r => (r._1, r._2, List.fill(p)(0)))

    val (meanList, covMat, _, _) = getMeansAndCovariances(periodogramByRowWithRowPartitions,
      colPartition = partitionPerColBc,
      KVec = List.fill(p)(1),
      fullCovariance = true)

    val loadingsAndVariance = (0 until p).map(j => {
      val loadindsAndVariancePerAxis = getLoadingsAndVarianceExplained(covMat(j).head, nPcaAxisMax, thresholdVarExplained)
      (loadindsAndVariancePerAxis._1, loadindsAndVariancePerAxis._2.sum)
    }).toList

    val loadings = loadingsAndVariance.map(_._1)
    val variances = loadingsAndVariance.map(_._2)

    val pcaCoefByRow = periodogramByRowWithRowPartitions.map(r =>
      (r._1, r._2.indices.map(j => {
        loadings(j) * (r._2(j) - meanList(j).head)
      }).toArray))

    (seriesByRow, periodogramByRow, pcaCoefByRow, variances.sum / variances.length.toDouble)

  }

  def toTSS(data: RDD[(Int, Int, List[Double], List[Double])])(implicit ss: SparkSession): TSS = {

    val rddNewEncoding = data.map(row =>
      (row._1.toString,
        row._2.toString,
        row._3.head,
        row._3.last,
        row._3(1)- row._3.head,
        row._4))

    val dfWithSchema = ss.createDataFrame(rddNewEncoding)
      .toDF("scenario_id", "varName", "timeFrom", "timeTo","timeGranularity", "series")

    val dfWithDecorator = dfWithSchema.select(
      map(lit("scenario_id"),
        col("scenario_id"),
        lit("varName"),
        col("varName")).alias("decorators"),
      col("timeFrom"),
      col("timeTo"),
      col("timeGranularity"),
      col("series").alias("series")
    )

    TSS(dfWithDecorator)
  }

  def addPCA(series: sql.DataFrame,
             outColName: String,
             inColName: String,
             maxK: Int,
             significancyThreshold: Double,
             pcaLoadingsOutFile: Option[File] = None,
             pcaVariancesOutFile: Option[File] = None,
             pcaCos2OutFile: Option[File] = None,
             pcaContribVariablesOutFile: Option[File] = None):(TSS, DenseMatrix) = {
    val pca = new PCA()
      .setInputCol(inColName)
      .setOutputCol(outColName)
      .setK(maxK)
      .fit(series)
    val significantPCADimensions =
      pca.explainedVariance.values.indices.find(i => {
        pca.explainedVariance.values.slice(0, i + 1).sum >= significancyThreshold
      }).getOrElse(maxK - 1) + 1
    val pcaRes = new PCA()
      .setInputCol(inColName)
      .setOutputCol(outColName)
      .setK(significantPCADimensions)
      .fit(series)

    val eigenValue = diag(BzDenseVector(pcaRes.explainedVariance.toArray))
    val pcDM = matrixToDenseMatrix(pcaRes.pc)
    (new TSS(pcaRes.transform(series)), denseMatrixToMatrix(pcDM*eigenValue).toDense)
  }

  def getPcaCoefs(tss:TSS)(implicit ss: SparkSession): RDD[Row] = {

    val withFourierTable =
    //Create a Z-normalized version of the original "series" column, and store it in "zseries" column (the last param is the zero precision)
      tss.addZNormalized("zseries", TSS.SERIES_COLNAME, 0.0001)
        //Then, add a "dft" column containing the Discrete Fourier Transform of the zseries created previously
        .addDFT("dft", "zseries")
        //Also add the Fourier frequencies and store them into "dftFreq" column
        .addDFTFrequencies("dftFreq", TSS.SERIES_COLNAME, TSS.TIMEGRANULARITY_COLNAME)
        //Finally create the DFT periodogram from the DFT
        .addDFTPeriodogram("dftPeriodogram", "dft")

    //Now we need some aggregations performed in order to continue with the pipeline, thus the intermediate withFourierTable object
    //Compute the average time gap between the two first consecutive time steps of each time series in the TSS object
    val meanFourierFrequencyStep = withFourierTable.colSeqFirstStep("dftFreq")
      .agg(functions.mean("value"))
      .first.getDouble(0)

    //Compute linear interpolation of periodograms on wanted number of first points
    //over a meanListDV scenario frequency step basis. Remove 0 bin Fourier coefficient to avoid 0 columns in matrices afterwards
    //Restrict to smallest interpolation points space if some series have frequencies outside the original bounds
    //=> Work on the intersection of frequency spaces
    val newInterpolationSamplePoints = (0 until 50).map(_.toDouble * meanFourierFrequencyStep)
    val minMaxAndMaxMinFourierFrequency = withFourierTable.series.select(min(array_max(col("dftFreq"))), max(array_min(col("dftFreq")))).first
    val minMaxFourierFrequency = minMaxAndMaxMinFourierFrequency.getDouble(0)
    val maxMinFourierFrequency = minMaxAndMaxMinFourierFrequency.getDouble(1)
    val keptInterpolationSamplePoints: Array[Double] = newInterpolationSamplePoints.filter(x => x < minMaxFourierFrequency && x > maxMinFourierFrequency).toArray
    //Now that the aligned fourier frequencies are computed, restore the vectorization pipeline:
    // - add a constant column made of the aligned fourier frequencies vector
    val interpolatedFourierTSS = withFourierTable
      .addConstant("interpolatedDFTFreq", keptInterpolationSamplePoints)
      // - add a column containing the linear interpolation of the DFT periodogram for the aligned Fourier frequencies
      .addLinearInterpolationPoints("interpolatedDFTPeriodogram", "dftFreq", "dftPeriodogram", keptInterpolationSamplePoints)
    //As a final step, compute the log10 values of each element of the interpolated DFT periodograms, for each time series
    val logScaledTSS = interpolatedFourierTSS
      .addUDFColumn("logInterpolatedDFTPeriodogram", "interpolatedDFTPeriodogram",
        functions.udf(tssFunctions.scale(0.001)
          .andThen(tssFunctions.log10(1D))
          .andThen((seq: Seq[Double]) => { Vectors.dense(seq.toArray) })))
      .repartition(100)

    //Scale the vectorization by columns before PCA computation
    val scaledTSS = logScaledTSS.addColScaled("logInterpolatedDFTPeriodogram_ScaledVecColScaled", "logInterpolatedDFTPeriodogram", scale = true, center = true)
    //Compute PCA from the per column scaled vectorization and store the reduced representation into "logInterpolatedDFTPeriodogram_PCAVec"
    //Use maximum 20 reduced coordinates or less if the explained variance reaches 0.99
    //Store the byproducts (such as loadings and variance explained ratios) into files
    val joinedTSS = scaledTSS.addPCA("logInterpolatedDFTPeriodogram_PCAVec", "logInterpolatedDFTPeriodogram_ScaledVecColScaled", 40, 0.9999,
      None,None)
      //Scale PCA results to enforce the correct relative importance of each feature to the afterwards weighting
      .addColScaled("logInterpolatedDFTPeriodogram_ColScaledPCAVec", "logInterpolatedDFTPeriodogram_PCAVec", scale = true, center = true)
      .addSeqFromMLVector("pcaCoordinatesV", "logInterpolatedDFTPeriodogram_ColScaledPCAVec")
      //Drop intermediate columns for cleaner output
      .drop("logInterpolatedDFTPeriodogram_ColScaledPCAVec", "logInterpolatedDFTPeriodogram_PCAVec", "logInterpolatedDFTPeriodogram_ScaledVecColScaled")

    joinedTSS.select("varName","scenario_id","pcaCoordinatesV", "series").series
      .select(
        col("element_at(decorators, scenario_id)").alias("scenario_id"),
        col("element_at(decorators, varName)").alias("varName"),
        col("pcaCoordinatesV"),
        col("series"))
      .rdd
  }

  def getPeriodogramsAndSeries(tss: TSS, sizePeriodogram:Int= 10)(implicit ss: SparkSession): RDD[(Int, Int, List[Double], List[Double])] = {

    val withFourierTable: TSS =
      tss.addZNormalized("zseries", TSS.SERIES_COLNAME, 0.0001)
        .addDFT("dft", "zseries")
        .addDFTFrequencies("dftFreq", TSS.SERIES_COLNAME, TSS.TIMEGRANULARITY_COLNAME)
        .addDFTPeriodogram("dftPeriodogram", "dft")

    val meanFourierFrequencyStep = withFourierTable
      .colSeqFirstStep("dftFreq")
      .agg(functions.mean("value"))
      .first.getDouble(0)

    val newInterpolationSamplePoints = (0 until sizePeriodogram).map(_.toDouble * meanFourierFrequencyStep)
    val minMaxAndMaxMinFourierFrequency = withFourierTable.series.select(min(array_max(col("dftFreq"))), max(array_min(col("dftFreq")))).first
    val minMaxFourierFrequency = minMaxAndMaxMinFourierFrequency.getDouble(0)
    val maxMinFourierFrequency = minMaxAndMaxMinFourierFrequency.getDouble(1)
    val keptInterpolationSamplePoints: Array[Double] = newInterpolationSamplePoints.filter(x => x < minMaxFourierFrequency && x > maxMinFourierFrequency).toArray

    val interpolatedFourierTSS = withFourierTable
      .addConstant("interpolatedDFTFreq", keptInterpolationSamplePoints)
      .addLinearInterpolationPoints("interpolatedDFTPeriodogram", "dftFreq", "dftPeriodogram", keptInterpolationSamplePoints)

    val logScaledTSS = interpolatedFourierTSS.addUDFColumn("logInterpolatedDFTPeriodogram",
      "interpolatedDFTPeriodogram",
      functions.udf(tssFunctions.log10(1D)
        .andThen((seq: Seq[Double]) => {Vectors.dense(seq.toArray)})))
      .repartition(200)
    val scaledTSS: TSS = logScaledTSS.addColScaled("logInterpolatedDFTPeriodogram_ScaledVecColScaled",
      "logInterpolatedDFTPeriodogram",scale = true,center = true)
    val seqScaledTSS = scaledTSS.addSeqFromMLVector("periodogram",
      "logInterpolatedDFTPeriodogram_ScaledVecColScaled")
    val series = seqScaledTSS.series

    val outputDf = series.select(
      seqScaledTSS.getDecoratorColumn("scenario_id").alias("scenario_id"),
      seqScaledTSS.getDecoratorColumn("varName").alias("varName"),
      col("periodogram"),
      col("series")).rdd

    outputDf.map(row => (
      row.getString(0).toInt,
      row.getString(1).toInt,
      row.getSeq[Double](2).toArray.toList,
      row.getSeq[Double](3).toArray.toList)
    )
  }

  def getPcaAndLoadings(dataRDD: RDD[(Int, Int, List[Double])], maxK:Int = 20)(implicit ss: SparkSession):
  (RDD[(Int, Int, BzDenseVector[Double])], BzDenseMatrix[Double]) = {

    val dfWithSchema = ss.createDataFrame(dataRDD).toDF("scenario_id", "varName", "periodogram")

    val tss = new TSS(dfWithSchema, forceIds = false)

    val tssVec = tss.addMLVectorized("periodogramMLVec", "periodogram")

    val (joinedTSS, loadings): (TSS, org.apache.spark.ml.linalg.DenseMatrix) =
      addPCA(tssVec.series,"logInterpolatedDFTPeriodogram_PCAVec",
        "periodogramMLVec",
        maxK = maxK,0.99, None,None)

    //Scale PCA results to enforce the correct relative importance of each feature to the afterwards weighting
    val scaledJoinedTSS = joinedTSS.addColScaled("logInterpolatedDFTPeriodogram_ColScaledPCAVec",
      "logInterpolatedDFTPeriodogram_PCAVec", scale = true, center = true)
      .addSeqFromMLVector("pcaCoordinatesV", "logInterpolatedDFTPeriodogram_ColScaledPCAVec")
      //Drop intermediate columns for cleaner output
      .drop("logInterpolatedDFTPeriodogram_ColScaledPCAVec", "logInterpolatedDFTPeriodogram_PCAVec", "periodogram")

    val series = scaledJoinedTSS.select("scenario_id","varName","pcaCoordinatesV").series
    val pcaCoefs = series.select(col("scenario_id"),
      col("varName"),
      col("pcaCoordinatesV")).rdd

    val pcaCoefsRDDs = pcaCoefs.map(row => (
      row.getInt(0),
      row.getInt(1),
      BzDenseVector[Double](row.getSeq[Double](2).toArray))
    )

    (pcaCoefsRDDs, matrixToDenseMatrix(loadings).t)

  }

}
