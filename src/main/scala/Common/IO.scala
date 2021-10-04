package Common

import java.io.{BufferedWriter, FileWriter}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.stats.distributions.MultivariateGaussian
import com.opencsv.CSVWriter
import org.json4s._
import org.json4s.native.Serialization
import org.json4s.native.Serialization._

import scala.collection.JavaConverters._
import scala.io.Source
import scala.util.{Failure, Try}

object IO {

  implicit val formats: AnyRef with Formats = Serialization.formats(NoTypeHints)

  def readDataSet(path: String): List[List[Double]] = {
    val lines = Source.fromFile(path).getLines.toList.drop(1)
    lines.indices.map(seg => {
      lines(seg).drop(1).dropRight(1).split(";").toList.map(string => string.split(":")(1).toDouble)
    }).toList
  }

  def readDenseMatrixDvDouble(path: String): DenseMatrix[DenseVector[Double]] = {
    val lines = Source.fromFile(path).getLines.toList.drop(1)
    DenseMatrix(lines.map(line =>  {
      val elementList = line.drop(1).dropRight(1).split("\",\"").toList
      elementList.map(string => DenseVector(string.split(":").map(_.toDouble)))
    }):_*)
  }

  def addIndex(content: List[List[String]]): List[List[String]] =
    content.foldLeft((1, List.empty[List[String]])){
      case ((serial: Int, acc: List[List[String]]), value: List[String]) =>
        (serial + 1, (serial.toString +: value) +: acc)
    }._2.reverse

  def writeMatrixStringToCsv(fileName: String, Matrix: DenseMatrix[String], append: Boolean = false): Unit = {
    val rows : List[List[String]] = Matrix(*, ::).map(dv => dv.toArray.toList).toArray.toList
    writeCsvFile(fileName, addIndex(rows), append=append)
  }

  def writeMatrixDoubleToCsv(fileName: String, Matrix: DenseMatrix[Double], withHeader:Boolean=true): Unit = {
    val header: List[String] = List("id") ++ (0 until Matrix.cols).map(_.toString).toList
    val rows : List[List[String]] = Matrix(*, ::).map(dv => dv.toArray.map(_.toString).toList).toArray.toList
    if(withHeader){
      writeCsvFile(fileName, addIndex(rows), header)
    } else {
      writeCsvFile(fileName, addIndex(rows))
    }
  }

  def writeMatrixDvDoubleToCsv(fileName: String, Matrix: DenseMatrix[DenseVector[Double]], withHeader:Boolean=true): Unit = {
    val header: List[String] = (0 until Matrix.cols).map(_.toString).toList

    val rows : List[List[String]] = Matrix(*, ::).map(dv => dv.toArray.map(_.toArray.mkString(":")).toList).toArray.toList
    if(withHeader){
      writeCsvFile(fileName, rows, header)
    } else {
      writeCsvFile(fileName, rows)
    }
  }

  def writeMatrixIntToCsv(fileName: String, Matrix: DenseMatrix[Int], withHeader:Boolean=true): Unit = {
    val header: List[String] = List("id") ++ (0 until Matrix.cols).map(_.toString).toList
    val rows : List[List[String]] = Matrix(*, ::).map(dv => dv.toArray.map(_.toString).toList).toArray.toList
    if(withHeader){
      writeCsvFile(fileName, addIndex(rows), header)
    } else {
      writeCsvFile(fileName, addIndex(rows))
    }
  }

  def writeCsvFile(fileName: String,
                   rows: List[List[String]],
                   header: List[String] = List.empty[String],
                   append:Boolean=false
                  ): Try[Unit] =
  {
    val content = if(header.isEmpty){rows} else {header +: rows}
    Try(new CSVWriter(new BufferedWriter(new FileWriter(fileName, append)))).flatMap((csvWriter: CSVWriter) =>
      Try{
        csvWriter.writeAll(
          content.map(_.toArray).asJava
        )
        csvWriter.close()
      } match {
        case f @ Failure(_) =>
          // Always return the original failure.  In production code we might
          // define a new exception which wraps both exceptions in the case
          // they both fail, but that is omitted here.
          Try(csvWriter.close()).recoverWith{
            case _ => f
          }
        case success =>
          success
      }
    )
  }

  def writeGaussianComponentsParameters(pathOutput: String, components: List[List[MultivariateGaussian]]): Unit = {
    val outputContent = components.map(gaussianList => {
      gaussianList.map(G =>
        List(
          G.mean.toArray.mkString(":"),
          G.covariance.toArray.mkString(":"))).reduce(_++_)
    })
    Common.IO.writeCsvFile(pathOutput, Common.IO.addIndex(outputContent))
  }

  def writeSimpleMixtureFinalClustering(pathOutputBase: String,
                                        data: List[DenseVector[Double]],
                                        partition: List[List[Int]],
                                        components: List[List[MultivariateGaussian]]): Unit = {
    val finalClustering = partition.transpose.map(m => m.groupBy(identity).mapValues(_.size).toList.maxBy(_._2)._1)
    val finalComponent = List(components.last)
    Common.IO.writeMatrixDoubleToCsv(pathOutputBase + "/dataLine.csv", DenseMatrix(data:_*), withHeader = false)
    Common.IO.writeMatrixIntToCsv(pathOutputBase + "/clustering.csv", DenseMatrix(finalClustering:_*), withHeader = false)
    Common.IO.writeGaussianComponentsParameters(pathOutputBase + "/components.csv", finalComponent)
  }

  def writeSimpleMixtureEveryStep(pathOutputBase: String,
                                  data: List[DenseVector[Double]],
                                  partition: List[List[Int]],
                                  components: List[List[MultivariateGaussian]]): Unit = {
    Common.IO.writeMatrixDoubleToCsv(pathOutputBase + "/data.csv", DenseMatrix(data:_*), withHeader = false)
    Common.IO.writeMatrixIntToCsv(pathOutputBase + "/clusteringList.csv", DenseMatrix(partition:_*), withHeader = false)
    Common.IO.writeGaussianComponentsParameters(pathOutputBase + "/componentsList.csv", components)
  }

  def writeNPLBMFinalClustering(pathOutputBase: String,
                                data: DenseMatrix[DenseVector[Double]],
                                rowPartition: List[List[Int]],
                                colPartition: List[List[Int]],
                                components: List[List[List[MultivariateGaussian]]]): Unit = {
    val finalRowClustering = rowPartition.transpose.map(m => m.groupBy(identity).mapValues(_.size).toList.maxBy(_._2)._1)
    val finalColClustering = colPartition.transpose.map(m => m.groupBy(identity).mapValues(_.size).toList.maxBy(_._2)._1)
    val finalComponent = components.last
    Common.IO.writeLBMDataAndCluster(pathOutputBase + "/dataAndCluster.csv", data, finalRowClustering, finalColClustering)
    Common.IO.writeLBMParameters(pathOutputBase + "/model.csv", finalComponent)
  }

  def writeFunNPLBMFinalClustering(pathOutputBase: String,
                                   data: DenseMatrix[DenseVector[Double]],
                                   series: DenseMatrix[DenseVector[Double]],
                                   finalRowPartition: List[Int],
                                   finalColPartition: List[Int],
                                   finalComponents: List[List[MultivariateGaussian]],
                                   rowDict: Option[List[String]] = None,
                                   colDict: Option[List[String]] = None): Unit = {

    Common.IO.writeMatrixIntToCsv(pathOutputBase + "/finalRowClustering.csv", DenseMatrix(finalRowPartition:_*), withHeader = false)
    Common.IO.writeMatrixIntToCsv(pathOutputBase + "/finalColumnClustering.csv", DenseMatrix(finalColPartition:_*), withHeader = false)
    Common.IO.writeFunLBMDataAndCluster(pathOutputBase + "/dataAndCluster.csv", data, series, finalRowPartition,
      finalColPartition, rowDict, colDict)
    Common.IO.writeFunLBMData(pathOutputBase + "/x.csv", data, series)
    Common.IO.writeLBMParameters(pathOutputBase + "/finalModel.csv", finalComponents)
  }


  def writeFunNPCLBMFinalClustering(pathOutputBase: String,
                                    data: DenseMatrix[DenseVector[Double]],
                                    series: DenseMatrix[DenseVector[Double]],
                                    finalRowClustering: List[List[Int]],
                                    finalColClustering: List[Int],
                                    finalComponent: List[List[MultivariateGaussian]],
                                    rowDict: Option[List[String]] = None,
                                    colDict: Option[List[String]] = None): Unit = {
    Common.IO.writeMatrixIntToCsv(pathOutputBase + "/finalRowClustering.csv", DenseMatrix(finalRowClustering:_*),
      withHeader = false)
    Common.IO.writeMatrixIntToCsv(pathOutputBase + "/finalColumnClustering.csv", DenseMatrix(finalColClustering:_*),
      withHeader = false)
    Common.IO.writeFunCLBMDataAndCluster(pathOutputBase + "/dataAndCluster.csv", data, series, finalRowClustering,
      finalColClustering, rowDict, colDict)
    Common.IO.writeFunLBMData(pathOutputBase + "/x.csv", data, series)
    Common.IO.writeCLBMParameters(pathOutputBase + "/finalModel.csv", finalComponent)
  }


  def writeFunCLBMDataAndCluster(pathOutput: String,
                                 data: DenseMatrix[DenseVector[Double]],
                                 series: DenseMatrix[DenseVector[Double]],
                                 rowPartition: List[List[Int]],
                                 colPartition: List[Int],
                                 rowDict: Option[List[String]] = None,
                                 colDict: Option[List[String]] = None): Try[Unit] = {

    val actualRowDict: List[String] = rowDict match {
      case Some(dict) => dict
      case None => rowPartition.head.indices.toList.map(_.toString)
    }

    val actualColDict: List[String] = colDict match {
      case Some(dict) => dict
      case None => colPartition.indices.toList.map(_.toString)
    }

    val outputContent = (0 until data.rows).map(i => {
      (0 until data.cols).map(j => {
        val l = colPartition(j)
        List(
          actualRowDict(i),
          actualColDict(j),
          rowPartition(l)(i).toString,
          colPartition(j).toString,
          data(i,j).toArray.mkString(":"),
          series(i,j).toArray.mkString(":")
        )
      }).toList
    }).reduce(_++_)

    Common.IO.writeCsvFile(pathOutput,outputContent)
  }

  def writeFunNPLBMEveryStep(pathOutputBase: String,
                             data: DenseMatrix[DenseVector[Double]],
                             series: DenseMatrix[DenseVector[Double]],
                             rowPartition: List[List[Int]],
                             colPartition: List[List[Int]],
                             components: List[List[List[MultivariateGaussian]]]): Unit = {
    Common.IO.writeMatrixIntToCsv(pathOutputBase + "/rowClusteringList.csv", DenseMatrix(rowPartition:_*), withHeader =  false)
    Common.IO.writeMatrixIntToCsv(pathOutputBase + "/columnClusteringList.csv", DenseMatrix(colPartition:_*), withHeader =  false)
    Common.IO.writeFunLBMData(pathOutputBase + "/coefsAndSeries.csv", data, series)
    Common.IO.writeLBMParametersEverySteps(pathOutputBase + "/modelsEverySteps.json", components)
  }

  def writeFunLBMDataAndCluster(pathOutput: String,
                                data: DenseMatrix[DenseVector[Double]],
                                series: DenseMatrix[DenseVector[Double]],
                                rowPartition: List[Int],
                                colPartition: List[Int],
                                rowDict: Option[List[String]] = None,
                                colDict: Option[List[String]] = None): Try[Unit] = {

    val actualRowDict: List[String] = rowDict match {
      case Some(dict) => dict
      case None => rowPartition.indices.toList.map(_.toString)
    }

    val actualColDict: List[String] = colDict match {
      case Some(dict) => dict
      case None => colPartition.indices.toList.map(_.toString)
    }

    val outputContent = (0 until data.rows).map(i => {
      (0 until data.cols).map(j => {
        List(actualRowDict(i),
          actualColDict(j),
          rowPartition(i).toString,
          colPartition(j).toString,
          data(i,j).toArray.mkString(":"),
          series(i,j).toArray.mkString(":")
        )
      }).toList
    }).reduce(_++_)
    Common.IO.writeCsvFile(pathOutput, outputContent)
  }

  def writeFunLBMData(pathOutput: String,
                      data: DenseMatrix[DenseVector[Double]],
                      series: DenseMatrix[DenseVector[Double]]): Try[Unit] = {
    val outputContent = (0 until data.rows).map(i => {
      (0 until data.cols).map(j => {
        List(i.toString,
          j.toString,
          data(i,j).toArray.mkString(":"),
          series(i,j).toArray.mkString(":")
        )
      }).toList
    }).reduce(_++_)
    Common.IO.writeCsvFile(pathOutput, outputContent)
  }

  def writeDataFromList (pathOutput: String,
                         data: List[(Int, Int, List[Double], List[Double])]): Try[Unit] = {
    val outputContent = data.map(r => {
      List(
        r._1.toString,
        r._2.toString,
        r._3.mkString(":"),
        r._4.mkString(":"))
    })
    Common.IO.writeCsvFile(pathOutput, outputContent)
  }

  def readDataFromList(path: String): List[(Int, Int, List[Double], List[Double])] = {

    val lines = Source.fromFile(path).getLines.toList
    lines.map(line =>  {
      val elementList = line.drop(1).dropRight(1).split("\",\"").toList
      val id1 = elementList.head.toInt
      val id2 = elementList(1).toInt
      val indices = elementList(2).split(":").map(_.toDouble).toList
      val values = elementList(3).split(":").map(_.toDouble).toList
      (id1, id2, indices, values)
    })

  }

  def writeLBMDataAndCluster(pathOutput: String,
                             data: DenseMatrix[DenseVector[Double]],
                             rowPartition: List[Int],
                             colPartition: List[Int]): Try[Unit] = {
    val outputContent = (0 until data.rows).map(i => {
      (0 until data.cols).map(j => {
        List(i.toString,
          j.toString,
          rowPartition(i).toString,
          colPartition(j).toString,
          data(i,j).toArray.mkString(":"))
      }).toList
    }).reduce(_++_)
    Common.IO.writeCsvFile(pathOutput, outputContent)
  }

  def writeCLBMParameters(pathOutput: String, components: List[List[MultivariateGaussian]]): Try[Unit] = {

    val outputContent =
      components.indices.map(l => {
        components(l).indices.map(k_l => {
          List(k_l.toString,
            l.toString,
            components(l)(k_l).mean.toArray.mkString(":"),
            components(l)(k_l).covariance.toArray.mkString(":"))
        }).toList
      }).reduce(_++_)
    val header: List[String] = List("id","rowCluster","colCluster","meanListDV","covariance")
    writeCsvFile(pathOutput, addIndex(outputContent),header)
  }

  def writeLBMParameters(pathOutput: String, components: List[List[MultivariateGaussian]]): Try[Unit] = {
    val outputContent =
      components.indices.map(k => {
        components.head.indices.map(l => {
          List(k.toString,
            l.toString,
            components(k)(l).mean.toArray.mkString(":"),
            components(k)(l).covariance.toArray.mkString(":"))
        }).toList
      }).reduce(_++_)
    val header: List[String] = List("id","rowCluster","colCluster","meanListDV","covariance")
    writeCsvFile(pathOutput, addIndex(outputContent),header)
  }

  def LBMParameterToStringArray(LBMParams: List[List[MultivariateGaussian]]): List[String] = {
    val means = LBMParams.indices.map(k =>{
      k.toString + ":" + LBMParams.head.indices.map(l => {
        LBMParams(k)(l).mean.toArray.mkString(",")
      }).toArray.mkString(";")
    }).toArray.mkString("#")
    val covariances = LBMParams.indices.map(k =>{
      k.toString + ":" + LBMParams.head.indices.map(l => {
        LBMParams(k)(l).covariance.toArray.mkString(",")
      }).toArray.mkString(";")
    }).toArray.mkString("#")
    List(means, covariances)
  }

  def writeLBMParametersEverySteps(pathOutput: String, componentsEveryIterations: List[List[List[MultivariateGaussian]]]): Unit = {
    val outputContent: Map[String, Any] = componentsEveryIterations.indices.map(iter => {
      (iter.toString,
        componentsEveryIterations(iter).indices.map(l => {
          (l.toString,
            componentsEveryIterations(iter)(l).indices.map(k => {
              (k.toString,
                Map("mean" -> componentsEveryIterations(iter)(l)(k).mean.toArray.mkString(":"),
                  "CovarianceMatrix" -> componentsEveryIterations(iter)(l)(k).covariance.toArray.mkString(":")))
            }).toMap)
        }).toMap)
    }).toMap

    Files.write(Paths.get(pathOutput), write(outputContent).getBytes(StandardCharsets.UTF_8))

  }

  def writeMCC(pathOutput: String,
               redundantColPartition: List[Int],
               correlatedColPartitions: List[List[Int]],
               rowPartitionPartitions: List[List[Int]],
               iterName: List[String],
               varName: List[String]): Unit = {

    val outputContent: Map[String, Any] = Map(
      "redundantColPartition" -> redundantColPartition,
      "NPLBMPartitions" -> correlatedColPartitions.indices.map(h =>
        h.toString -> Map("correlatedColPartition" -> correlatedColPartitions(h), "rowPartition" -> rowPartitionPartitions(h))).toMap,
      "varName" -> varName)

    Files.write(Paths.get(pathOutput), write(outputContent).getBytes(StandardCharsets.UTF_8))

  }

}
