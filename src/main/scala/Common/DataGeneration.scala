package Common

import Common.ProbabilisticTools._
import Common.TSSInterface.{getSeriesPeriodogramsAndPcaCoefs, getSeriesPeriodogramsAndPcaCoefsPerVariable, toTSS}
import Common.Tools.allEqual
import Common.ToolsRDD.RDDToMatrix
import breeze.linalg.{*, DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.{MultivariateGaussian, RandBasis}
import com.github.unsupervise.spark.tss.core.TSS
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.util.Random

object DataGeneration  {

  def randomMixture(modes: List[DenseVector[Double]],
                    covariances: List[DenseMatrix[Double]],
                    sizeCluster: List[Int],
                    shuffle: Boolean = false): List[DenseVector[Double]] = {

    require(modes.length == covariances.length, "modes and covariances lengths do not match")
    require(modes.length == sizeCluster.length, "sizeCluster and modes lengths do not match")
    val K = modes.length

    val data = (0 until K).map(k => {
      MultivariateGaussian(modes(k), covariances(k)).sample(sizeCluster(k))
    }).reduce(_++_).toList
    if(shuffle){Random.shuffle(data)} else data
  }

  def randomFunLBMDataGeneration(prototypes: List[List[List[Double]=> List[Double]]],
                                 sigma: Double,
                                 sizeClusterRow: List[Int],
                                 sizeClusterCol: List[Int],
                                 shuffle: Boolean = false,
                                 mixingProportion: Double = 0D,
                                 seed: Option[Int] = None):
  List[(Int, Int, List[Double], List[Double])] = {

    require(sizeClusterRow.length == prototypes.head.length)
    require(sizeClusterCol.length == prototypes.length)
    val K = prototypes.head.length
    val L = prototypes.length

    val length = 1000
    val indices:List[Double] = (1 to length).map(_/length.toDouble).toList

    val modes = (0 until L).map(l => {(0 until K).map(k => {DenseVector(prototypes(l)(k)(indices).toArray)}).toList}).toList
    val covariances = (0 until L).map(_ => {(0 until K).map(_ => {diag(DenseVector(Array.fill(indices.length){sigma}))}).toList}).toList

    val sizeClusterRowEachColumn = List.fill(L)(sizeClusterRow)
    val dataPerBlock = generateDataPerBlock(modes, covariances, sizeClusterRowEachColumn, sizeClusterCol, mixingProportion, seed)
    val data: DenseMatrix[DenseVector[Double]] = doubleReduce(dataPerBlock)
    val dataShuffled = applyConditionalShuffleData(data, shuffle)
    (0 until data.rows).map(i => {(0 until data.cols).map(j => {(i, j, indices, dataShuffled(i,j).toArray.toList)}).toList}).reduce(_++_)
  }

  def randomFunCLBMDataGeneration(prototypes: List[List[List[Double]=> List[Double]]],
                                  sigma: Double,
                                  sizeClusterRow: List[List[Int]],
                                  sizeClusterCol: List[Int],
                                  shuffle: Boolean=false,
                                  mixingProportion: Double=0D):
  List[(Int, Int, List[Double], List[Double])] = {

    require(allEqual(prototypes.map(_.length), sizeClusterRow.map(_.length)), "SizeClusteRow and prototype lists lengths differ")
    require(sizeClusterCol.length == prototypes.length, "sizeClusterCol and prototype lengths differ")
    require(sizeClusterRow.map(_.sum == sizeClusterRow.head.sum).forall(identity), "Each sizeClusterRow list should sum up to the same number (= n, the number of observations)")

    val length = 100
    val indices:List[Double] = (1 to length).map(_/length.toDouble).toList

    val modes = prototypes.indices.map(l => {prototypes(l).indices.map(k => {DenseVector(prototypes(l)(k)(indices).toArray)}).toList}).toList
    val covariances = prototypes.indices.map(l => {prototypes(l).indices.map(_ => {
      diag(DenseVector(Array.fill(indices.length){sigma}))}).toList}).toList

    val dataPerBlock = generateDataPerBlock(modes, covariances, sizeClusterRow, sizeClusterCol, mixingProportion)
    val data: DenseMatrix[DenseVector[Double]] = doubleReduce(dataPerBlock)
    val dataShuffled = applyConditionalShuffleData(data, shuffle)
    (0 until data.rows).map(i => {(0 until data.cols).map(j => {(i, j, indices, dataShuffled(i,j).toArray.toList)}).toList}).reduce(_++_)
  }

  def randomLBMDataGeneration(modes: List[List[DenseVector[Double]]],
                              covariances: List[List[DenseMatrix[Double]]],
                              sizeClusterRow: List[Int],
                              sizeClusterCol: List[Int],
                              shuffle: Boolean = false,
                              mixingProportion: Double = 0D): DenseMatrix[DenseVector[Double]] = {

    val modeLengths = modes.map(_.length)
    val K = modeLengths.head
    val covLengths = covariances.map(_.length)
    val L = modes.length

    require(modeLengths.forall(_ == modeLengths.head), "In LBM case, every column must have the same number of modes")
    require(covLengths.forall(_ == covLengths.head), "In LBM case, every column must have the same number of covariances matrices")
    require(K == covLengths.head, "modes and covariances K do not match")
    require(modes.length == covariances.length, "modes and covariances L do not match")
    require(sizeClusterRow.length == K)
    require(sizeClusterCol.length == L)

    val sizeClusterRowEachColumn = List.fill(L)(sizeClusterRow)
    val dataPerBlock: List[List[DenseMatrix[DenseVector[Double]]]] = generateDataPerBlock(
      modes,
      covariances,
      sizeClusterRowEachColumn,
      sizeClusterCol,
      mixingProportion)
    val data: DenseMatrix[DenseVector[Double]] = doubleReduce(dataPerBlock)

    applyConditionalShuffleData(data, shuffle)

  }

  def randomLBMDataGeneration(mvGaussian: List[List[MultivariateGaussian]],
                              sizeClusterRow: List[Int],
                              sizeClusterCol: List[Int],
                              shuffle: Boolean,
                              mixingProportion: Double): DenseMatrix[DenseVector[Double]] = {

    val mvGaussisnLengths = mvGaussian.map(_.length)
    val K = mvGaussian.head.length
    val L = mvGaussian.length

    require(mvGaussisnLengths.forall(_ == K), "In LBM case, every column must have the same number of modes")
    require(sizeClusterRow.length == K)
    require(sizeClusterCol.length == L)

    val sizeClusterRowEachColumn = List.fill(L)(sizeClusterRow)
    val dataPerBlock: List[List[DenseMatrix[DenseVector[Double]]]] = generateDataPerBlock(
      mvGaussian,
      sizeClusterRowEachColumn,
      sizeClusterCol,
      mixingProportion,
      None)
    val data: DenseMatrix[DenseVector[Double]] = doubleReduce(dataPerBlock)

    applyConditionalShuffleData(data, shuffle)

  }


  def generateDataPerBlock(modes: List[List[DenseVector[Double]]],
                           covariances: List[List[DenseMatrix[Double]]],
                           sizeClusterRow: List[List[Int]],
                           sizeClusterCol: List[Int],
                           mixingProportion: Double=0D,
                           seed: Option[Int] = None): List[List[DenseMatrix[DenseVector[Double]]]]={

    val actualSeed = seed match {
      case Some(s) => s;
      case None => scala.util.Random.nextInt()
    }

    implicit val basis: RandBasis = RandBasis.withSeed(actualSeed)

    require(mixingProportion>=0D & mixingProportion <=1D)
    val L = modes.length
    val KVec = modes.map(_.length)
    val MGaussians = (0 until L).map(l => {
      modes(l).indices.map(k => {
        MultivariateGaussian(modes(l)(k),covariances(l)(k))
      })
    })
    modes.indices.map(l => {
      modes(l).indices.map(k => {
        val dataList: Array[DenseVector[Double]] = MGaussians(l)(k).sample(sizeClusterRow(l)(k)*sizeClusterCol(l)).toArray

        val mixedData = dataList.map(data => {
          val isMixed = sampleWithSeed(Array(1-mixingProportion, mixingProportion), 2)
          if (isMixed == 0){
            data
          } else {
            val newl = sample(L)
            val newk = sample(KVec(newl))
            MGaussians(newl)(newk).draw()
          }
        })

        DenseMatrix(mixedData).reshape(sizeClusterRow(l)(k),sizeClusterCol(l))
      }).toList
    }).toList
  }

  def generateDataPerBlock(mvGaussian: List[List[MultivariateGaussian]],
                           sizeClusterRow: List[List[Int]],
                           sizeClusterCol: List[Int],
                           mixingProportion: Double,
                           seed: Option[Int]): List[List[DenseMatrix[DenseVector[Double]]]]={

    val actualSeed = seed match {
      case Some(s) => s;
      case None => scala.util.Random.nextInt()
    }

    implicit val basis: RandBasis = RandBasis.withSeed(actualSeed)

    require(mixingProportion>=0D & mixingProportion <=1D)
    val L = mvGaussian.length
    val KVec = mvGaussian.map(_.length)

    mvGaussian.indices.map(l => {
      mvGaussian(l).indices.map(k => {
        val dataList: Array[DenseVector[Double]] = mvGaussian(l)(k).sample(sizeClusterRow(l)(k)*sizeClusterCol(l)).toArray

        val mixedData = dataList.map(data => {
          val isMixed = sampleWithSeed(Array(1-mixingProportion, mixingProportion), 2)
          if (isMixed == 0){
            data
          } else {
            val newl = sample(L)
            val newk = sample(KVec(newl))
            mvGaussian(newl)(newk).draw()
          }
        })

        DenseMatrix(mixedData).reshape(sizeClusterRow(l)(k),sizeClusterCol(l))
      }).toList
    }).toList
  }


  def doubleReduce(dataList: List[List[DenseMatrix[DenseVector[Double]]]]): DenseMatrix[DenseVector[Double]] = {
    dataList.indices.map(l => {
      dataList(l).indices.map(k_l => {
        dataList(l)(k_l)
      }).reduce(DenseMatrix.vertcat(_,_))
    }).reduce(DenseMatrix.horzcat(_,_))
  }

  def applyConditionalShuffleData(data: DenseMatrix[DenseVector[Double]], shuffle:Boolean): DenseMatrix[DenseVector[Double]]= {
    if(shuffle){
      val newRowIndex: List[Int] = Random.shuffle((0 until data.rows).toList)
      val newColIndex: List[Int] = Random.shuffle((0 until data.cols).toList)
      DenseMatrix.tabulate[DenseVector[Double]](data.rows,data.cols){ (i,j) => data(newRowIndex(i), newColIndex(j))}
    } else data
  }




}
