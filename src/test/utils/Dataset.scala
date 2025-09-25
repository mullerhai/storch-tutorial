package torch.utils

import scala.util.continuations._
import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._
import scala.virtualization.lms.common._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.{Map => MutableMap}
import scala.math._

trait Dataset extends TensorDsl {

  class Timer (val index: Int){
    unchecked[Unit](s"clock_t begin_$index, end_$index; double time_spent_$index")
    def startTimer = { unchecked[Unit](s"begin_$index = clock()") }
    def stopTimer = { unchecked[Unit](s"end_$index = clock()") }
    def printElapsedTime = {
      unchecked[Unit](
        s"end_$index = clock(); printf(",
        "\"Time elapsed: %f\\n\", ",
        s"(double)(end_$index - begin_$index) / CLOCKS_PER_SEC)")
    }
  }

  object Timer {
    var index: Int = 0
    def apply(): Timer = {
      val timer = new Timer(index)
      index += 1
      timer
    }
  }

  def get_time() = unchecked[Double]("((double)clock() / CLOCKS_PER_SEC)")

  class Timer2 (index: Int) {
    unchecked[Unit](s"struct timeval begin_$index, end_$index, diff_$index")
    def startTimer = { unchecked[Unit](s"gettimeofday(&begin_$index, NULL)") }
    def getElapsedTime: Rep[Long] = {
      unchecked[Unit](s"gettimeofday(&end_$index, NULL)")
      unchecked[Unit](s"timeval_subtract(&diff_$index, &end_$index, &begin_$index);")
      unchecked[Long](s"((diff_$index.tv_sec * 1000000L) + (diff_$index.tv_usec))")
    }
  }

  object Timer2 {
    var index: Int = 0
    def apply(): Timer2 = {
      val timer = new Timer2(index)
      index += 1
      timer
    }
  }

  object Encoding {
    val ix_a = 96  // index starts from 1

    def char_to_ix(ch: Rep[Char]): Rep[Int] = ch.AsInstanceOf[Int] - ix_a
    def ix_to_char(ix: Rep[Int]): Rep[Char] = (ix + ix_a).AsInstanceOf[Char]
  }

  class DataLoader(name: String, train: Boolean, mean: Float, std: Float, dims: Seq[Int]) {

    val fd = open(s"../data/bin/${name}_${if (train) "train" else "test"}.bin")
    val len = filelen(fd)
    val data = mmap[Float](fd, len)
    val dLength = (len/4L).toInt

    val tfd = open(s"../data/bin/${name}_${if (train) "train" else "test"}_target.bin")
    val tlen = filelen(tfd)
    val target = mmap[Int](tfd, tlen)
    val length: Rep[Int] = tlen.toInt/4

    def dataset = new Tensor(data, Seq(60000, dims(1), dims(2)))

    @virtualize
    def normalize() = {
      this.foreach { (i, t, d) =>
        t.normalize(mean, std, inPlace = true)
      }
    }

    @virtualize
    def foreach(f: (Rep[Int], Tensor, Rep[Int]) => Unit) = {
      var off = var_new(0)
      for (index <- 0 until length: Rep[Range]) {
        val dataPtr = slice(data, off)
        val t = Tensor(dataPtr, dims : _*)
        f(index, t, target(index))
        off += t.scalarCount
      }
      assertC(off == dLength, "Data length doesn't match\\n")
    }

    @virtualize
    def foreachBatch(batchSize: Int)(f: (Rep[Int], Tensor, Rep[Array[Int]]) => Unit) = {
      var off = var_new(0)
      for (batchIndex <- 0 until (length / batchSize): Rep[Range]) {
        val dataPtr = slice(data, off)
        val t = Tensor(dataPtr, (batchSize +: dims.toSeq): _*)
        val targets = slice(target, batchIndex * batchSize)
        f(batchIndex, t, targets)
        off += t.scalarCount
      }
    }
  }

  class Cifar10DataLoader(name: String, train: Boolean, dims: Seq[Int]) {

    val fd = open(name)
    val len = filelen(fd)
    val data = mmap[Char](fd, len)
    // each entry is target + image
    val entrySize = (dims.product + 1)
    val dLength = (len/entrySize.toLong).toInt
    val length = dLength

    val x = NewArray[Float](dLength * dims.product)
    val y = NewArray[Int](dLength)

    for (i <- (0 until dLength): Rep[Range]) {
      y(i) = unchecked[Int]("(int32_t)(unsigned char)", data(i * entrySize))
      for (j <- (0 until dims.product): Rep[Range]) {
        x(i * dims.product + j) = uncheckedPure[Float]("(float)(unsigned char)", data(i * entrySize + 1 + j)) / 255.0f
      }
    }

    @virtualize
    def foreachBatch(batchSize: Int)(f: (Rep[Int], Tensor, Rep[Array[Int]]) => Unit) = {
      for (batchIndex <- 0 until (dLength / batchSize): Rep[Range]) {
        val dataPtr = slice(x, batchIndex * batchSize * dims.product)
        val targets = slice(y, batchIndex * batchSize)
        val t = Tensor(dataPtr, (batchSize +: dims.toSeq): _*)
        f(batchIndex, t, targets)
      }
    }
  }

  @virtualize
  class DeepSpeechDataLoader(name: String, train: Boolean) {

    // open file
    val fd = open(name)
    val len = filelen(fd)
    printf("file size is %ld\\n", len)

    val data = mmap[Char](fd, len)
    object reader {
      val pointer = var_new(unchecked[Long]("(long)", data))
      def nextI(size: Rep[Int] = 1): Rep[Array[Int]] = {
        val temp: Rep[Long] = pointer
        val intArray = unchecked[Array[Int]]("(int32_t*) ", temp)
        pointer += 4 * size
        intArray
      }
      def nextInt(): Rep[Int] = nextI()(0)
      def nextF(size: Rep[Int] = 1): Rep[Array[Float]] = {
        val temp: Rep[Long] = pointer
        val floatArray = unchecked[Array[Float]]("(float*) ", temp)
        pointer += 4 * size
        floatArray
      }
    }

    // get batchSize and numBatches
    val batchSize = reader.nextInt  // batchSize is 32, and numBatches is 5
    val num_Batches = reader.nextInt
    val numBatches = 200
    val length = batchSize * numBatches
    printf("data size is %d batches, %d batch size\\n", numBatches, batchSize)

    // get array to store information for each batch
    val freqSizes: Rep[Array[Int]] = NewArray[Int](numBatches)
    val maxLengths: Rep[Array[Int]] = NewArray[Int](numBatches)
    // get array of arrays to store the pointers to data
    val inputs: Rep[Array[Array[Float]]] = NewArray[Array[Float]](numBatches)
    val percents: Rep[Array[Array[Float]]] = NewArray[Array[Float]](numBatches)
    // val inputSizes: Rep[Array[Array[Int]]] = NewArray[Array[Int]](numBatches)
    // val inputs = NewArray[Tensor](numBatches)
    // val percents = NewArray[Tensor](numBatches)
    val targetSizes: Rep[Array[Array[Int]]] = NewArray[Array[Int]](numBatches)
    val targets: Rep[Array[Array[Int]]] = NewArray[Array[Int]](numBatches)

    generateRawComment("load data by batchs")
    for (batch <- (0 until numBatches: Rep[Range])) {
      // First, get frequency_size and max_length
      freqSizes(batch) = reader.nextInt  // freqSize is 161, and maxLength is 229
      maxLengths(batch) = reader.nextInt
      // then the sound tensor of float [batchSize * 1 * freqSize * maxLength]
      inputs(batch) = reader.nextF(batchSize * freqSizes(batch) * maxLengths(batch))
      // then the percentage tensor of float [batchSize] (percentage of padding for each sound)
      percents(batch) = reader.nextF(batchSize)

      // then the targetSize tensor of Int[batchSize]
      targetSizes(batch) = reader.nextI(batchSize)
      val sumTargetSize: Rep[Int] = unchecked[Int]("accumulate(", targetSizes(batch), ", ", targetSizes(batch), " + ", batchSize, ", 0)")
      // then the targets tensor of Int[sum(targetSize)]
      targets(batch) = reader.nextI(sumTargetSize)
    }

    @virtualize
    // the lossFun takes a Batch (Tensor), inputLengths, labels, labelLengths (all Rep[Array[Int]])
    def foreachBatch(f: (Rep[Int], Tensor, Rep[Array[Float]], Rep[Array[Int]], Rep[Array[Int]]) => Unit) = {
      for (batchIndex <- 0 until numBatches: Rep[Range]) {
        val maxLength = maxLengths(batchIndex)
        val freqSize = freqSizes(batchIndex)
        val input: Tensor = Tensor(inputs(batchIndex), batchSize, 1, freqSize, maxLength)
        val percent: Rep[Array[Float]] = percents(batchIndex)
        val target: Rep[Array[Int]] = targets(batchIndex)
        val targetSize: Rep[Array[Int]] = targetSizes(batchIndex)
        f(batchIndex, input, percent, target, targetSize)
      }
    }
  }
}