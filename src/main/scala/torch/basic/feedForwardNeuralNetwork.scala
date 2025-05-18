package torch.basic

import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import torch.*
import torch.Device.{CPU, CUDA}
import torch.nn.modules
import torch.nn.modules.HasParams
import torchvision.datasets.FashionMNIST

import java.nio.file.Paths
//import torchvision.datasets.FashionMNIST
//import scala.runtime.stdLibPatches.Predef.nn
import torch.internal.NativeConverters.fromNative

import scala.util.{Random, Using}

class NeuralNet2[D <: FloatNN : Default](input_size: Int, hidden_size: Int, num_classes: Int) extends HasParams[D] {

  val fc1 = register(nn.Linear(input_size, hidden_size))
  val relu = register(nn.ReLU(true))
  val fc2 = register(nn.Linear(hidden_size, num_classes))

  def apply(input: Tensor[D]): Tensor[D] = {
    val out = fc2(relu(fc1(input)))
    out
  }

}

object feedForwardNeuralNetwork {

  //  @main
  def main(): Unit =
    val device = if torch.cuda.isAvailable then CUDA else CPU
    val input_size = 28 * 28
    val hidden_size = 500
    val num_classes = 10
    val num_epochs = 50
    val batch_size = 100
    val learning_rate = 0.001f
    val dataPath = Paths.get("D:\\data\\FashionMNIST")
    val dataPathCIFAR10 = Paths.get("D:\\data\\CIFAR10")
    println("try to read cifar10 ...")
    //    val traincifar_dataset = CIFAR10(dataPathCIFAR10, train = true, download = true)

    println("read cifar10 finish")
    val train_dataset = FashionMNIST(dataPath, train = true, download = true)
    val test_dataset = FashionMNIST(dataPath, train = false)

    val evalFeatures = test_dataset.features.to(device)
    val evalTargets = test_dataset.targets.to(device)
    val r = Random(seed = 0)
    val criterion = nn.loss.CrossEntropyLoss()

    def dataLoader: Iterator[(Tensor[Float32], Tensor[Int64])] =
      r.shuffle(train_dataset).grouped(8).map { batch =>
        val (features, targets) = batch.unzip
        (torch.stack(features).to(device), torch.stack(targets).to(device))
      }

    val model = NeuralNet[Float32](input_size, hidden_size, num_classes).to(device)
    val optimizer = torch.optim.SGD(model.parameters(true), lr = learning_rate)
    import org.bytedeco.pytorch.*
    import torch.data.DataLoaderOptions
    import torch.data.datareader.ChunkDataReader
    def exampleVectorToExample(exVec: ExampleVector): Example = {
      val example = new Example(exVec.get(0).data(), exVec.get(0).target())
      example
    }


    val exampleSeq = train_dataset.map(x => new Example(x._1.native, x._2.native))
    //  val ex1 = new Example(mnistTrain.features.native ,mnistTrain.targets.native)
    val exampleVector = new ExampleVector(exampleSeq *)
    val reader = new ChunkDataReader()
    reader(exampleVector)
    val prefetch_count = 1
    //  val ds = new ChunkSharedBatchDataset(new ChunkDataset(reader, new RandomSampler(exampleSeq.size), new RandomSampler(exampleSeq.size), new ChunkDatasetOptions(prefetch_count, batch_size))).map(new ExampleStack)
    //  val ds  = new ChunkSharedTensorBatchDataset(new ChunkTensorDataset(reader,new RS(exampleTensorSeq.size),new ChunkDatasetOptions(prefetch_count, batch_size))).map(new TensorExampleStack)
    val ds = new ChunkSharedBatchDataset(
      new ChunkDataset(
        reader,
        new RandomSampler(exampleSeq.size),
        new RandomSampler(exampleSeq.size),
        new ChunkDatasetOptions(prefetch_count, batch_size)
      )
    ).map(new ExampleStack)

    val opts = new DataLoaderOptions(100)
    //  opts.workers.put(5)
    opts.batch_size.put(100)
    //  opts.enforce_ordering.put(true)
    //  opts.drop_last.put(false)
    val data_loader = new ChunkRandomDataLoader(ds, opts)
    val total_step = train_dataset.length // 2000 //data_loader //
    (1 to num_epochs).foreach(epoch => {
      var it: ExampleIterator = data_loader.begin
      var batchIndex = 0
      println("coming in for loop")
      while (!it.equals(data_loader.end)) {
        Using.resource(new PointerScope()) { p =>
          val batch = it.access
          optimizer.zeroGrad()
          val trainDataTensor = fromNative(batch.data())
          val ze = batch.data().shape
          //          println(s"ze: ${ze(0)} 1: ${ze(1)} 2 : ${ze(2)} ${ze(3)}")
          val prediction = model(fromNative(batch.data().view(-1, 28 * 28)).reshape(-1, 28 * 28))
          val loss = criterion(prediction, fromNative(batch.target()))
          loss.backward()
          optimizer.step()
          it = it.increment
          //          println(s"train Loss grad_fn: ${loss.grad_fn()}")
          //          println(s"train out grad_fn: ${prediction.grad_fn()}")
          batchIndex += 1
          if batchIndex % 200 == 0 then
            // run evaluation
            torch.noGrad {
              val correct = 0
              val total = 0
              val predictions = model(evalFeatures.reshape(-1, 28 * 28))
              val evalLoss = criterion(predictions, evalTargets)
              val featuresData = new Array[Float](1000)
              val fp4 = new FloatPointer(predictions.native.data_ptr_float())
              fp4.get(featuresData)
              println(s"\n ffff size ${featuresData.size} shape ${
                evalFeatures.shape
                  .mkString(", ")
              }a data ${featuresData.mkString(" ")}")
              println(s"predictions : ${predictions} \n")
              println(s"evalLoss grad_fn: ${evalLoss.grad_fn()}")
              println(s"evalout grad_fn: ${predictions.grad_fn()}")
              val accuracy =
                (predictions.argmax(dim = 1).eq(evalTargets).sum / test_dataset.length).item
              println(
                f"Epoch: $epoch | Batch: $batchIndex%4d | Training loss: ${loss.item}%.4f | Eval loss: ${evalLoss.item}%.4f | Eval accuracy: $accuracy%.4f"
              )
            }

          //        it = it.increment

        }
      }

    })
    println(s"model ${model.modules.toSeq.mkString(" \n")}")
    println(s"model ${model.summarize}")
  //    torch.pickleSave()
  //    model.state_dict()
  //    torch.save(model.state_dict(), 'model.ckpt')


}