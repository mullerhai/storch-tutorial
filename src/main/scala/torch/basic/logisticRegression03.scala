package torch.basic

//import spire.random.Random

import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import org.bytedeco.pytorch.{OutputArchive, TensorExampleVectorIterator}
import torch.*
import torch.Device.{CPU, CUDA}
import torch.data.dataset.ChunkSharedBatchDataset
import torch.nn.functional as F
import torch.nn.modules.HasParams
import torch.optim.Adam
import torchvision.datasets.FashionMNIST

import java.nio.file.Paths
//import torchvision.datasets.FashionMNIST

import java.nio.file.Paths
//import scala.runtime.stdLibPatches.Predef.nn
import torch.internal.NativeConverters.{fromNative, toNative}

import scala.util.{Random, Using}

object logisticRegression03 extends App {

  //  @main
  def main(): Unit =
    val device = if torch.cuda.isAvailable then CUDA else CPU

    val input_size = 28 * 28
    val num_classes = 10
    val num_epochs = 50
    val batch_size = 100
    val learning_rate = 0.001f
    val dataPath = Paths.get("D:\\data\\FashionMNIST")
    val train_dataset = FashionMNIST(dataPath, train = true, download = true)
    val test_dataset = FashionMNIST(dataPath, train = false)
    val evalFeatures = test_dataset.features.to(device)
    val evalTargets = test_dataset.targets.to(device)
    val r = Random(seed = 0)

    def dataLoader: Iterator[(Tensor[Float32], Tensor[Int64])] =
      r.shuffle(train_dataset).grouped(8).map { batch =>
        val (features, targets) = batch.unzip
        (torch.stack(features).to(device), torch.stack(targets).to(device))
      }

    val model = nn.Linear(input_size, num_classes)
    val criterion = nn.loss.CrossEntropyLoss()
    val optimizer = torch.optim.SGD(model.parameters(true), lr = learning_rate)

    import org.bytedeco.pytorch.{ChunkDatasetOptions, Example, ExampleIterator, ExampleStack, ExampleVector, ExampleVectorIterator, JavaDataset, JavaDistributedRandomTensorDataLoader, JavaDistributedSequentialTensorDataLoader, JavaRandomDataLoader, JavaRandomTensorDataLoader, JavaSequentialTensorDataLoader, JavaStatefulDataset, JavaStreamDataLoader, RandomSampler, SizeTArrayRef, SizeTOptional, TensorExample, TensorExampleIterator, TensorExampleStack, TensorExampleVector, AbstractTensor as Tensor, ChunkDataReader as CDR, ChunkDataset as CD, ChunkRandomDataLoader as CRDL, ChunkSharedBatchDataset as CSBD, DataLoaderOptions as DLO, DistributedRandomSampler as DRS, DistributedSequentialSampler as DSS, JavaStreamDataset as JSD, JavaTensorDataset as TD, StreamSampler as STS}
    import torch.data.DataLoaderOptions
    import torch.data.dataloader.*
    import torch.data.datareader.{ChunkDataReader, ChunkTensorDataReader, ExampleVectorReader, TensorExampleVectorReader}
    import torch.data.dataset.*
    import torch.data.dataset.java.{StatefulDataset, StatefulTensorDataset, StreamDataset, StreamTensorDataset, TensorDataset, JavaDataset as JD}
    import torch.data.sampler.{DistributedRandomSampler, DistributedSequentialSampler, StreamSampler, RandomSampler as RS, SequentialSampler as SS}

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
              //              println(s"loss grad_fn: ${evalLoss.grad_fn()}")
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

  //    Using.resource(torch.noGrad()){
  //      
  //    }
}
