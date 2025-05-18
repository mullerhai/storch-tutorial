package torch
package advance

import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import org.bytedeco.pytorch.{OutputArchive, TensorExampleVectorIterator}
import torch.Device.{CPU, CUDA}
import torch.{FloatNN, *}
import torch.data.DataLoaderOptions
import torch.data.dataloader.*
import torch.data.datareader.{ChunkDataReader, ChunkTensorDataReader, ExampleVectorReader, TensorExampleVectorReader}
import torch.nn.functional as F
import torch.nn.modules.HasParams
import torch.optim.Adam
import torchvision.datasets.FashionMNIST

import java.nio.file.Paths
import torch.data.dataset.*
import torch.data.dataset.java.{StatefulDataset, StatefulTensorDataset, StreamDataset, StreamTensorDataset, TensorDataset, JavaDataset as JD}
import torch.data.sampler.{DistributedRandomSampler, DistributedSequentialSampler, StreamSampler, RandomSampler as RS, SequentialSampler as SS}
import torch.internal.NativeConverters.fromNative

import scala.util.{Random, Using}

class BiRNNNetwork[D <: FloatNN : Default](input_size: Int, hidden_size: Int, num_layers: Int, num_classes: Int) extends HasParams[D] {

  val lstm = register(nn.LSTM(input_size, hidden_size, num_layers, batch_first = true, bidirectional = true))
  val fc = register(nn.Linear(hidden_size * 2, num_classes))

  def apply(input: Tensor[D]): Tensor[D] = {
    val h0 = torch.zeros(Seq(num_layers, input.size.head, hidden_size))
    val c0 = torch.zeros(Seq(num_layers, input.size.head, hidden_size))
    //    println(s"input shape ${ input.size.mkString(",")}")
    val outTuple = lstm(input, h0.to(dtype = this.paramType), c0.to(dtype = this.paramType))
    //    val out2 = fc(outTuple._1)
    var out: Tensor[D] = outTuple._1
    out = out.index(torch.indexing.::, -1, ::)
    out = F.logSoftmax(fc(out), dim = 1)
    out
  }

}

object BiRNNNetwork03 {
  //  @main
  def main(): Unit = {
    val sequence_length = 10
    val input_size = 28
    val hidden_size = 128
    val num_layers = 2
    val num_classes = 10
    val batch_size = 100
    val num_epochs = 2
    val learning_rate = 0.003
    val model = BiRNNNetwork[Float32](input_size, hidden_size, num_layers, num_classes)
    val criterion = nn.loss.CrossEntropyLoss()
    val optimizer = torch.optim.Adam(model.parameters, lr = learning_rate)
  }
}
