package torch
package advance

import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import org.bytedeco.pytorch.{OutputArchive, TensorExampleVectorIterator}
import torch.Device.{CPU, CUDA}
import torch.{FloatNN, *}
//import torch.data.DataLoaderOptions
//import torch.data.dataloader.*
//import torch.data.datareader.{ChunkDataReader, ChunkTensorDataReader, ExampleVectorReader, TensorExampleVectorReader}

import torch.nn.functional as F
import torch.nn.modules.HasParams
import torch.optim.Adam
import torch.utils.data.dataset.custom.FashionMNIST
import java.nio.file.Paths
//import torch.data.dataset.*
//import torch.data.dataset.java.{StatefulDataset, StatefulTensorDataset, StreamDataset, StreamTensorDataset, TensorDataset, JavaDataset as JD}
//import torch.data.sampler.{DistributedRandomSampler, DistributedSequentialSampler, StreamSampler, RandomSampler as RS, SequentialSampler as SS}
import torch.nn as nn
import torch.internal.NativeConverters.fromNative

import scala.util.{Random, Using}


class RNNLM[D <: FloatNN : Default](vocab_size: Int, embed_size: Int, hidden_size: Int, num_layers: Int) extends HasParams[D] {
  val embed = register(nn.Embedding(vocab_size, embed_size))
  val lstm = register(nn.LSTM(embed_size, hidden_size, num_layers, batch_first = true))
  val linear = register(nn.Linear(hidden_size, vocab_size))

  def apply(input: Tensor[D], hidden_states: (Tensor[D], Tensor[D])): Tuple2[Tensor[D], Tuple2[Tensor[D], Tensor[D]]] = {
    var out = embed(input)
    val outTriple = lstm(input, hidden_states._1, hidden_states._2)
    out = outTriple._1
    val new_h = outTriple._2
    val new_c = outTriple._3
    out = out.reshape(out.size.head * out.size(1), out.size(2))
    out = linear(out)
    Tuple2(out, Tuple2(new_h, new_c))
  }
}

object RNNLM04 {

}
