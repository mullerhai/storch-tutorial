package torch
package pro

import torch.FloatNN
import torch.nn.modules.{HasParams, TensorModule}
import org.bytedeco.pytorch.{OutputArchive, TensorExampleVectorIterator}
import torch.Device.{CPU, CUDA}
import torch.data.dataset.ChunkSharedBatchDataset
import torch.nn.functional as F
import torch.nn.modules.HasParams
import torch.optim.Adam
import torch.*
import torchvision.datasets.FashionMNIST

import java.nio.file.Paths
import scala.util.{Random, Using}
import torch.data.DataLoaderOptions
import torch.data.dataloader.*
import torch.data.datareader.{ChunkDataReader, ChunkTensorDataReader, ExampleVectorReader, TensorExampleVectorReader}
import torch.data.dataset.*
import torch.data.dataset.java.{StatefulDataset, StatefulTensorDataset, StreamDataset, StreamTensorDataset, TensorDataset, JavaDataset as JD}
import torch.data.sampler.{DistributedRandomSampler, DistributedSequentialSampler, StreamSampler, RandomSampler as RS, SequentialSampler as SS}
import torch.internal.NativeConverters.fromNative
import torch.nn.modules.conv.Conv2d

import scala.collection.mutable.ListBuffer

class ImageCaptioningEncoderCNNNetwork[D <: FloatNN :Default](embed_size: Int) extends HasParams[D] {

  val bn = nn.BatchNorm1d(embed_size,momentum = 0.01f)
}

//class EncoderCNN(nn.Module):
//    def __init__(self, embed_size):
//        """Load the pretrained ResNet-152 and replace top fc layer."""
//        super(EncoderCNN, self).__init__()
//        resnet = models.resnet152(pretrained=True)
//        modules = list(resnet.children())[:-1]  # delete the last fc layer.
//        self.resnet = nn.Sequential(*modules)
//        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
//        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
//
//    def forward(self, images):
//        """Extract feature vectors from input images."""
//        with torch.no_grad():
//            features = self.resnet(images)
//        features = features.reshape(features.size(0), -1)
//        features = self.bn(self.linear(features))
//        return features

//class ImageCaptioningEncoderRNNNetwork[D <: FloatNN : Default](embed_size: Int = 784, hidden_size: Int = 400, vocab_size: Int, num_layers: Int, max_seq_length: Int = 20) extends HasParams[D] {
//  val resnet:TensorModule[D] = null
//  val modules = resnet.namedChildren.values
//  val resnetNew = nn.Sequential(modules.toSeq.map(_.asInstanceOf[TensorModule[D]])*)
//  val linear = register(nn.Linear(resnet.fc.in_features,embed_size))
//  val bn = register(nn.BatchNorm1d(embed_size, momentum = 0.01f))
//  
//  def apply(images: Tensor[D]): Tensor[D] ={
////    var features: Tensor[D] = null
//    torch.noGrad(
//      var features = resnetNew(images)
//      features = features.reshape(features.shape(0), -1)
//      features = bn(linear(features))
//      features
//    )
////    features = features.reshape(features.shape(0),-1)
////    features = bn(linear(features))
////    features
//  }
//  
//}

class ImageCaptioningDecoderRNNNetwork[D <: FloatNN : Default](embed_size: Int = 784, hidden_size: Int = 400, vocab_size: Int , num_layers: Int, max_seq_length: Int = 20) extends HasParams[D] {

  val embed = register(nn.Embedding(vocab_size, embed_size))
  val lstm = register(nn.LSTM(embed_size, hidden_size, num_layers,batch_first = true))
  val linear = register(nn.Linear(hidden_size, vocab_size))

  def apply(features: Tensor[D], captions: Tensor[D], lengths: Tensor[D]): Tensor[D]={
    var embeddings = embed(captions)
    embeddings = torch.cat(Seq(features.unsqueeze(1),embeddings), 1)
    val packed = torch.pack_padded_sequence(embeddings,lengths,batch_first =true) //注意修改
    val hidTriple = lstm.forward_with_packed_input(packed)
    val outputs = linear(hidTriple._2)
    outputs
  }

  def sample(features: Tensor[D], states : Option[Tensor[D]] = None):Tensor[Int64] ={
    var sampled_ids = new  ListBuffer[Tensor[Int64]]()
    var inputs  = features.unsqueeze(1)
    (1 to max_seq_length).foreach(i => {
      val outTriple = lstm(inputs, states)
      val outputs = linear(outTriple._1.squeeze(1))
      val predOut = outputs.max(1)
      sampled_ids.append(predOut._2)
      inputs = embed(predOut._1)
      inputs = inputs.unsqueeze(1)



    })
    val samplerIds = torch.stack(sampled_ids.toSeq,1)
    samplerIds
  }

}
object ImageCaptioningNetwork {

}
