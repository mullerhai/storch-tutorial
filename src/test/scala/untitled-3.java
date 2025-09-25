//import org.bytedeco.javacpp.PointerScope;
//import org.bytedeco.pytorch.global.torch as torchNative;
//import org.bytedeco.pytorch.AbstractTensor;
//import org.bytedeco.pytorch.Node;
//import torch.Tensor;
//import torch.nn.functional.F;
//import torch.optim.SGD;
//
//import java.util.Arrays;
//import java.util.List;
//
//public class TensorOps01 {
//
//    public static void main(String[] args) {
//        // randSelect01();
//        seqDataTensor();
//        singleDataTensor();
//        randnDataTensor();
//        rawTensorGrad();
//    }
//
//    public static void rawTensorGrad() {
//        try (PointerScope scope = new PointerScope()) {
//            AbstractTensor dfx = AbstractTensor.create(1.0);
//            AbstractTensor dfw = AbstractTensor.create(2.0);
//            dfw.set_requires_grad(true);
//            AbstractTensor dfb = AbstractTensor.create(3.0);
//            AbstractTensor dfy = dfx.mul(dfw).add(dfb);
//            dfy.grad();
//            System.out.printf("before backward dfy grad %s grad require %s dfy grad_fn %s\n",
//                    dfy.grad(), dfy.requires_grad(), dfy.grad_fn());
//            
//            dfy.backward();
//            System.out.printf("after backward dfy grad: %s grad require %s dfy.grad_fn： %s\n",
//                    dfy.grad(), dfy.requires_grad(), dfy.grad_fn());
//            System.out.printf("after backward dfw grad: %s dfw.grad_fn： %s\n",
//                    dfw.grad(), dfw.grad_fn());
//        }
//    }
//
//    public static void singleDataTensor() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor x = torch.Tensor(1.0, true);
//            Tensor w = torch.Tensor(2.0, true);
//            
//            System.out.printf("before backward w requiresGrad %s w grad %s\n",
//                    w.requiresGrad(), w.grad());
//            w.set_requires_grad(true);
//            System.out.printf("after set grad w requiresGrad %s w grad %s\n",
//                    w.requiresGrad(), w.grad());
//            
//            Tensor b = torch.Tensor(3.0, true);
//            Tensor y = w.mul(x).add(b);
//            
//            System.out.printf("y %s\n", y);
//            System.out.printf("y grad: %s ,,w grad %s\n",
//                    y.native_().grad(), w.grad());
//            System.out.printf(" y before backward : grad fn %s native fn %s\n",
//                    y.grad_fn(), y.native_().grad_fn());
//            System.out.printf("before backward  y.grad: %s\n", y.grad());
//            System.out.printf("before backward  x.grad: %s\n", x.grad());
//            System.out.printf("before backward  w.grad: %s\n", w.grad());
//            System.out.printf("before backward  b.grad: %s\n", b.grad());
//            
//            System.out.printf(" y grad fn %s\n", y.grad_fn());
//            y.grad_fn();
//            System.out.printf("before backward y requiresGrad %s\n", y.requiresGrad());
//            
//            try {
//                y.backward(); // 可能抛出异常
//            } catch (RuntimeException e) {
//                System.out.println("Exception in backward(): " + e.getMessage());
//            }
//            
//            System.out.printf("after backward y requiresGrad %s\n", y.requiresGrad());
//            System.out.printf("y grad: %s ,,w grad %s\n",
//                    y.native_().grad(), w.grad());
//            System.out.printf("y after backward : grad fn %s native fn %s\n",
//                    y.grad_fn(), y.native_().grad_fn());
//            System.out.printf(" after backward : y.grad: %s\n", y.grad());
//            System.out.printf(" after backward : x.grad: %s\n", x.grad());
//            System.out.printf(" after backward : w.grad: %s\n", w.grad());
//            System.out.printf("b.grad: %s\n", b.grad());
//            System.out.printf("y after backward : grad: %s\n", y.native_().grad());
//        }
//    }
//
//    public static void randnDataTensor() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor x1 = torch.randn(Arrays.asList(10L, 3L));
//            Tensor y1 = torch.randn(Arrays.asList(10L, 2L));
//            
//            Linear linear = nn.Linear(3, 2);
//            System.out.printf("weight %s\n", linear.weight());
//            System.out.printf("bias %s\n", linear.bias());
//            
//            MSELoss criterion = nn.loss.MSELoss();
//            SGD optimizer = torch.optim.SGD(linear.parameters(true), 0.01);
//            optimizer.zeroGrad();
//            
//            Tensor pred = linear.apply(x1);
//            System.out.printf("pred %s\n", pred.shape());
//            
//            Tensor loss = criterion.apply(pred.to(torch.float32()), y1.to(torch.float32()));
//            System.out.printf("loss  %s\n", loss.item());
//            loss.requiresGrad(true);
//            
//            try {
//                loss.backward(); // 可能抛出异常
//            } catch (RuntimeException e) {
//                System.out.println("Exception in backward(): " + e.getMessage());
//            }
//            
//            System.out.printf("dL/dw %s\n", linear.weight().grad());
//            System.out.printf("dL/db %s\n", linear.bias().grad());
//            System.out.printf("dL/dw %s\n", loss.grad_fn());
//            optimizer.step();
//        }
//    }
//
//    public static void seqDataTensor() {
//        try (PointerScope scope = new PointerScope()) {
//            List<Double> ddValues = Arrays.asList(24.0, 36.0);
//            Tensor dd = torch.Tensor(ddValues, true).reshape(1, 2);
//            
//            List<Double> wwValues = Arrays.asList(12.0, 18.0);
//            Tensor ww = torch.tensor(wwValues, true).reshape(2, 1);
//            
//            Tensor bb = torch.tensor(2.0, true);
//            ww.set_requires_grad(true);
//            bb.set_requires_grad(true);
//            bb.requiresGrad(true);
//
//            List<Double> xTrainValues = Arrays.asList(
//                    3.3, 4.4, 5.5, 6.71, 6.93, 4.169, 9.779, 6.182, 7.59, 2.167, 
//                    7.042, 10.791, 5.312, 7.993, 3.1
//            );
//            Tensor x_train = torch.Tensor(xTrainValues, true).view(5, 3).to(torch.float32());
//            
//            List<Double> yTrainValues = Arrays.asList(1.7, 2.76, 2.09, 3.19, 1.694);
//            Tensor y_train = torch.Tensor(yTrainValues, true).view(5, 1).to(torch.float32());
//            
//            Linear model = nn.Linear(3, 5);
//            Tensor outputs = model.apply(x_train);
//            
//            MSELoss criterion = nn.loss.MSELoss();
//            Tensor loss = criterion.apply(outputs, y_train);
//            
//            Tensor kk = dd.mul(ww).add(bb);
//            loss.backward();
//            
//            System.out.printf("loss after backward : grad: %s ,%s  grad fn %s \n",
//                    loss.native_().grad(), loss.grad(), loss.grad_fn());
//            
//            kk.set_requires_grad(true);
//            kk.requiresGrad(true);
//            
//            Node wwNode = ww.native_().grad_fn();
//            if (wwNode == null) {
//                System.out.println("null ... node");
//            }
//            System.out.printf("wwNode %s\n", wwNode);
//            System.out.printf("ww before backward requiresGrad %s grad %s grad fn %s\n",
//                    ww.requiresGrad(), ww.grad(), ww.grad_fn());
//            System.out.printf("bb before backward requiresGrad %s grad %s grad fn %s\n",
//                    bb.requiresGrad(), bb.grad(), bb.grad_fn());
//            System.out.printf("kk before backward requiresGrad %s grad %s grad fn %s\n",
//                    kk.requiresGrad(), kk.grad(), kk.grad_fn());
//
//            // kk.backward(); // pytorch grad can be implicitly created only for scalar outputs
//            System.out.printf("kk after backward : grad: %s ,%s bb grad %s \n",
//                    kk.native_().grad(), kk.grad(), bb.grad());
//            
//            System.out.printf(" kk after backward : grad fn %s native fn %s\n",
//                    kk.grad_fn(), kk.native_().grad_fn());
//            System.out.printf("after backward kk.grad %s\n", kk.grad());
//            System.out.printf("after backward bb.grad %s\n", bb.grad());
//            System.out.printf("after backward dd.grad %s\n", bb.grad());
//        }
//    }
//}
