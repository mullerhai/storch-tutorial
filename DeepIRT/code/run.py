'''
Deep-IRT: Make Deep Learning Based Knowledge Tracing Explainable Using Item Response Theory

python3.8
torch==1.4.0
tqdm==4.48.2
'''
import os
import random
import logging
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
from dataloader import getDataLoader
import eval
from parse import parse_args
from model import MODEL


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    args = parse_args()

    length = args.length
    questions = args.questions
    lr = args.lr
    bs = args.batch_size
    seed = args.seed
    epochs = args.epochs
    cuda = args.cuda
    final_fc_dim = args.final_fc_dim
    question_dim = args.question_dim
    question_and_answer_dim = args.question_and_answer_dim
    memory_size = args.memory_size
    model_type = args.model

    logger = logging.getLogger('main')
    logger.setLevel(level=logging.DEBUG)
    date = datetime.now()
    handler = logging.FileHandler(
        f'log/{date.year}_{date.month}_{date.day}_{model_type}_result.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info('DeepIRT')
    logger.info(args)

    setup_seed(seed)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    trainLoader, validationLoader, testLoader = getDataLoader(bs, questions, length)

    model = MODEL(n_question=questions, batch_size=bs, q_embed_dim=question_dim, qa_embed_dim=question_and_answer_dim,
                  memory_size=memory_size, final_fc_dim=final_fc_dim)
    model.init_params()
    model.init_embeddings()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_auc = 0
    for epoch in range(epochs):
        print('epoch: ' + str(epoch+1))
        model, optimizer = eval.train_epoch(model, trainLoader, optimizer, device)
        logger.info(f'epoch {epoch+1}')
        auc = eval.test_epoch(model, validationLoader, device)
        if auc > best_auc:
            print('best checkpoint')
            torch.save({'state_dict': model.state_dict()}, 'checkpoint/'+model_type+'.pth.tar')
            best_auc = auc
    eval.test_epoch(model, testLoader, device, ckpt='checkpoint/'+model_type+'.pth.tar')


if __name__ == '__main__':
    main()
