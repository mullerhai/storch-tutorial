import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go DeepIRT")

    parser.add_argument('--length', type=int, default=50, help=' max length of question sequenc')
    parser.add_argument('--questions', type=int, default=100, help='num of question ')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--seed', type=int, default=59, help='random seed')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--cuda', type=str, default='0', help='use GPU id')
    parser.add_argument('--final_fc_dim', type=int, default=10, help='dimension of final dim')
    parser.add_argument('--question_dim', type=int, default=50, help='dimension of question dim')
    parser.add_argument('--question_and_answer_dim', type=int, default=100, help='dimension of question and answer dim')
    parser.add_argument('--memory_size', type=int, default=20, help='memory size')
    parser.add_argument('--model', type=str, default='DeepIRT', help='model')

    return parser.parse_args()
