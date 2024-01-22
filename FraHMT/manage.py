from learner.dataset import FragmentDataset,FragmentDataset_transfer
from learner.sampler import Sampler
from learner.trainer import Trainer, save_ckpt
from utils.config import Config
from utils.parser import command_parser
from utils.plots import plot_paper_figures
from utils.preprocess import preprocess_dataset, preprocess_dataset_transfer
from utils.postprocess import postprocess_samples, score_samples, dump_scores
from utils.filesystem import load_dataset
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')
import copy
from learner.skipgram import Vocab
from utils.filesystem import save_pickle, load_pickle

def train_model(config):
    dataset = FragmentDataset(config)
    vocab = dataset.get_vocab()
    trainer = Trainer(config, vocab)
    trainer.train(dataset.get_loader(), 0,config)

def train_model_transfer(config,transfer_run_dir):
    config_transfer = copy.deepcopy(config)
    dataset_vocab = FragmentDataset_transfer(config_transfer)
    vocab = dataset_vocab.get_vocab()
    dataset = FragmentDataset(config)
    dataset.get_vocab()
    load_last = config.get('load_last')
    trainer, epoch = Trainer.load_transfer(config, vocab, last=load_last,trans_rundir=transfer_run_dir)
    trainer.train(dataset.get_loader(), 0,config)


def sample_model(config, num_sample):
    dataset = FragmentDataset(config)
    vocab = dataset.get_vocab()
    load_last = config.get('load_last')
    trainer, epoch = Trainer.load(config, vocab, last=load_last)
    sampler = Sampler(config, vocab, trainer.model)
    seed = config.get('sampling_seed') if config.get('reproduce') else None
    samples = sampler.sample(num_sample, seed=seed)
    dataset = load_dataset(config, kind="test")
    _, scores = score_samples(samples, dataset)
    is_max = dump_scores(config, scores, epoch)
    if is_max:
        save_ckpt(trainer, epoch, filename=f"best.pt")
    config.save()

if __name__ == "__main__":
    parser = command_parser()
    args = vars(parser.parse_args())
    command = args.pop('command')

    if command == 'preprocess':
        dataset = args.pop('dataset')
        n_jobs = args.pop('n_jobs')
        preprocess_dataset(dataset, n_jobs)
    
    elif command == 'preprocess_transfer':
        dataset = args.pop('dataset')
        n_jobs = args.pop('n_jobs')
        preprocess_dataset_transfer(dataset, n_jobs)

    elif command == 'train':
        config = Config(args.pop('dataset'), **args)
        train_model(config)

    elif command == 'train_transfer':
        config = Config(args.pop('dataset'), **args)
        transfer_run_dir = "./RUNS/XXX-ZINC/ckpt/best_valid.pt"
        train_model_transfer(config,transfer_run_dir)

    elif command == 'sample':
        args.update(use_gpu=False)
        run_dir = args.pop('run_dir')
        config = Config.load(run_dir, **args)
        num_sample = 10000
        sample_model(config, num_sample)

    elif command == 'postprocess':
        run_dir = args.pop('run_dir')
        config = Config.load(run_dir, **args)
        postprocess_samples(config, **args)
    
    elif command == 'plot':
        run_dir = args.pop('run_dir')
        plot_paper_figures(run_dir)