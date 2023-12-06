from guides import *
from settings import get_settings

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Choose pipeline stage.')
    parser.add_argument('module', type=str,
                        help='Which stage of the pipeline. Options are setup, search, evidence, posterior.')
    parser.add_argument('--gpu', default=None,
                    help='GPU index if provided.')
                    
    args = parser.parse_args()

    settings = get_settings()
    gpu = int(args.gpu)

    if args.module == "setup":
        runner = TriplesSetupGuide(settings, batch_size=1000, gpu=gpu)
    
    elif args.module == "search":
        runner = SearchRuns(settings, gpu=gpu)
    
    elif args.module == "evidence":
        runner = EvidenceRuns(settings, gpu=gpu) 
    
    elif args.module == "posterior":
        runner = PosteriorRuns(settings, gpu=gpu) 

    runner.run_directory_list()