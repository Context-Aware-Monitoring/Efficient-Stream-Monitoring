"""Generates context.csv files that are used by contextual bandits.
"""
import sys
import csv
import context

def _get_headers_from_csv_file(filepath):
    headers = None
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)

    return headers


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        print("Possible arguments are --context --all")
    args = args[1:]
    if "--all" in args:
        args = ['--context']

    for arg in args:
        if arg == '--context':
            context.generate_context_csv(
                context.HostExtractor(),
                ['../../data/raw/sequential_data/traces/boot_delete/'],
                outdir='../../data/processed/'
            )
            context_filepath = context.generate_context_csv(
                context.WorkloadExtractor(),
                [
                    '../../data/raw/sequential_data/traces/boot_delete/',
                    '../../data/raw/sequential_data/traces/image_create_delete/',
                    '../../data/raw/sequential_data/traces/network_create_delete/'
                ],
                outdir='../../data/processed/'
            )

            context_filepath = '../../data/processed/context_workload-extractor-start-stop_s20191119-183839_e20191120-013000_sbd-sicd-sncd_w30_s5.csv'

            reward_filepath = '../../data/processed/seq_rewards_w30_s5_MI_n1.csv'

            arms = _get_headers_from_csv_file(reward_filepath)
            context_header = _get_headers_from_csv_file(context_filepath)

            context_transformer = context.PushContextTransformer(
                arms,
                context_header
            )
            context.transform_context_csv(
                context_filepath, context_transformer, '../../data/processed/context_push_active_hosts.csv')

            context_distance_transformer = context.PushContextDistanceBasedTransformer(
                arms,
                context_header
            )

            for d in [50, 100, 200, 500, 1000]:
                context_distance_transformer.set_distance(d)
                context.transform_context_csv(context_filepath, context_distance_transformer,
                                              '../../data/processed/context_push_distance_based_d%d.csv' % d)

        else:
            sys.exit('Invalid argument %s found' % arg)
