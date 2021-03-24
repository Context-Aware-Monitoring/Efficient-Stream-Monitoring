"""Generates context.csv files that are used by contextual bandits.
"""
import sys
import context

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
        else:
            sys.exit('Invalid argument %s found' %arg)
