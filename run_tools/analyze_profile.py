import pstats

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-lines", type=int, default=20, help="Number of lines to print"
    )
    parser.add_argument(
        "--sort-by", type=str, default="cumulative", help="Sort by column"
    )
    parser.add_argument(
        "--name-filter", type=str, default=None, help="Filter by function name"
    )
    parser.add_argument("logFile", nargs=1, type=str)
    args = parser.parse_args()

    p = pstats.Stats(args.logFile[0])
    p.sort_stats(args.sort_by).print_stats(args.n_lines, args.name_filter)
