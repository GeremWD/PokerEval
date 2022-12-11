
import poker_eval


if __name__ == '__main__':
    poker_eval.process_csv(
        csv_path="input.csv",
        output_path="output.csv",
        n_samples=20000,
        n_workers=8
    )