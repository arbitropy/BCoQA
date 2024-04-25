"""Official evaluation script for BCoQA.

The code is based on CoQA evaluation script.
"""
import argparse
import json
import re
import sys
from datasets import load_dataset

from collections import Counter

OPTS = None


class BCoQAEvaluator():

    def __init__(self, gold_file):
        self.gold_data = BCoQAEvaluator.gold_answers_to_dict(gold_file)

    @staticmethod
    def gold_answers_to_dict(gold_file):
        """
        Converts the gold answers from a JSON file to a dictionary.

        Args:
            gold_file (str): The path to the JSON file of entire dataset.

        Returns:
            dict: A dictionary where the keys are tuples of (story_id, turn_id) and the values are the corresponding gold answers.
        """
        dataset = load_dataset("json", data_files=gold_file)['train']
        gold_dict = {}   
        for story in dataset:
            story_id = story['id']
            answers = story['answers']
            for i, ad in enumerate(answers):
                turn_id = ad['turn_id']
                key = (story_id, turn_id)
                if key in gold_dict:
                    sys.stderr.write("Gold file has duplicate, check id {}".format(story_id))
                gold_dict[key] = ad['answer']
        return gold_dict

    @staticmethod
    def preds_to_dict(pred_file):
        """
        Convert predictions from a JSON file to a dictionary.

        Args:
            pred_file (str): The path to the JSON file containing the predictions.

        Returns:
            dict: A dictionary where the keys are tuples of (id, turn_id) and the values are the corresponding answers.
        """
        preds = json.load(open(pred_file, encoding='utf-8'))
        pred_dict = {}
        for pred in preds:
            pred_dict[(pred['id'], pred['turn_id'])] = pred['answer']
        return pred_dict

    @staticmethod
    def normalize_answer(s):
        """Remove punctuation, storys and extra whitespace."""
        
        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            sentence = re.sub(r'[^\u0980-\u09FF\s]', '', text)
            return sentence

        return white_space_fix(remove_punc(s))

    @staticmethod
    def get_tokens(s):
        if not s: return []
        return BCoQAEvaluator.normalize_answer(s).split()

    @staticmethod
    def compute_exact(a_gold, a_pred):
        return int(BCoQAEvaluator.normalize_answer(a_gold) == BCoQAEvaluator.normalize_answer(a_pred))

    @staticmethod
    def compute_f1(a_gold, a_pred):
        gold_toks = BCoQAEvaluator.get_tokens(a_gold)
        pred_toks = BCoQAEvaluator.get_tokens(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def model_performance(self, pred_data):
        exact_scores, f1_scores = self.get_raw_scores(pred_data)
        return self.get_total_scores(exact_scores, f1_scores)
    
    def get_total_scores(self, exact_scores, f1_scores):
        sum_f1 = 0
        sum_em = 0
        for key in exact_scores:
            sum_f1 += f1_scores[key]
            sum_em += exact_scores[key]
        total = len(exact_scores)
        return {'exact': sum_em/total, 'f1': sum_f1/total}
    
    def get_raw_scores(self, pred_data):
        ''''Returns a dict with score with each turn prediction'''
        exact_scores = {}
        f1_scores = {}
        for story_id, turn_id in self.gold_data:
            key = (story_id, turn_id)
            if key not in pred_data:
                sys.stderr.write('Missing prediction for {} and turn_id: {}\n'.format(story_id, turn_id))
                continue
            a_pred = pred_data[key]
            scores = self.compute_turn_score(story_id, turn_id, a_pred)
            exact_scores[key] = scores['em']
            f1_scores[key] = scores['f1']
        return exact_scores, f1_scores
    
    def compute_turn_score(self, story_id, turn_id, a_pred):
        a_gold = self.gold_data[(story_id, turn_id)]
        return {'em': self.compute_exact(a_gold, a_pred), 'f1': self.compute_f1(a_gold, a_pred)}

def parse_args():
    parser = argparse.ArgumentParser('Official evaluation script for BCoQA.')
    parser.add_argument('--test-file','-t', dest="data_file", help='Input data JSON file.')
    parser.add_argument('--pred-file','-p', dest="pred_file", help='Model predictions.')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    evaluator = BCoQAEvaluator(OPTS.data_file)
    if OPTS.pred_file:
        with open(OPTS.pred_file) as f:
            pred_data = BCoQAEvaluator.preds_to_dict(OPTS.pred_file)
        print(json.dumps(evaluator.model_performance(pred_data), indent=2))

if __name__ == '__main__':
    OPTS = parse_args()
    main()