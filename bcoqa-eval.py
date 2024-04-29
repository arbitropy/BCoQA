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
        """Remove punctuation, unnecessary words for f1 scoring and extra whitespace."""

        def remove_punc(text):
            sentence = re.sub(r'[^\u0980-\u09FF\s]', '', text)
            return sentence
        
        def remove_stopwords_whitespace(text):
            with open('stopwords.txt', 'r', encoding='utf-8') as f:
                stopwords = [word.strip() for word in f.readlines()]
            return ' '.join([word for word in text.split() if word not in stopwords])

        return remove_stopwords_whitespace(remove_punc(s))

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
        """
        Calculates the performance metrics of the model based on the predicted data.

        Args:
            pred_data (dict): A dictionary containing the predicted data.

        Returns:
            dict: A dictionary containing the performance metrics including average exact match (average_em),
                  average F1 score (average_f1), F1 score for yes/no answers (yesno_f1), F1 score for unknown answers (unknown_f1),
                  exact match score for yes/no answers (yesno_em), and exact match score for unknown answers (unknown_em).
        """
        exact_scores, f1_scores = self.get_raw_scores(pred_data)
        total_scores = self.get_total_scores(exact_scores, f1_scores)
        yesno_count = 0
        unknown_count = 0
        sum_yesno_em = 0
        sum_unknown_em = 0
        small_count = 0
        small_count_f1 = 0
        long_count = 0
        long_count_f1 = 0
        for key in exact_scores.keys():
            if self.gold_data[key] == 'হ্যাঁ।' or self.gold_data[key] == 'না।':
                yesno_count += 1
                sum_yesno_em += exact_scores[key]
            elif self.gold_data[key] == 'অজানা।':
                unknown_count += 1
                sum_unknown_em += exact_scores[key]
            elif len(self.gold_data[key].split()) < 3:
                small_count += 1
                small_count_f1 += f1_scores[key]
            else:
                long_count += 1
                long_count_f1 += f1_scores[key]
        return {'average_em': total_scores['exact'], 
                'average_f1': total_scores['f1'], 
                'yesno_em': sum_yesno_em/yesno_count, 
                'unknown_em': sum_unknown_em/unknown_count,
                'small_f1': small_count_f1/small_count,
                'long_f1': long_count_f1/long_count}
    
    def get_total_scores(self, exact_scores, f1_scores):
        """
        Calculates the total scores for exact match (EM) and F1 score.

        Args:
            exact_scores (dict): A dictionary containing the exact match scores for each key.
            f1_scores (dict): A dictionary containing the F1 scores for each key.

        Returns:
            dict: A dictionary containing the total scores for exact match (EM) and F1 score.
                The keys are 'exact' and 'f1', and the values are the average scores.

        """
        sum_f1 = 0
        sum_em = 0
        for key in exact_scores:
            sum_f1 += f1_scores[key]
            sum_em += exact_scores[key]
        total = len(exact_scores)
        return {'exact': sum_em/total, 'f1': sum_f1/total}
    
    def get_raw_scores(self, pred_data):
        """
        Computes the raw scores for each prediction in the given `pred_data`.

        Args:
            pred_data (dict): A dictionary containing the predicted data.

        Returns:
            tuple: A tuple containing two dictionaries - `exact_scores` and `f1_scores`.
                - `exact_scores`: A dictionary mapping (story_id, turn_id) tuples to exact match scores.
                - `f1_scores`: A dictionary mapping (story_id, turn_id) tuples to F1 scores.
        """
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
        with open("result.json", "w") as f:
            result = json.dumps(evaluator.model_performance(pred_data), indent=2)
            f.write(result)
            print(result)

if __name__ == '__main__':
    OPTS = parse_args()
    main()
