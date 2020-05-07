#!/usr/bin/env python

#  This code will read an input csv file and break specified text fields into words and n-grams.
#  Basic sentiment scores will also be assigned to each word and n-gram.
#  Stop words will also be identified in the word breakout.
#  Two csv files (one for words and one for n-grams) will be written.

#  To add more stop words, edit the language file in nltk_data\corpora\stopwords

from collections import defaultdict
import csv
from functools import lru_cache
import math
import os
import re
import sys
import argparse

import nltk
from nltk.corpus import stopwords 
from nltk.sentiment import sentiment_analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import SnowballStemmer 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.translate.gdfa import grow_diag_final_and
from nltk.util import ngrams
from pip._vendor.requests.api import head

from logging_service import LoggingService

class TextAnalyzer(object):
    
    # How many records to process before
    # a progress report:
    PROGRESS_EVERY = 1000

    stopWordLanguageList = ['arabic','azerbaijani','danish','dutch',
                            'english','finnish','french','german','greek',
                            'hungarian','indonesian','italian','kazakh',
                            'nepali','norwegian','portuguese','romanian',
                            'russian','slovene','spanish','swedish','tajik','turkish'
                            ]
    stopword_language = 'english'
    
    word_file_suffix = '_wordstats'
    ngram_file_suffix = '_ngrams'

    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, 
                 text_file,
                 text_fields,
                 outfiles_dir=None,
                 record_id_col='id',
                 stopword_language='english',
                 ngram_len=3,
                 separator=','
                 ):

        self.log = LoggingService()
        if outfiles_dir is not None:
            if not os.path.exists(outfiles_dir):
                os.mkdir(outfiles_dir)
            elif not os.path.isdir(outfiles_dir):
                print(f"Outfile directory is a file: {outfiles_dir}; quitting")
                sys.exit(1)

        self.text_file = text_file
        self.outfiles_dir = outfiles_dir
        self.text_fields = text_fields
        self.record_id_col = record_id_col
        self.stopword_language = stopword_language
        self.ngram_len = ngram_len
        self.separator = separator

        self.create_wordnet_nltk_pos_xlation()
        # Generate outfile names for word stats,
        # and ngrams:

        if outfiles_dir is None:
            # Output result files to same dir as infile:
            (infile_path, ext) = os.path.splitext(text_file)
            wordstats_path = f"{infile_path}{self.word_file_suffix}.csv"
            ngrams_path = f"{infile_path}{self.ngram_file_suffix}.csv"
        else:
            # Outdir different from infile's:
            wordstats_path = os.path.join(outfiles_dir,
                                          f"{os.path.basename(text_file)}{self.word_file_suffix}.csv")

        self.log.info(f"Infile: {text_file}")
        self.log.info(f"Word stats will be in {wordstats_path}")
        self.log.info(f"NGrams will be in {ngrams_path}")
        self.generate_outfiles(text_file,
                               wordstats_path,
                               ngrams_path)

    #------------------------------------
    # generate_outfiles
    #-------------------
    
    def generate_outfiles(self, 
                          text_file,
                          word_stats_path,
                          ngrams_path):
        '''
        Generates both word stats and ngrams outfile.
        
        Assumptions about inst var initializations:
            o text_fields,
            o record_id_col,
            o stopword_language,
            o ngram_len
            o pos_tag_map
             
        @param text_file:
        @type text_file:
        @param word_stats_path output path for word stats
        @type word_stats_path: str
        @param ngrams_path: output path for ngrams
        @type ngrams_path: str
        '''

        try:
            # Open outfiles:
            word_stats_fd = open(word_stats_path, 'w', newline='')
            ngram_fd      = open(ngrams_path, 'w', newline='')
            
            # The words stats .csv writer
            word_stats_cols = [self.record_id_col,
                               'word',
                               'stem',
                               'lemmatized',
                               'pos',
                               'sent_neg',
                               'sent_neu',
                               'sent_pos',
                               'sent_compound',
                               'stop_word',
                               'word_number']
            word_stats_writer = csv.DictWriter(word_stats_fd, word_stats_cols)
            word_stats_writer.writeheader()

            # The ngrams .csv writer. The col header depends
            # on the number of ngrams requested:
            #   id,Word1,Word2,...,full_ngram, ngram_sentiment, ngram_number
            
            self.word_col_names =[f"Word{indx+1}" for indx in range(self.ngram_len)]
            heading = [self.record_id_col]
            heading.extend(self.word_col_names)

            self.log.info("Creating NLTK tool instances...")
            heading.extend(['full_ngram', 'ngram_sentiment', 'ngram_number'])
            ngrams_writer     = csv.DictWriter(ngram_fd, heading)
            ngrams_writer.writeheader()
            
            # The NLTK tools:
            tokenizer  = TweetTokenizer(preserve_case=False)
            stemmer    = SnowballStemmer(self.stopword_language)
            # Speed up stemming:
            self.stem = lru_cache(maxsize=50000)(stemmer.stem)
            lemmatizer = WordNetLemmatizer()
            # Speed up lemmatization by caching:
            self.lemmatize = lru_cache(maxsize=50000)(lemmatizer.lemmatize)
            # Create the sentiment analyzer for some basic sentiment tests.
            sentiment_analyzer = SentimentIntensityAnalyzer()

            self.log.info("Done creating NLTK tool instances.")
            # List of all_stopwords:
            self.all_stopwords = set(stopwords.words(self.stopword_language)) 
            
            # Regex to keep only alpha and apostrophe.
            # Eliminates punctuation, but keeps contractions,
            # such as "can't":
            spec_char_pat = re.compile(r"[\w']+")

            # How many records since last progress
            # report:
            records_since_prog_rep = 0
            
            record_num = 0
            with open(text_file, 'r') as in_fd:
                csv_reader = csv.DictReader(in_fd, delimiter=self.separator)
                for row_dict in csv_reader:
                    record_num += 1
                    records_since_prog_rep += 1
                    for txt_field in self.text_fields:
                        text = row_dict[txt_field]
                        if text is None:
                            continue
                        # Tokenize:
                        token_arr = tokenizer.tokenize(text)
                        # Remove punctuation:
                        clean_token_arr = [kept_token for kept_token \
                                           in token_arr \
                                           if spec_char_pat.match(kept_token) \
                                           is not None]
                        self.write_ngrams(record_num, 
                                          clean_token_arr, 
                                          ngrams_writer, 
                                          sentiment_analyzer, 
                                          self.ngram_len)
                        self.write_word_stats(record_num,
                                              row_dict,
                                              text,
                                              clean_token_arr, 
                                              word_stats_writer, 
                                              sentiment_analyzer)
                    # Time for progress report?
                    if records_since_prog_rep >= self.PROGRESS_EVERY:
                        self.log.info(f"Processed {record_num} input file records.")
                        records_since_prog_rep = 0
        finally:
            word_stats_fd.close()
            ngram_fd.close()
                
    #------------------------------------
    # write_word_stats 
    #-------------------
    
    def write_word_stats(self,
                         record_num,
                         row_dict,
                         text,
                         clean_token_arr, 
                         word_stats_writer, 
                         sentiment_analyzer):
        '''
        Compute and write to file all word stats
        of one record.
        
        @param record_num: record count of this record
        @type record_num: int
        @param row_dict: dict returned from csv reader; one key
            per column
        @type row_dict: {str : str}
        @param text: text to analyze; one field of one in file record
        @type text: str
        @param clean_token_arr: tokenized array of the text
        @type clean_token_arr: [str]
        @param word_stats_writer: csv dict writer for output
        @type word_stats_writer: csv.DictWriter
        @param sentiment_analyzer: sentiment analysis NLTK instance
        @type sentiment_analyzer: SentimentIntensityAnalyzer
        '''
                
        # First, get sentiment neg/neu/pos/compount for
        # the text:
        sent_dict = sentiment_analyzer.polarity_scores(text)
        sent_arr  = [sent_dict['neg'],
                     sent_dict['neu'],
                     sent_dict['pos'],
                     sent_dict['compound'],
                     ]

        stem_arr  = [self.stem(word) for word in clean_token_arr]
        pos_tuples = nltk.pos_tag(clean_token_arr)
        lem_arr   = [self.lemmatize(word, pos=self.pos_tag_map[pos]) \
                     for (word,pos) in pos_tuples]
        pos_arr   = [pos for (word,pos) in pos_tuples]
        stopword_status_arr = [word in self.all_stopwords for word in clean_token_arr]

        rec_id = row_dict[self.record_id_col]
        # Finally, put the rows together, a row for each value:
        out_row = {}
        for (i, token) in enumerate(clean_token_arr):
            out_row[self.record_id_col] = rec_id
            out_row['word'] = token
            out_row['stem'] = stem_arr[i]
            out_row['lemmatized'] = lem_arr[i]
            out_row['pos'] = pos_arr[i]
            out_row['sent_neg'] = sent_dict['neg'] 
            out_row['sent_neu'] = sent_dict['neu']
            out_row['sent_pos'] = sent_dict['pos']
            out_row['sent_compound'] = sent_dict['compound']
            out_row['stop_word'] = stopword_status_arr[i]
            out_row['word_number'] = record_num
            
            word_stats_writer.writerow(out_row)

    #------------------------------------
    # write_ngrams
    #-------------------

    
    def write_ngrams(self, row_id, clean_token_arr, ngram_writer, nltkSentiment, ngram_len):
        
        ngram_tuples = list(ngrams(clean_token_arr, ngram_len))
        
        for (i, ngram_tuple) in enumerate(ngram_tuples):
            out_dict = dict(list(zip(self.word_col_names, ngram_tuple)))
            full_ngram = ' '.join(ngram_tuple)
            out_dict['full_ngram'] = full_ngram
            out_dict['ngram_sentiment'] = nltkSentiment.polarity_scores(full_ngram)['compound']
            out_dict['ngram_number'] = str(i)
            out_dict[self.record_id_col] = row_id
            ngram_writer.writerow(out_dict)
        
    
    #------------------------------------
    # create_wordnet_nltk_pos_xlation
    #-------------------
    
    def create_wordnet_nltk_pos_xlation(self):
        '''
        Wordnet has different part of speech tags
        than NLTK. Build a dict from one to the other.
        When in doubt, make pos be a noun.
        '''
        self.pos_tag_map = defaultdict(lambda : 'n')
        self.pos_tag_map['J'] = 'a'
        self.pos_tag_map['V'] = 'v'
        self.pos_tag_map['R'] = 'r'

            
# ----------------------------- Main ----------------

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Here is what this package does."
                                     )

    parser.add_argument('-d', '--outdir',
                        help='directory for output; default: same as infile',
                        default=None)
    parser.add_argument('-n', '--ngramlen',
                        type=int,
                        help='Ngram order: How many words per ngram; default 3',
                        default=3)
    parser.add_argument('-i', '--idcol',
                        help="name of column that holds unique id; default: 'id'",
                        default='id')
    parser.add_argument('-l', '--language',
                        help="language of text; default: 'english'",
                        default='english')
    parser.add_argument('-s', '--separator',
                        help="field separator char in input file; default is comma.",
                        default=',')
    parser.add_argument('input_file',
                        help='CSV input file')
    parser.add_argument('columns',
                        type=str,
                        nargs='+',
                        help="Column names that contain text to be analyzed; space separated list")

    args = parser.parse_args();
    

    if not os.path.exists(args.input_file):
        print(f"Input file '{args.input_file}' does not exist")
        sys.exit(1)

#     text_file = '/tmp/propub_msgs.csv'
#     outfiles_dir = os.path.dirname(text_file)
#     text_fields = ['message']
#     record_id_col = 'id'
    
    TextAnalyzer(args.input_file,
                 args.columns,
                 outfiles_dir=args.outdir,
                 record_id_col=args.idcol,
                 stopword_language=args.language,
                 ngram_len=args.ngramlen,
                 separator=args.separator
                 )

